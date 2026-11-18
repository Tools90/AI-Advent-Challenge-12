package main

import (
	"context"
	"os"
	"strings"
	"testing"
)

type fakeLLM struct {
	calls int
}

func (f *fakeLLM) Chat(ctx context.Context, messages []Message) (LLMResponse, error) {
	f.calls++
	content := "ok"
	if len(messages) > 0 && messages[0].Role == "system" {
		system := messages[0].Content
		switch {
		case strings.Contains(system, "обновляешь долгосрочный профиль"):
			content = `{"style":"detailed"}`
		case strings.Contains(system, "проверяешь, соблюдён"):
			content = `{"compliant":true,"reason":"ok"}`
		}
	}
	return LLMResponse{
		Message: Message{Role: "assistant", Content: content},
		Usage:   Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
	}, nil
}

func (f *fakeLLM) Model() string { return "fake" }

func TestBuildRequestProfileAndLayers(t *testing.T) {
	profile := &Profile{
		ID: "p1", Name: "Краткий", Style: "concise", Format: "without_code", Tone: "formal",
		Memory: MemoryState{
			ShortTerm: []Message{
				{Role: "user", Content: "u1"}, {Role: "assistant", Content: "a1"},
				{Role: "user", Content: "u2"}, {Role: "assistant", Content: "a2"},
				{Role: "user", Content: "u3"}, {Role: "assistant", Content: "a3"},
			},
			Working:  []MemoryItem{{ID: "w1", Layer: LayerWorking, Value: "бюджет 2,5 млн", Pinned: true}},
			LongTerm: []MemoryItem{{ID: "l1", Layer: LayerLong, Key: "Android", Value: "после MVP"}},
		},
	}
	request, stats := BuildRequest(profile.Memory, "Что решили по Android?", profile)
	if len(request) != 7 {
		t.Fatalf("request messages = %d, want 7", len(request))
	}
	system := request[0].Content
	if !strings.Contains(system, "User profile") || !strings.Contains(system, "concise") || !strings.Contains(system, "2,5 млн") || !strings.Contains(system, "Android") {
		t.Errorf("system = %s", system)
	}
	if stats.ProfileTokens == 0 || stats.ShortTokens == 0 || stats.WorkingTokens == 0 || stats.LongTokens == 0 {
		t.Errorf("stats = %+v", stats)
	}
}

func TestSeedProfilesAndSeparateMemory(t *testing.T) {
	store, err := NewProfileStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := store.SeedProfiles(); err != nil {
		t.Fatal(err)
	}
	state := store.Get()
	if len(state.Profiles) < 3 || state.ActiveID != "p1" {
		t.Fatalf("state = %+v", state)
	}
	if _, err := store.AddItem("p1", MemoryItem{Layer: LayerLong, Key: "a", Value: "one"}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.AddItem("p2", MemoryItem{Layer: LayerLong, Key: "b", Value: "two"}); err != nil {
		t.Fatal(err)
	}
	state = store.Get()
	p1 := findProfile(state, "p1")
	p2 := findProfile(state, "p2")
	if !hasKey(p1.Memory.LongTerm, "a") || !hasKey(p2.Memory.LongTerm, "b") || hasKey(p1.Memory.LongTerm, "b") {
		t.Errorf("memory not separate")
	}
}

func hasKey(items []MemoryItem, key string) bool {
	for _, item := range items {
		if item.Key == key {
			return true
		}
	}
	return false
}

func TestChatAutoProfileAndShortTerm(t *testing.T) {
	store, err := NewProfileStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := store.SeedProfiles(); err != nil {
		t.Fatal(err)
	}
	llm := &fakeLLM{}
	result, err := store.Chat(context.Background(), llm, "p1", "Отвечай кратко")
	if err != nil {
		t.Fatal(err)
	}
	if !result.ProfileUpdated || result.Profile.Style != "detailed" {
		t.Errorf("profile update = %+v", result)
	}
	profile := findProfile(store.Get(), "p1")
	if len(profile.Memory.Messages) != 2 || len(profile.Memory.ShortTerm) != 2 {
		t.Errorf("messages = %+v", profile.Memory)
	}
	if profile.Memory.ProfileUpdateUsage.TotalTokens == 0 {
		t.Error("profile update usage not counted")
	}
}

func TestClearShortKeepsChatAndOtherLayers(t *testing.T) {
	store, err := NewProfileStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := store.SeedProfiles(); err != nil {
		t.Fatal(err)
	}
	if _, err := store.AddItem("p1", MemoryItem{Layer: LayerLong, Key: "a", Value: "one"}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.AddItem("p1", MemoryItem{Layer: LayerWorking, Value: "task"}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Chat(context.Background(), &fakeLLM{}, "p1", "привет"); err != nil {
		t.Fatal(err)
	}
	if err := store.ClearLayer("p1", LayerShort); err != nil {
		t.Fatal(err)
	}
	profile := findProfile(store.Get(), "p1")
	if len(profile.Memory.Messages) == 0 || len(profile.Memory.ShortTerm) != 0 {
		t.Errorf("short clear changed chat: %+v", profile.Memory)
	}
	if len(profile.Memory.Working) == 0 || len(profile.Memory.LongTerm) == 0 {
		t.Errorf("other layers cleared")
	}
}

func TestProfileJudge(t *testing.T) {
	store, err := NewProfileStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := store.SeedProfiles(); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Chat(context.Background(), &fakeLLM{}, "p1", "привет"); err != nil {
		t.Fatal(err)
	}
	result, err := store.JudgeProfile(context.Background(), &fakeLLM{}, "p1", "вопрос", "ответ")
	if err != nil {
		t.Fatal(err)
	}
	if !result.Compliant || result.Usage.TotalTokens == 0 {
		t.Errorf("judge = %+v", result)
	}
	profile := findProfile(store.Get(), "p1")
	if profile.Memory.Turns[0].ProfileJudge == nil {
		t.Error("profile judge not attached")
	}
}

func TestProfileStorePersistence(t *testing.T) {
	dir := t.TempDir()
	store, err := NewProfileStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.SeedProfiles(); err != nil {
		t.Fatal(err)
	}
	if err := store.UpdateProfile("p1", "Test", "detailed", "with_code", "informal", []string{"x"}, []string{"y"}); err != nil {
		t.Fatal(err)
	}
	reloaded, err := NewProfileStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	profile := findProfile(reloaded.Get(), "p1")
	if profile == nil || profile.Style != "detailed" || profile.Name != "Test" {
		t.Errorf("profile = %+v", profile)
	}
}

func TestClearAllRestoresFirstRun(t *testing.T) {
	dir := t.TempDir()
	store, err := NewProfileStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.SeedProfiles(); err != nil {
		t.Fatal(err)
	}
	if _, err := store.AddItem("p1", MemoryItem{Layer: LayerLong, Key: "x", Value: "y"}); err != nil {
		t.Fatal(err)
	}
	if err := store.ClearAll(); err != nil {
		t.Fatal(err)
	}
	state := store.Get()
	if !store.Fresh() || state.ActiveID != "p1" || len(state.Profiles) != 3 {
		t.Fatalf("state = %+v", state)
	}
	for _, p := range state.Profiles {
		if p.Style != "" || p.Format != "" || p.Tone != "" || len(p.Memory.Messages) != 0 || len(p.Memory.ShortTerm) != 0 || len(p.Memory.Working) != 0 || len(p.Memory.LongTerm) != 0 {
			t.Fatalf("profile not reset: %+v", p)
		}
	}
	if _, err := os.Stat(dir + string(os.PathSeparator) + "profiles.json"); !os.IsNotExist(err) {
		t.Fatalf("profiles.json still exists: %v", err)
	}
}

func TestClearSessionKeepsWorkingAndLongTerm(t *testing.T) {
	store, err := NewProfileStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := store.SeedProfiles(); err != nil {
		t.Fatal(err)
	}
	llm := &fakeLLM{}
	if _, err := store.Chat(context.Background(), llm, "p1", "привет"); err != nil {
		t.Fatal(err)
	}
	before := findProfile(store.Get(), "p1").Memory
	if len(before.Working) == 0 || len(before.LongTerm) == 0 || len(before.Messages) == 0 {
		t.Fatalf("seed/session missing: %+v", before)
	}
	if err := store.ClearSession("p1"); err != nil {
		t.Fatal(err)
	}
	after := findProfile(store.Get(), "p1").Memory
	if len(after.Working) != len(before.Working) || len(after.LongTerm) != len(before.LongTerm) {
		t.Fatalf("working/long changed: before=%+v after=%+v", before, after)
	}
	if len(after.Messages) != 0 || len(after.ShortTerm) != 0 || len(after.Turns) != 0 {
		t.Fatalf("session not cleared: %+v", after)
	}
}

func TestOneOffRequestDoesNotUpdateProfile(t *testing.T) {
	store, err := NewProfileStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := store.SeedProfiles(); err != nil {
		t.Fatal(err)
	}
	result, err := store.Chat(context.Background(), &fakeLLM{}, "p1", "Покажи пример.")
	if err != nil {
		t.Fatal(err)
	}
	if result.ProfileUpdated {
		t.Fatalf("profile updated by one-off request: %+v", result.Profile)
	}
	profile := findProfile(store.Get(), "p1")
	if profile.Style != "concise" || profile.Format != "without_code" || profile.Name != "Кратко / без кода / формально" {
		t.Fatalf("profile changed: %+v", profile)
	}
}

func TestGeneratedProfileNameFollowsFields(t *testing.T) {
	store, err := NewProfileStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := store.SeedProfiles(); err != nil {
		t.Fatal(err)
	}
	if err := store.UpdateProfile("p1", "Кратко / без кода / формально", "concise", "with_code", "formal", nil, nil); err != nil {
		t.Fatal(err)
	}
	profile := findProfile(store.Get(), "p1")
	if profile.Name != "Кратко / с кодом / формально" {
		t.Fatalf("name = %q", profile.Name)
	}
}
