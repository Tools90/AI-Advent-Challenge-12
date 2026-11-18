package main

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

const (
	LayerShort   = "short"
	LayerWorking = "working"
	LayerLong    = "long"
	ShortWindow  = 5
	RelevantTopK = 3

	systemPrompt        = "Ты — персонализированный ассистент. Соблюдай профиль пользователя и используй память, но не выдумывай факты."
	profileUpdatePrompt = "Ты обновляешь долгосрочный профиль пользователя. Меняй поле только если пользователь явно просит изменить будущий стиль, формат или тон ответов (например: «отвечай кратко», «в дальнейшем с кодом», «пиши на ты», «всегда без кода»). Разовые просьбы к текущему ответу («покажи пример», «приведи код», «объясни подробнее», «сделай таблицу») НЕ меняют профиль. style: concise = кратко, detailed = подробно. format: with_code = с кодом, without_code = без кода. tone: formal = формально/на вы, informal = неформально/на ты. Если явного указания на будущий формат нет, верни {}. Верни только JSON без markdown."
	profileJudgePrompt  = "Ты проверяешь, соблюдён ли профиль пользователя в ответе. Учитывай стиль, формат, тон и ограничения. Верни только JSON: {\"compliant\": true|false, \"reason\": \"кратко\"}."
)

type MemoryItem struct {
	ID        string `json:"id"`
	Layer     string `json:"layer"`
	Key       string `json:"key"`
	Value     string `json:"value"`
	Pinned    bool   `json:"pinned,omitempty"`
	CreatedAt string `json:"createdAt,omitempty"`
}

type Profile struct {
	ID          string      `json:"id"`
	Name        string      `json:"name"`
	Style       string      `json:"style,omitempty"`
	Format      string      `json:"format,omitempty"`
	Tone        string      `json:"tone,omitempty"`
	Constraints []string    `json:"constraints,omitempty"`
	Preferences []string    `json:"preferences,omitempty"`
	Memory      MemoryState `json:"memory"`
}

type MemoryState struct {
	Version            int          `json:"version"`
	Messages           []Message    `json:"messages,omitempty"`
	ShortTerm          []Message    `json:"shortTerm,omitempty"`
	Turns              []TurnMeta   `json:"turns,omitempty"`
	Working            []MemoryItem `json:"working,omitempty"`
	LongTerm           []MemoryItem `json:"longTerm,omitempty"`
	ChatUsage          Usage        `json:"chatUsage"`
	JudgeUsage         Usage        `json:"judgeUsage"`
	ProfileUpdateUsage Usage        `json:"profileUpdateUsage"`
}

type ProfileState struct {
	Version  int        `json:"version"`
	ActiveID string     `json:"activeId"`
	Profiles []*Profile `json:"profiles"`
}

type ProfileStore struct {
	mu    sync.RWMutex
	path  string
	state ProfileState
	fresh bool
}

type LayerStats struct {
	ShortChars    int `json:"shortChars"`
	ShortTokens   int `json:"shortTokens"`
	WorkingChars  int `json:"workingChars"`
	WorkingTokens int `json:"workingTokens"`
	LongChars     int `json:"longChars"`
	LongTokens    int `json:"longTokens"`
	ProfileChars  int `json:"profileChars"`
	ProfileTokens int `json:"profileTokens"`
}

type JudgeInfo struct {
	Correct bool   `json:"correct"`
	Reason  string `json:"reason,omitempty"`
}

type TurnMeta struct {
	Request            []Message  `json:"request,omitempty"`
	PromptTokens       int        `json:"promptTokens"`
	CompletionTokens   int        `json:"completionTokens"`
	SessionTotal       int        `json:"sessionTotal"`
	LayerStats         LayerStats `json:"layerStats"`
	ProfileUpdate      bool       `json:"profileUpdate,omitempty"`
	ProfileUpdateUsage Usage      `json:"profileUpdateUsage,omitempty"`
	ProfileJudge       *JudgeInfo `json:"profileJudge,omitempty"`
}

type ChatResult struct {
	Reply              string       `json:"reply"`
	Request            []Message    `json:"request"`
	TurnUsage          Usage        `json:"turnUsage"`
	ChatUsage          Usage        `json:"chatUsage"`
	JudgeUsage         Usage        `json:"judgeUsage"`
	ProfileUpdateUsage Usage        `json:"profileUpdateUsage"`
	TotalUsage         Usage        `json:"totalUsage"`
	LayerStats         LayerStats   `json:"layerStats"`
	Profile            *Profile     `json:"profile,omitempty"`
	ProfileUpdated     bool         `json:"profileUpdated"`
	Working            []MemoryItem `json:"working,omitempty"`
	LongTerm           []MemoryItem `json:"longTerm,omitempty"`
}

type JudgeResult struct {
	Compliant    bool   `json:"compliant"`
	Reason       string `json:"reason"`
	Usage        Usage  `json:"usage"`
	SessionTotal Usage  `json:"sessionTotal"`
}

func NewProfileStore(dir string) (*ProfileStore, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("создание %s: %w", dir, err)
	}
	store := &ProfileStore{path: filepath.Join(dir, "profiles.json")}
	data, err := os.ReadFile(store.path)
	if err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			return nil, fmt.Errorf("чтение %s: %w", store.path, err)
		}
		store.state = newProfileState()
		store.fresh = true
		if err := store.saveLocked(); err != nil {
			return nil, err
		}
		return store, nil
	}
	if err := json.Unmarshal(data, &store.state); err != nil {
		return nil, fmt.Errorf("разбор %s: %w", store.path, err)
	}
	if store.state.Version == 0 {
		store.state.Version = 1
	}
	repairGeneratedNames(&store.state)
	return store, nil
}

func presetProfiles() []*Profile {
	return []*Profile{
		{ID: "p1", Name: "Кратко / без кода / формально", Style: "concise", Format: "without_code", Tone: "formal"},
		{ID: "p2", Name: "Подробно / с кодом / неформально", Style: "detailed", Format: "with_code", Tone: "informal"},
		{ID: "p3", Name: "Кратко / с кодом / формально", Style: "concise", Format: "with_code", Tone: "formal"},
	}
}

func emptyProfiles() []*Profile {
	return []*Profile{
		{ID: "p1", Name: "Профиль 1"},
		{ID: "p2", Name: "Профиль 2"},
		{ID: "p3", Name: "Профиль 3"},
	}
}

func newProfileState() ProfileState {
	return ProfileState{
		Version:  1,
		ActiveID: "p1",
		Profiles: emptyProfiles(),
	}
}

func (s *ProfileStore) Fresh() bool { return s.fresh }

func (s *ProfileStore) Get() ProfileState {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return clone(s.state)
}

func (s *ProfileStore) Save(state ProfileState) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state = clone(state)
	return s.saveLocked()
}

func (s *ProfileStore) saveLocked() error {
	data, err := json.MarshalIndent(s.state, "", "  ")
	if err != nil {
		return err
	}
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	_ = os.Remove(s.path)
	return os.Rename(tmp, s.path)
}

func (s *ProfileStore) SeedProfiles() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	presets := presetProfiles()
	for _, preset := range presets {
		profile := findProfile(s.state, preset.ID)
		if profile == nil {
			s.state.Profiles = append(s.state.Profiles, preset)
			profile = preset
		} else {
			profile.Name = preset.Name
			profile.Style = preset.Style
			profile.Format = preset.Format
			profile.Tone = preset.Tone
			profile.Constraints = preset.Constraints
			profile.Preferences = preset.Preferences
		}
		seedTestMemory(profile)
	}
	s.state.ActiveID = "p1"
	return s.saveLocked()
}

func seedTestMemory(profile *Profile) {
	items := []MemoryItem{
		{Layer: LayerWorking, Value: "Проект: API для доставки еды", Pinned: true},
		{Layer: LayerWorking, Value: "Дедлайн: 15 марта", Pinned: true},
		{Layer: LayerWorking, Value: "Бюджет: 2,5 млн рублей"},
		{Layer: LayerWorking, Value: "Платформа: iOS-клиент + Go backend"},
		{Layer: LayerWorking, Value: "Ограничение: в MVP без корзины", Pinned: true},
		{Layer: LayerWorking, Value: "Стек: Go, PostgreSQL, REST"},
		{Layer: LayerWorking, Value: "Нагрузка: 10 000 RPS"},
		{Layer: LayerLong, Key: "роль", Value: "backend-разработчик", Pinned: true},
		{Layer: LayerLong, Key: "основной стек", Value: "Go, PostgreSQL", Pinned: true},
		{Layer: LayerLong, Key: "запрещено на старте", Value: "Kubernetes и микросервисы", Pinned: true},
		{Layer: LayerLong, Key: "подход", Value: "сначала рабочее MVP, потом оптимизация"},
		{Layer: LayerLong, Key: "риски", Value: "показывать риски и компромиссы"},
		{Layer: LayerLong, Key: "проект", Value: "сервис доставки еды"},
	}
	for _, item := range items {
		if profileHasItem(profile, item) {
			continue
		}
		item.ID = newID()
		item.CreatedAt = time.Now().Format(time.RFC3339)
		if item.Layer == LayerWorking {
			profile.Memory.Working = append(profile.Memory.Working, item)
		} else {
			profile.Memory.LongTerm = append(profile.Memory.LongTerm, item)
		}
	}
}

func profileHasItem(profile *Profile, item MemoryItem) bool {
	items := profile.Memory.LongTerm
	if item.Layer == LayerWorking {
		items = profile.Memory.Working
	}
	for _, existing := range items {
		if existing.Key == item.Key && existing.Value == item.Value {
			return true
		}
	}
	return false
}

func (s *ProfileStore) UpdateProfile(profileID, name, style, format, tone string, constraints, preferences []string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	profile := findProfile(s.state, profileID)
	if profile == nil {
		return fmt.Errorf("профиль не найден")
	}
	oldName := profile.Name
	profile.Name = strings.TrimSpace(name)
	profile.Style = strings.TrimSpace(style)
	profile.Format = strings.TrimSpace(format)
	profile.Tone = strings.TrimSpace(tone)
	profile.Constraints = constraints
	profile.Preferences = preferences
	if profile.Name == "" || profile.Name == oldName || isGeneratedProfileName(profile.Name) {
		refreshGeneratedName(profile)
	}
	return s.saveLocked()
}

func (s *ProfileStore) SelectProfile(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if findProfile(s.state, id) == nil {
		return fmt.Errorf("профиль не найден")
	}
	s.state.ActiveID = id
	return s.saveLocked()
}

func (s *ProfileStore) AddItem(profileID string, item MemoryItem) (MemoryItem, error) {
	if item.Layer != LayerWorking && item.Layer != LayerLong {
		return MemoryItem{}, fmt.Errorf("неизвестный слой")
	}
	if strings.TrimSpace(item.Value) == "" {
		return MemoryItem{}, fmt.Errorf("value обязательно")
	}
	if item.Layer == LayerLong && strings.TrimSpace(item.Key) == "" {
		return MemoryItem{}, fmt.Errorf("key обязателен для long-term")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	profile := findProfile(s.state, profileID)
	if profile == nil {
		return MemoryItem{}, fmt.Errorf("профиль не найден")
	}
	item.ID = newID()
	item.Key = strings.TrimSpace(item.Key)
	item.Value = strings.TrimSpace(item.Value)
	item.CreatedAt = time.Now().Format(time.RFC3339)
	if item.Layer == LayerWorking {
		profile.Memory.Working = append(profile.Memory.Working, item)
	} else {
		profile.Memory.LongTerm = append(profile.Memory.LongTerm, item)
	}
	return item, s.saveLocked()
}

func (s *ProfileStore) DeleteItem(profileID, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	profile := findProfile(s.state, profileID)
	if profile == nil {
		return fmt.Errorf("профиль не найден")
	}
	for i, item := range profile.Memory.Working {
		if item.ID == id {
			profile.Memory.Working = append(profile.Memory.Working[:i], profile.Memory.Working[i+1:]...)
			return s.saveLocked()
		}
	}
	for i, item := range profile.Memory.LongTerm {
		if item.ID == id {
			profile.Memory.LongTerm = append(profile.Memory.LongTerm[:i], profile.Memory.LongTerm[i+1:]...)
			return s.saveLocked()
		}
	}
	return fmt.Errorf("элемент памяти не найден")
}

func (s *ProfileStore) ClearLayer(profileID, layer string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	profile := findProfile(s.state, profileID)
	if profile == nil {
		return fmt.Errorf("профиль не найден")
	}
	switch layer {
	case LayerShort:
		profile.Memory.ShortTerm = nil
	case LayerWorking:
		profile.Memory.Working = nil
	case LayerLong:
		profile.Memory.LongTerm = nil
	default:
		return fmt.Errorf("неизвестный слой")
	}
	return s.saveLocked()
}

func (s *ProfileStore) ClearSession(profileID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	profile := findProfile(s.state, profileID)
	if profile == nil {
		return fmt.Errorf("профиль не найден")
	}
	profile.Memory.Messages = nil
	profile.Memory.ShortTerm = nil
	profile.Memory.Turns = nil
	profile.Memory.ChatUsage = Usage{}
	profile.Memory.JudgeUsage = Usage{}
	profile.Memory.ProfileUpdateUsage = Usage{}
	return s.saveLocked()
}

func (s *ProfileStore) ClearAll() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state = newProfileState()
	s.fresh = true
	if err := os.Remove(s.path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return nil
}

func (s *ProfileStore) ClearMemory(profileID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	profile := findProfile(s.state, profileID)
	if profile == nil {
		return fmt.Errorf("профиль не найден")
	}
	profile.Memory = MemoryState{Version: 1}
	return s.saveLocked()
}

func (s *ProfileStore) Chat(ctx context.Context, llm LLMClient, profileID, user string) (ChatResult, error) {
	state := s.Get()
	profile := findProfile(state, profileID)
	if profile == nil {
		return ChatResult{}, fmt.Errorf("профиль не найден")
	}

	updated, updateUsage, err := updateProfile(ctx, llm, profile, user)
	if err != nil {
		profile.Memory.ProfileUpdateUsage.Add(updateUsage)
		_ = s.Save(state)
		return ChatResult{}, err
	}
	profile.Memory.ProfileUpdateUsage.Add(updateUsage)

	request, layerStats := BuildRequest(profile.Memory, user, profile)
	response, err := llm.Chat(ctx, request)
	if err != nil {
		return ChatResult{}, err
	}
	if strings.TrimSpace(response.Message.Content) == "" {
		return ChatResult{}, fmt.Errorf("модель вернула пустой ответ")
	}

	profile.Memory.Messages = append(profile.Memory.Messages,
		Message{Role: "user", Content: user},
		Message{Role: "assistant", Content: response.Message.Content},
	)
	profile.Memory.ShortTerm = append(profile.Memory.ShortTerm,
		Message{Role: "user", Content: user},
		Message{Role: "assistant", Content: response.Message.Content},
	)
	if len(profile.Memory.ShortTerm) > ShortWindow {
		profile.Memory.ShortTerm = profile.Memory.ShortTerm[len(profile.Memory.ShortTerm)-ShortWindow:]
	}
	profile.Memory.ChatUsage.Add(response.Usage)
	profile.Memory.Turns = append(profile.Memory.Turns, TurnMeta{
		Request:            request,
		PromptTokens:       response.Usage.PromptTokens,
		CompletionTokens:   response.Usage.CompletionTokens,
		SessionTotal:       profile.Memory.ChatUsage.TotalTokens,
		LayerStats:         layerStats,
		ProfileUpdate:      updated,
		ProfileUpdateUsage: updateUsage,
	})
	if err := s.Save(state); err != nil {
		return ChatResult{}, err
	}

	return ChatResult{
		Reply:              response.Message.Content,
		Request:            request,
		TurnUsage:          response.Usage,
		ChatUsage:          profile.Memory.ChatUsage,
		JudgeUsage:         profile.Memory.JudgeUsage,
		ProfileUpdateUsage: profile.Memory.ProfileUpdateUsage,
		TotalUsage:         profile.Memory.ChatUsage,
		LayerStats:         layerStats,
		Profile:            clone(profile),
		ProfileUpdated:     updated,
		Working:            append([]MemoryItem(nil), profile.Memory.Working...),
		LongTerm:           append([]MemoryItem(nil), profile.Memory.LongTerm...),
	}, nil
}

func (s *ProfileStore) JudgeProfile(ctx context.Context, llm LLMClient, profileID, question, answer string) (JudgeResult, error) {
	state := s.Get()
	profile := findProfile(state, profileID)
	if profile == nil {
		return JudgeResult{}, fmt.Errorf("профиль не найден")
	}
	prompt := fmt.Sprintf("Профиль:\n%s\n\nВопрос: %s\nОтвет: %s", profileText(profile), question, answer)
	response, err := llm.Chat(ctx, []Message{
		{Role: "system", Content: profileJudgePrompt},
		{Role: "user", Content: prompt},
	})
	if err != nil {
		return JudgeResult{}, err
	}
	profile.Memory.JudgeUsage.Add(response.Usage)
	content := strings.TrimSpace(response.Message.Content)
	start := strings.Index(content, "{")
	end := strings.LastIndex(content, "}")
	if start < 0 || end < start {
		return JudgeResult{}, fmt.Errorf("судья вернул не JSON: %s", content)
	}
	var parsed struct {
		Compliant bool   `json:"compliant"`
		Reason    string `json:"reason"`
	}
	if err := json.Unmarshal([]byte(content[start:end+1]), &parsed); err != nil {
		return JudgeResult{}, fmt.Errorf("разбор ответа судьи: %w", err)
	}
	if len(profile.Memory.Turns) > 0 {
		last := &profile.Memory.Turns[len(profile.Memory.Turns)-1]
		last.ProfileJudge = &JudgeInfo{Correct: parsed.Compliant, Reason: parsed.Reason}
	}
	if err := s.Save(state); err != nil {
		return JudgeResult{}, err
	}
	return JudgeResult{
		Compliant:    parsed.Compliant,
		Reason:       parsed.Reason,
		Usage:        response.Usage,
		SessionTotal: profile.Memory.ChatUsage,
	}, nil
}

func BuildRequest(state MemoryState, user string, profile *Profile) ([]Message, LayerStats) {
	var stats LayerStats
	system := systemPrompt
	if profile != nil {
		section := "\n\n## User profile\n" + profileText(profile)
		system += section
		stats.ProfileChars = utf8.RuneCountInString(section)
		stats.ProfileTokens = approxTokens(stats.ProfileChars)
	}

	longText := formatItems(selectItems(state.LongTerm, user))
	workingText := formatItems(selectItems(state.Working, user))
	if longText != "" {
		section := "\n\n## Long-term memory\n" + longText
		system += section
		stats.LongChars = utf8.RuneCountInString(section)
		stats.LongTokens = approxTokens(stats.LongChars)
	}
	if workingText != "" {
		section := "\n\n## Working memory\n" + workingText
		system += section
		stats.WorkingChars = utf8.RuneCountInString(section)
		stats.WorkingTokens = approxTokens(stats.WorkingChars)
	}

	short := lastMessages(state.ShortTerm, ShortWindow)
	for _, message := range short {
		stats.ShortChars += utf8.RuneCountInString(message.Content)
	}
	stats.ShortTokens = approxTokens(stats.ShortChars)

	messages := []Message{{Role: "system", Content: system}}
	messages = append(messages, short...)
	messages = append(messages, Message{Role: "user", Content: user})
	return messages, stats
}

func updateProfile(ctx context.Context, llm LLMClient, profile *Profile, user string) (bool, Usage, error) {
	current := profileText(profile)
	response, err := llm.Chat(ctx, []Message{
		{Role: "system", Content: profileUpdatePrompt},
		{Role: "user", Content: "Текущий профиль:\n" + current + "\n\nНовое сообщение пользователя:\n" + user},
	})
	if err != nil {
		return false, Usage{}, err
	}
	content := strings.TrimSpace(response.Message.Content)
	start := strings.Index(content, "{")
	end := strings.LastIndex(content, "}")
	if start < 0 || end < start {
		return false, response.Usage, fmt.Errorf("profile update: не JSON")
	}
	var patch struct {
		Style       string   `json:"style"`
		Format      string   `json:"format"`
		Tone        string   `json:"tone"`
		Constraints []string `json:"constraints"`
		Preferences []string `json:"preferences"`
	}
	if err := json.Unmarshal([]byte(content[start:end+1]), &patch); err != nil {
		return false, response.Usage, fmt.Errorf("profile update: %w", err)
	}
	if !hasExplicitProfileIntent(user) {
		return false, response.Usage, nil
	}
	changed := false
	if patch.Style != "" && patch.Style != profile.Style {
		profile.Style = patch.Style
		changed = true
	}
	if patch.Format != "" && patch.Format != profile.Format {
		profile.Format = patch.Format
		changed = true
	}
	if patch.Tone != "" && patch.Tone != profile.Tone {
		profile.Tone = patch.Tone
		changed = true
	}
	if patch.Constraints != nil {
		profile.Constraints = patch.Constraints
		changed = true
	}
	if patch.Preferences != nil {
		profile.Preferences = patch.Preferences
		changed = true
	}
	if changed {
		refreshGeneratedName(profile)
	}
	return changed, response.Usage, nil
}

func hasExplicitProfileIntent(user string) bool {
	lower := strings.ToLower(user)
	markers := []string{
		"отвечай", "пиши", "общайся", "говори со мной",
		"в дальнейшем", "дальше", "всегда", "предпочита",
		"с этого момента", "хочу чтобы", "хочу, чтобы",
		"мне нужно чтобы", "мне нужно, чтобы", "запомни",
		"не надо", "без кода", "с кодом", "на ты", "на вы",
		"формально", "неформально",
	}
	for _, marker := range markers {
		if strings.Contains(lower, marker) {
			return true
		}
	}
	return false
}

func applyExplicitProfile(profile *Profile, user string) bool {
	lower := strings.ToLower(user)
	changed := false
	if strings.Contains(lower, "кратко") && profile.Style != "concise" {
		profile.Style = "concise"
		changed = true
	}
	if strings.Contains(lower, "подробно") && profile.Style != "detailed" {
		profile.Style = "detailed"
		changed = true
	}
	if strings.Contains(lower, "без кода") && profile.Format != "without_code" {
		profile.Format = "without_code"
		changed = true
	} else if strings.Contains(lower, "с кодом") && profile.Format != "with_code" {
		profile.Format = "with_code"
		changed = true
	}
	if strings.Contains(lower, "на ты") && profile.Tone != "informal" {
		profile.Tone = "informal"
		changed = true
	}
	if (strings.Contains(lower, "на вы") || strings.Contains(lower, "формально")) && profile.Tone != "formal" {
		profile.Tone = "formal"
		changed = true
	}
	return changed
}

func profileLabel(style, format, tone string) string {
	styleLabel := map[string]string{"concise": "Кратко", "detailed": "Подробно"}[style]
	formatLabel := map[string]string{"without_code": "без кода", "with_code": "с кодом"}[format]
	toneLabel := map[string]string{"formal": "формально", "informal": "неформально"}[tone]
	if styleLabel == "" {
		styleLabel = "—"
	}
	if formatLabel == "" {
		formatLabel = "—"
	}
	if toneLabel == "" {
		toneLabel = "—"
	}
	return styleLabel + " / " + formatLabel + " / " + toneLabel
}

func isGeneratedProfileName(name string) bool {
	return name == "" || strings.HasPrefix(name, "Кратко /") || strings.HasPrefix(name, "Подробно /") || strings.HasPrefix(name, "Профиль ")
}

func refreshGeneratedName(profile *Profile) {
	if profile.Style == "" && profile.Format == "" && profile.Tone == "" {
		return
	}
	if isGeneratedProfileName(profile.Name) {
		profile.Name = profileLabel(profile.Style, profile.Format, profile.Tone)
	}
}

func repairGeneratedNames(state *ProfileState) {
	for _, profile := range state.Profiles {
		refreshGeneratedName(profile)
	}
}

func profileText(profile *Profile) string {
	if profile == nil {
		return ""
	}
	var b strings.Builder
	if profile.Name != "" {
		b.WriteString("- Имя: " + profile.Name + "\n")
	}
	if profile.Style != "" {
		b.WriteString("- Стиль: " + profile.Style + "\n")
	}
	if profile.Format != "" {
		b.WriteString("- Формат: " + profile.Format + "\n")
	}
	if profile.Tone != "" {
		b.WriteString("- Тон: " + profile.Tone + "\n")
	}
	if len(profile.Constraints) > 0 {
		b.WriteString("- Ограничения: " + strings.Join(profile.Constraints, "; ") + "\n")
	}
	if len(profile.Preferences) > 0 {
		b.WriteString("- Предпочтения: " + strings.Join(profile.Preferences, "; ") + "\n")
	}
	return strings.TrimRight(b.String(), "\n")
}

func selectItems(items []MemoryItem, query string) []MemoryItem {
	if len(items) == 0 {
		return nil
	}
	queryTokens := tokenSet(query)
	selected := map[string]bool{}
	out := []MemoryItem{}
	for _, item := range items {
		if item.Pinned {
			out = append(out, item)
			selected[item.ID] = true
		}
	}
	type scored struct {
		item  MemoryItem
		score int
	}
	var candidates []scored
	for _, item := range items {
		if selected[item.ID] {
			continue
		}
		score := overlapScore(queryTokens, tokenSet(item.Key+" "+item.Value))
		if score > 0 {
			candidates = append(candidates, scored{item: item, score: score})
		}
	}
	sort.SliceStable(candidates, func(i, j int) bool {
		if candidates[i].score == candidates[j].score {
			return candidates[i].item.Key < candidates[j].item.Key
		}
		return candidates[i].score > candidates[j].score
	})
	for i := 0; i < len(candidates) && i < RelevantTopK; i++ {
		out = append(out, candidates[i].item)
	}
	return out
}

func formatItems(items []MemoryItem) string {
	if len(items) == 0 {
		return ""
	}
	var b strings.Builder
	for _, item := range items {
		b.WriteString("- ")
		if item.Key != "" {
			b.WriteString(item.Key)
			b.WriteString(": ")
		}
		b.WriteString(item.Value)
		b.WriteString("\n")
	}
	return strings.TrimRight(b.String(), "\n")
}

func lastMessages(messages []Message, n int) []Message {
	if n <= 0 || len(messages) <= n {
		return append([]Message(nil), messages...)
	}
	return append([]Message(nil), messages[len(messages)-n:]...)
}

func tokenSet(text string) map[string]bool {
	set := map[string]bool{}
	for _, word := range strings.Fields(strings.ToLower(text)) {
		word = strings.Trim(word, ".,!?;:()[]{}\"'«»—-")
		if utf8.RuneCountInString(word) >= 3 {
			set[word] = true
		}
	}
	return set
}

func overlapScore(a, b map[string]bool) int {
	score := 0
	for word := range a {
		if b[word] {
			score++
		}
	}
	return score
}

func approxTokens(chars int) int {
	if chars <= 0 {
		return 0
	}
	return (chars + 2) / 3
}

func findProfile(state ProfileState, id string) *Profile {
	for _, profile := range state.Profiles {
		if profile.ID == id {
			return profile
		}
	}
	return nil
}

func newID() string {
	buf := make([]byte, 8)
	if _, err := rand.Read(buf); err != nil {
		return fmt.Sprintf("%d", time.Now().UnixNano())
	}
	return fmt.Sprintf("%x", buf)
}

func clone[T any](v T) T {
	data, err := json.Marshal(v)
	if err != nil {
		return v
	}
	var out T
	if err := json.Unmarshal(data, &out); err != nil {
		return v
	}
	return out
}
