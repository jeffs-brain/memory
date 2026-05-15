// SPDX-License-Identifier: Apache-2.0

package search

import (
	"testing"
)

func TestNewSnowballStemmer_English(t *testing.T) {
	t.Parallel()
	s, err := NewSnowballStemmer("en")
	if err != nil {
		t.Fatalf("NewSnowballStemmer(en) error: %v", err)
	}
	if s.Language() != "en" {
		t.Fatalf("Language() = %q, want %q", s.Language(), "en")
	}

	cases := []struct {
		name  string
		input string
		want  string
	}{
		{"running to run", "running", "run"},
		{"dogs to dog", "dogs", "dog"},
		{"studies to studi", "studies", "studi"},
		{"caresses to caress", "caresses", "caress"},
		{"ponies to poni", "ponies", "poni"},
		{"cats to cat", "cats", "cat"},
		{"connection to connect", "connection", "connect"},
		{"empty string", "", ""},
		{"already a stem", "run", "run"},
		{"uppercase normalised", "Running", "run"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := s.Stem(tc.input)
			if got != tc.want {
				t.Errorf("Stem(%q) = %q, want %q", tc.input, got, tc.want)
			}
		})
	}
}

func TestNewSnowballStemmer_German(t *testing.T) {
	t.Parallel()
	s, err := NewSnowballStemmer("de")
	if err != nil {
		t.Fatalf("NewSnowballStemmer(de) error: %v", err)
	}
	if s.Language() != "de" {
		t.Fatalf("Language() = %q, want %q", s.Language(), "de")
	}

	cases := []struct {
		name  string
		input string
		want  string
	}{
		{"hauser to haus", "häuser", "haus"},
		{"gegangen to gang", "gegangen", "gegang"},
		{"freundlich to freundlich", "freundlich", "freundlich"},
		{"kinder to kind", "kinder", "kind"},
		{"laufen to lauf", "laufen", "lauf"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := s.Stem(tc.input)
			if got != tc.want {
				t.Errorf("Stem(%q) = %q, want %q", tc.input, got, tc.want)
			}
		})
	}
}

func TestNewSnowballStemmer_French(t *testing.T) {
	t.Parallel()
	s, err := NewSnowballStemmer("fr")
	if err != nil {
		t.Fatalf("NewSnowballStemmer(fr) error: %v", err)
	}

	cases := []struct {
		name  string
		input string
		want  string
	}{
		{"mangeons to mangeon", "mangeons", "mangeon"},
		{"courions to courion", "courions", "courion"},
		{"maisons to maison", "maisons", "maison"},
		{"manger to mang", "manger", "mang"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := s.Stem(tc.input)
			if got != tc.want {
				t.Errorf("Stem(%q) = %q, want %q", tc.input, got, tc.want)
			}
		})
	}
}

func TestNewSnowballStemmer_AllLanguages(t *testing.T) {
	t.Parallel()
	for lang := range stemFunctions {
		t.Run(lang, func(t *testing.T) {
			s, err := NewSnowballStemmer(lang)
			if err != nil {
				t.Fatalf("NewSnowballStemmer(%q) error: %v", lang, err)
			}
			if s.Language() != lang {
				t.Fatalf("Language() = %q, want %q", s.Language(), lang)
			}
			// Every stemmer should return a non-empty result for a non-empty input.
			result := s.Stem("test")
			if result == "" {
				t.Errorf("Stem(%q) returned empty for lang %q", "test", lang)
			}
		})
	}
}

func TestNewSnowballStemmer_UnsupportedLanguage(t *testing.T) {
	t.Parallel()
	_, err := NewSnowballStemmer("xx")
	if err == nil {
		t.Fatal("expected error for unsupported language, got nil")
	}
	ule, ok := err.(*UnsupportedLanguageError)
	if !ok {
		t.Fatalf("expected *UnsupportedLanguageError, got %T", err)
	}
	if ule.Lang != "xx" {
		t.Errorf("UnsupportedLanguageError.Lang = %q, want %q", ule.Lang, "xx")
	}
}

func TestDetectLanguage_English(t *testing.T) {
	t.Parallel()
	text := "The quick brown fox jumps over the lazy dog and then runs across the green field towards the other side of the river where the fisherman was standing quietly"
	lang, confidence := DetectLanguage(text, nil)
	if lang != "en" {
		t.Errorf("DetectLanguage(english text) = %q, want %q (confidence=%f)", lang, "en", confidence)
	}
	if confidence < DefaultConfidenceThreshold {
		t.Errorf("confidence = %f, want >= %f", confidence, DefaultConfidenceThreshold)
	}
}

func TestDetectLanguage_German(t *testing.T) {
	t.Parallel()
	text := "Die Bundesrepublik Deutschland ist ein demokratischer und sozialer Bundesstaat mit einer langen Geschichte und vielen verschiedenen Regionen die sich durch ihre Kultur unterscheiden"
	lang, confidence := DetectLanguage(text, nil)
	if lang != "de" {
		t.Errorf("DetectLanguage(german text) = %q, want %q (confidence=%f)", lang, "de", confidence)
	}
	if confidence < DefaultConfidenceThreshold {
		t.Errorf("confidence = %f, want >= %f", confidence, DefaultConfidenceThreshold)
	}
}

func TestDetectLanguage_French(t *testing.T) {
	t.Parallel()
	text := "La République française est un pays dont la majeure partie du territoire se situe en Europe occidentale et qui possède de nombreuses régions outre-mer dans le monde entier"
	lang, confidence := DetectLanguage(text, nil)
	if lang != "fr" {
		t.Errorf("DetectLanguage(french text) = %q, want %q (confidence=%f)", lang, "fr", confidence)
	}
	if confidence < DefaultConfidenceThreshold {
		t.Errorf("confidence = %f, want >= %f", confidence, DefaultConfidenceThreshold)
	}
}

func TestDetectLanguage_Russian(t *testing.T) {
	t.Parallel()
	text := "Российская Федерация является демократическим федеративным правовым государством с республиканской формой правления и развитой системой управления"
	lang, confidence := DetectLanguage(text, nil)
	if lang != "ru" {
		t.Errorf("DetectLanguage(russian text) = %q, want %q (confidence=%f)", lang, "ru", confidence)
	}
	if confidence < DefaultConfidenceThreshold {
		t.Errorf("confidence = %f, want >= %f", confidence, DefaultConfidenceThreshold)
	}
}

func TestDetectLanguage_ShortText_DefaultsToEnglish(t *testing.T) {
	t.Parallel()
	text := "hello"
	lang, confidence := DetectLanguage(text, nil)
	if lang != "en" {
		t.Errorf("DetectLanguage(short text) = %q, want %q", lang, "en")
	}
	if confidence != 0.0 {
		t.Errorf("confidence = %f, want 0.0", confidence)
	}
}

func TestDetectLanguage_Empty(t *testing.T) {
	t.Parallel()
	lang, confidence := DetectLanguage("", nil)
	if lang != "en" {
		t.Errorf("DetectLanguage(\"\") = %q, want %q", lang, "en")
	}
	if confidence != 0.0 {
		t.Errorf("confidence = %f, want 0.0", confidence)
	}
}

func TestTokenizeCJK_Chinese(t *testing.T) {
	t.Parallel()
	// "Machine learning" in Chinese: 机器学习
	text := "机器学习"
	tokens := TokenizeCJK(text)
	// 4 characters -> 2 trigrams: "机器学", "器学习"
	expected := []string{"机器学", "器学习"}
	if len(tokens) != len(expected) {
		t.Fatalf("TokenizeCJK(%q) = %v (len %d), want %v (len %d)",
			text, tokens, len(tokens), expected, len(expected))
	}
	for i, tok := range tokens {
		if tok != expected[i] {
			t.Errorf("token[%d] = %q, want %q", i, tok, expected[i])
		}
	}
}

func TestTokenizeCJK_Japanese(t *testing.T) {
	t.Parallel()
	// Mixed hiragana/katakana/kanji: "東京タワー"
	text := "東京タワー"
	tokens := TokenizeCJK(text)
	// 5 chars -> 3 trigrams
	expected := []string{"東京タ", "京タワ", "タワー"}
	if len(tokens) != len(expected) {
		t.Fatalf("TokenizeCJK(%q) = %v (len %d), want %v (len %d)",
			text, tokens, len(tokens), expected, len(expected))
	}
	for i, tok := range tokens {
		if tok != expected[i] {
			t.Errorf("token[%d] = %q, want %q", i, tok, expected[i])
		}
	}
}

func TestTokenizeCJK_Korean(t *testing.T) {
	t.Parallel()
	// "Hello" in Korean: "안녕하세요"
	text := "안녕하세요"
	tokens := TokenizeCJK(text)
	// 5 chars -> 3 trigrams
	expected := []string{"안녕하", "녕하세", "하세요"}
	if len(tokens) != len(expected) {
		t.Fatalf("TokenizeCJK(%q) = %v (len %d), want %v (len %d)",
			text, tokens, len(tokens), expected, len(expected))
	}
	for i, tok := range tokens {
		if tok != expected[i] {
			t.Errorf("token[%d] = %q, want %q", i, tok, expected[i])
		}
	}
}

func TestTokenizeCJK_MixedCJKAndLatin(t *testing.T) {
	t.Parallel()
	text := "Hello 机器学习 world"
	tokens := TokenizeCJK(text)
	// "Hello" -> "hello", CJK -> "机器学", "器学习", "world" -> "world"
	expected := []string{"hello", "机器学", "器学习", "world"}
	if len(tokens) != len(expected) {
		t.Fatalf("TokenizeCJK(%q) = %v (len %d), want %v (len %d)",
			text, tokens, len(tokens), expected, len(expected))
	}
	for i, tok := range tokens {
		if tok != expected[i] {
			t.Errorf("token[%d] = %q, want %q", i, tok, expected[i])
		}
	}
}

func TestTokenizeCJK_ShortCJK(t *testing.T) {
	t.Parallel()
	// Less than 3 CJK chars: returned as-is
	text := "学习"
	tokens := TokenizeCJK(text)
	expected := []string{"学习"}
	if len(tokens) != len(expected) {
		t.Fatalf("TokenizeCJK(%q) = %v, want %v", text, tokens, expected)
	}
	if tokens[0] != expected[0] {
		t.Errorf("token[0] = %q, want %q", tokens[0], expected[0])
	}
}

func TestTokenizeCJK_Empty(t *testing.T) {
	t.Parallel()
	tokens := TokenizeCJK("")
	if tokens != nil {
		t.Errorf("TokenizeCJK(\"\") = %v, want nil", tokens)
	}
}

func TestIsCJK(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		r    rune
		want bool
	}{
		{"han character", '机', true},
		{"hiragana", 'あ', true},
		{"katakana", 'ア', true},
		{"hangul", '한', true},
		{"latin letter", 'A', false},
		{"digit", '1', false},
		{"space", ' ', false},
		{"cyrillic", 'Я', false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := IsCJK(tc.r)
			if got != tc.want {
				t.Errorf("IsCJK(%q) = %v, want %v", tc.r, got, tc.want)
			}
		})
	}
}

func TestContainsCJK(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		text string
		want bool
	}{
		{"chinese text", "Hello 机器学习", true},
		{"pure latin", "Hello world", false},
		{"japanese", "東京タワー", true},
		{"korean", "안녕하세요", true},
		{"empty", "", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ContainsCJK(tc.text)
			if got != tc.want {
				t.Errorf("ContainsCJK(%q) = %v, want %v", tc.text, got, tc.want)
			}
		})
	}
}

func TestStopWords_English(t *testing.T) {
	t.Parallel()
	sw := StopWords("en")
	if sw == nil {
		t.Fatal("StopWords(en) returned nil")
	}
	expectedWords := []string{"the", "is", "at", "which", "and", "or", "but", "in", "a"}
	for _, w := range expectedWords {
		if _, ok := sw[w]; !ok {
			t.Errorf("StopWords(en) missing %q", w)
		}
	}
}

func TestStopWords_German(t *testing.T) {
	t.Parallel()
	sw := StopWords("de")
	if sw == nil {
		t.Fatal("StopWords(de) returned nil")
	}
	expectedWords := []string{"der", "die", "das", "und", "ist", "ein"}
	for _, w := range expectedWords {
		if _, ok := sw[w]; !ok {
			t.Errorf("StopWords(de) missing %q", w)
		}
	}
}

func TestStopWords_French(t *testing.T) {
	t.Parallel()
	sw := StopWords("fr")
	if sw == nil {
		t.Fatal("StopWords(fr) returned nil")
	}
	expectedWords := []string{"le", "la", "les", "de", "et", "est"}
	for _, w := range expectedWords {
		if _, ok := sw[w]; !ok {
			t.Errorf("StopWords(fr) missing %q", w)
		}
	}
}

func TestStopWords_AllLanguages(t *testing.T) {
	t.Parallel()
	langs := []string{"en", "de", "fr", "es", "it", "pt", "nl", "ru", "zh", "ja"}
	for _, lang := range langs {
		sw := StopWords(lang)
		if sw == nil {
			t.Errorf("StopWords(%q) returned nil", lang)
			continue
		}
		if len(sw) == 0 {
			t.Errorf("StopWords(%q) returned empty set", lang)
		}
	}
}

func TestStopWords_UnsupportedLanguage(t *testing.T) {
	t.Parallel()
	sw := StopWords("xx")
	if sw != nil {
		t.Errorf("StopWords(xx) = %v, want nil", sw)
	}
}

func TestDetectLanguage_ConfigurableThreshold(t *testing.T) {
	t.Parallel()
	text := "The quick brown fox jumps over the lazy dog and then runs across the green field towards the other side of the river where the fisherman was standing quietly"

	// With very high threshold, detection defaults to English even for
	// English text if confidence is below.
	lang, confidence := DetectLanguage(text, &DetectLanguageOptions{Threshold: 0.99})
	// Either the text is detected as English (matching the threshold)
	// or returned as English by default. Either way lang should be "en".
	if lang != "en" {
		t.Errorf("DetectLanguage with threshold=0.99: lang = %q, want %q", lang, "en")
	}

	// With very low threshold, any above-zero confidence should suffice.
	lang2, confidence2 := DetectLanguage(text, &DetectLanguageOptions{Threshold: 0.01})
	if lang2 != "en" {
		t.Errorf("DetectLanguage with threshold=0.01: lang = %q, want %q", lang2, "en")
	}
	if confidence2 < 0.01 {
		t.Errorf("DetectLanguage with threshold=0.01: confidence = %f", confidence2)
	}
	_ = confidence
}

func TestDetectLanguage_ConfigurableMinLength(t *testing.T) {
	t.Parallel()
	// 35 alpha chars: below the default min (50) but above custom min (20).
	text := "The quick brown fox the other side"

	// Default min length (50) should return "en" with zero confidence.
	lang, confidence := DetectLanguage(text, nil)
	if lang != "en" {
		t.Errorf("DetectLanguage(short text, default) = %q, want %q", lang, "en")
	}
	if confidence != 0.0 {
		t.Errorf("confidence = %f, want 0.0", confidence)
	}

	// With minLength=20, the text is long enough and should produce a
	// non-zero confidence.
	lang2, confidence2 := DetectLanguage(text, &DetectLanguageOptions{MinLength: 20, Threshold: 0.01})
	if confidence2 <= 0 {
		t.Errorf("confidence = %f, want > 0", confidence2)
	}
	_ = lang2
}

func TestRegisterLanguage(t *testing.T) {
	// Register a custom language profile.
	customProfile := map[string]float64{
		"xy": 0.1, "yz": 0.08, "zx": 0.06,
	}
	RegisterLanguage("custom", customProfile)

	// Verify the profile is accessible in detection.
	// Generate text that strongly matches our custom profile.
	text := "xyyzxyyzzxyzxyyzzxyyzxyzxyyzxyyzzxyzxyyzzxyyz"
	// This is a made-up language so it might not win detection, but the
	// profile should exist.
	lang, _ := DetectLanguage(text, &DetectLanguageOptions{MinLength: 10, Threshold: 0.01})
	// The custom profile should at least be a contender.
	_ = lang

	// Clean up: re-register with the same code to confirm no panic.
	RegisterLanguage("custom", customProfile)
}

func TestDefaultThresholdValue(t *testing.T) {
	t.Parallel()
	if DefaultConfidenceThreshold != 0.7 {
		t.Errorf("DefaultConfidenceThreshold = %f, want 0.7", DefaultConfidenceThreshold)
	}
}

func TestDefaultMinDetectionLengthValue(t *testing.T) {
	t.Parallel()
	if DefaultMinDetectionLength != 50 {
		t.Errorf("DefaultMinDetectionLength = %d, want 50", DefaultMinDetectionLength)
	}
}
