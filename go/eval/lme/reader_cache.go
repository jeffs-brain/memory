// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	readerCachePollInterval = 100 * time.Millisecond
	readerCacheLockTTL      = 10 * time.Minute
)

type readerCacheRecord struct {
	Answer string `json:"answer"`
	Usage  Usage  `json:"usage"`
}

type judgeCacheRecord struct {
	Raw   json.RawMessage `json:"raw"`
	Usage Usage           `json:"usage"`
}

func readCachedReaderAnswer(
	ctx context.Context,
	cacheDir string,
	model string,
	prompt string,
	compute func() (string, Usage, error),
) (string, Usage, error) {
	cachePath := readerCachePath(cacheDir, model, prompt)
	if record, ok, err := loadReaderCache(cachePath); err != nil {
		return "", Usage{}, err
	} else if ok {
		return record.Answer, Usage{}, nil
	}

	lockPath := cachePath + ".lock"
	unlock, err := acquireReaderCacheLock(ctx, cachePath, lockPath)
	if err != nil {
		return "", Usage{}, err
	}
	defer unlock()

	if record, ok, err := loadReaderCache(cachePath); err != nil {
		return "", Usage{}, err
	} else if ok {
		return record.Answer, Usage{}, nil
	}

	answer, usage, err := compute()
	if err != nil {
		return "", Usage{}, err
	}
	if err := storeReaderCache(cachePath, readerCacheRecord{
		Answer: answer,
		Usage:  usage,
	}); err != nil {
		return "", Usage{}, err
	}
	return answer, usage, nil
}

func readerCachePath(cacheDir, model, prompt string) string {
	sum := sha256.Sum256([]byte(model + "\n" + prompt))
	key := hex.EncodeToString(sum[:])
	return filepath.Join(cacheDir, key[:2], key+".json")
}

func judgeCachePath(cacheDir, model, prompt string) string {
	sum := sha256.Sum256([]byte("judge" + "\n" + model + "\n" + prompt))
	key := hex.EncodeToString(sum[:])
	return filepath.Join(cacheDir, "judge", key[:2], key+".json")
}

func readCachedJudgeResponse(
	ctx context.Context,
	cacheDir string,
	model string,
	prompt string,
	compute func() (json.RawMessage, Usage, error),
) (json.RawMessage, Usage, error) {
	cachePath := judgeCachePath(cacheDir, model, prompt)
	if record, ok, err := loadJudgeCache(cachePath); err != nil {
		return nil, Usage{}, err
	} else if ok {
		return record.Raw, Usage{}, nil
	}

	lockPath := cachePath + ".lock"
	unlock, err := acquireReaderCacheLock(ctx, cachePath, lockPath)
	if err != nil {
		return nil, Usage{}, err
	}
	defer unlock()

	if record, ok, err := loadJudgeCache(cachePath); err != nil {
		return nil, Usage{}, err
	} else if ok {
		return record.Raw, Usage{}, nil
	}

	raw, usage, err := compute()
	if err != nil {
		return nil, Usage{}, err
	}
	if err := storeJudgeCache(cachePath, judgeCacheRecord{
		Raw:   raw,
		Usage: usage,
	}); err != nil {
		return nil, Usage{}, err
	}
	return raw, usage, nil
}

func loadReaderCache(path string) (readerCacheRecord, bool, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return readerCacheRecord{}, false, nil
		}
		return readerCacheRecord{}, false, fmt.Errorf("read reader cache %s: %w", path, err)
	}
	var record readerCacheRecord
	if err := json.Unmarshal(data, &record); err != nil {
		return readerCacheRecord{}, false, fmt.Errorf("decode reader cache %s: %w", path, err)
	}
	return record, true, nil
}

func storeReaderCache(path string, record readerCacheRecord) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("mkdir reader cache %s: %w", filepath.Dir(path), err)
	}
	data, err := json.Marshal(record)
	if err != nil {
		return fmt.Errorf("marshal reader cache %s: %w", path, err)
	}
	data = append(data, '\n')
	tmp := path + fmt.Sprintf(".tmp.%d", os.Getpid())
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return fmt.Errorf("write reader cache temp %s: %w", tmp, err)
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("rename reader cache %s: %w", path, err)
	}
	return nil
}

func loadJudgeCache(path string) (judgeCacheRecord, bool, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return judgeCacheRecord{}, false, nil
		}
		return judgeCacheRecord{}, false, fmt.Errorf("read judge cache %s: %w", path, err)
	}
	var record judgeCacheRecord
	if err := json.Unmarshal(data, &record); err != nil {
		return judgeCacheRecord{}, false, fmt.Errorf("decode judge cache %s: %w", path, err)
	}
	return record, true, nil
}

func storeJudgeCache(path string, record judgeCacheRecord) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("mkdir judge cache %s: %w", filepath.Dir(path), err)
	}
	data, err := json.Marshal(record)
	if err != nil {
		return fmt.Errorf("marshal judge cache %s: %w", path, err)
	}
	data = append(data, '\n')
	tmp := path + fmt.Sprintf(".tmp.%d", os.Getpid())
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return fmt.Errorf("write judge cache temp %s: %w", tmp, err)
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("rename judge cache %s: %w", path, err)
	}
	return nil
}

func acquireReaderCacheLock(
	ctx context.Context,
	cachePath string,
	lockPath string,
) (func(), error) {
	if err := os.MkdirAll(filepath.Dir(cachePath), 0o755); err != nil {
		return func() {}, fmt.Errorf("mkdir reader cache lock dir %s: %w", filepath.Dir(cachePath), err)
	}
	for {
		file, err := os.OpenFile(lockPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
		if err == nil {
			_, _ = file.WriteString(time.Now().UTC().Format(time.RFC3339Nano))
			_ = file.Close()
			return func() {
				_ = os.Remove(lockPath)
			}, nil
		}
		if !errors.Is(err, os.ErrExist) {
			return func() {}, fmt.Errorf("create reader cache lock %s: %w", lockPath, err)
		}
		if unlock, ok, err := maybeBreakStaleReaderCacheLock(cachePath, lockPath); err != nil {
			return func() {}, err
		} else if ok {
			return unlock, nil
		}
		select {
		case <-ctx.Done():
			return func() {}, ctx.Err()
		case <-time.After(readerCachePollInterval):
		}
		if _, ok, err := loadReaderCache(cachePath); err == nil && ok {
			return func() {}, nil
		} else if err != nil && !strings.Contains(err.Error(), "decode reader cache") {
			return func() {}, err
		}
	}
}

func maybeBreakStaleReaderCacheLock(
	cachePath string,
	lockPath string,
) (func(), bool, error) {
	info, err := os.Stat(lockPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return func() {}, false, nil
		}
		return func() {}, false, fmt.Errorf("stat reader cache lock %s: %w", lockPath, err)
	}
	if time.Since(info.ModTime()) < readerCacheLockTTL {
		return func() {}, false, nil
	}
	if err := os.Remove(lockPath); err != nil && !errors.Is(err, os.ErrNotExist) {
		return func() {}, false, fmt.Errorf("remove stale reader cache lock %s: %w", lockPath, err)
	}
	file, err := os.OpenFile(lockPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		if errors.Is(err, os.ErrExist) {
			return func() {}, false, nil
		}
		return func() {}, false, fmt.Errorf("recreate reader cache lock %s: %w", lockPath, err)
	}
	_, _ = file.WriteString(time.Now().UTC().Format(time.RFC3339Nano))
	_ = file.Close()
	return func() {
		_ = os.Remove(lockPath)
	}, true, nil
}
