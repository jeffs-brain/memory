// SPDX-License-Identifier: Apache-2.0

package brain

import (
	"context"
	"errors"
	"fmt"
	"log/slog"

	"gopkg.in/yaml.v3"
)

// SchemaVersion is the current layout version understood by this binary.
// Increment this when the logical path layout changes and add a
// corresponding migration step to [schemaMigrations].
const SchemaVersion = 1

// schemaVersionPath is the brain-root-relative path for the version file.
// The string value lives in paths.go alongside the other root constants.
const schemaVersionPath = Path(schemaVersionName)

// schemaVersionFile is the on-disk YAML representation.
type schemaVersionFile struct {
	Version int `yaml:"version"`
}

// SchemaVersionPath returns the logical path of the schema version file.
// Exported so callers (e.g. the contract test suite) can reference it.
func SchemaVersionPath() Path { return schemaVersionPath }

// CheckSchemaVersion reads schema-version.yaml from the store and enforces
// compatibility. It must be called once at startup before any mutations.
//
// Three cases:
//   - File missing: first run or pre-versioning brain. Writes version 1.
//   - version == SchemaVersion: current, no action.
//   - version > SchemaVersion: newer than this binary. Returns an error
//     wrapping [ErrSchemaVersion].
func CheckSchemaVersion(ctx context.Context, s Store) error {
	v, err := readSchemaVersion(ctx, s)
	if err != nil {
		return err
	}

	switch {
	case v == 0:
		// Missing or unparseable — treat as uninitialised.
		if writeErr := writeSchemaVersion(ctx, s, SchemaVersion); writeErr != nil {
			return fmt.Errorf("brain: writing initial schema version: %w", writeErr)
		}
		slog.Info("brain: initialised schema version", "version", SchemaVersion)
		return nil

	case v == SchemaVersion:
		return nil

	case v > SchemaVersion:
		slog.Error("brain: schema version is newer than this binary supports",
			"found", v, "max", SchemaVersion)
		return fmt.Errorf("brain: schema version %d is newer than this binary supports (max: %d): %w",
			v, SchemaVersion, ErrSchemaVersion)

	default:
		// v < SchemaVersion and v > 0: needs migration. CheckSchemaVersion
		// does not migrate; callers should call MigrateSchema first.
		return fmt.Errorf("brain: schema version %d is older than current (%d); run migration: %w",
			v, SchemaVersion, ErrSchemaVersion)
	}
}

// schemaMigration migrates from version N to N+1. It runs inside the
// batch provided by [MigrateSchema].
type schemaMigration func(ctx context.Context, b Batch) error

// schemaMigrations holds the ordered migration steps. Index 0 migrates
// from version 1 to 2, index 1 from 2 to 3, and so on. Today there are
// no migrations because version 1 is the initial layout.
var schemaMigrations = []schemaMigration{}

// MigrateSchema brings the brain store up to [SchemaVersion] by running
// each migration step in sequence. All steps execute inside a single
// batch so they form one atomic commit.
//
// If the store is already at the current version, MigrateSchema is a
// no-op. If the store version is ahead of the binary, it returns
// [ErrSchemaVersion].
func MigrateSchema(ctx context.Context, s Store) error {
	v, err := readSchemaVersion(ctx, s)
	if err != nil {
		return err
	}

	// Treat missing (0) as pre-versioning — equivalent to version 1 with
	// no migrations needed, just stamp it.
	if v == 0 {
		v = 1
	}

	if v == SchemaVersion {
		return nil
	}

	if v > SchemaVersion {
		return fmt.Errorf("brain: schema version %d is newer than this binary supports (max: %d): %w",
			v, SchemaVersion, ErrSchemaVersion)
	}

	// Run migrations v to v+1 to SchemaVersion inside one batch.
	return s.Batch(ctx, BatchOptions{Reason: "schema-migration"}, func(b Batch) error {
		for step := v; step < SchemaVersion; step++ {
			idx := step - 1 // migrations[0] = v1 to v2
			if idx < 0 || idx >= len(schemaMigrations) {
				continue
			}
			if migErr := schemaMigrations[idx](ctx, b); migErr != nil {
				return fmt.Errorf("brain: migration v%d to v%d: %w", step, step+1, migErr)
			}
		}

		// Stamp the new version inside the same batch.
		data, marshalErr := yaml.Marshal(schemaVersionFile{Version: SchemaVersion})
		if marshalErr != nil {
			return fmt.Errorf("brain: marshalling schema version: %w", marshalErr)
		}
		return b.Write(ctx, schemaVersionPath, data)
	})
}

// readSchemaVersion returns the version from schema-version.yaml, or 0 if
// the file is missing.
func readSchemaVersion(ctx context.Context, s Store) (int, error) {
	data, err := s.Read(ctx, schemaVersionPath)
	if err != nil {
		if errors.Is(err, ErrNotFound) {
			return 0, nil
		}
		return 0, fmt.Errorf("brain: reading schema version: %w", err)
	}

	var f schemaVersionFile
	if err := yaml.Unmarshal(data, &f); err != nil {
		return 0, fmt.Errorf("brain: parsing schema version: %w", err)
	}
	return f.Version, nil
}

// writeSchemaVersion writes a standalone schema-version.yaml (outside a
// batch). Used only by CheckSchemaVersion for the initial stamp on a
// fresh brain.
func writeSchemaVersion(ctx context.Context, s Store, version int) error {
	data, err := yaml.Marshal(schemaVersionFile{Version: version})
	if err != nil {
		return fmt.Errorf("brain: marshalling schema version: %w", err)
	}
	return s.Write(ctx, schemaVersionPath, data)
}
