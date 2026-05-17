// SPDX-License-Identifier: Apache-2.0

package connector

import (
	"context"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"io"

	"github.com/jeffs-brain/memory/go/brain"
)

// Token store sentinel errors.
var (
	// ErrTokenNotFound is returned when no token exists for the given
	// connector/brain combination.
	ErrTokenNotFound = errors.New("connector: token not found")

	// ErrDecryptionFailed is returned when the stored token cannot be
	// decrypted (wrong key, corrupted data, or tampered ciphertext).
	ErrDecryptionFailed = errors.New("connector: token decryption failed")

	// ErrInvalidEncryptionKey is returned when the encryption key does
	// not meet the required length.
	ErrInvalidEncryptionKey = errors.New("connector: invalid encryption key")
)

// minEncryptionKeyLength is the minimum byte length for the encryption
// passphrase before it is derived into an AES-256 key.
const minEncryptionKeyLength = 16

// TokenStore persists OAuth2 tokens with encryption at rest. Tokens are
// scoped by (connectorName, brainID) to prevent cross-brain leakage.
type TokenStore interface {
	// Save encrypts and persists a token for the given connector and brain.
	Save(ctx context.Context, connectorName, brainID string, token OAuth2Token) error

	// Load retrieves and decrypts the token for the given connector and
	// brain. Returns (zero, false, nil) when no token exists.
	Load(ctx context.Context, connectorName, brainID string) (OAuth2Token, bool, error)

	// Delete removes the stored token. No error if no token exists.
	Delete(ctx context.Context, connectorName, brainID string) error
}

// encryptedTokenEnvelope is the JSON structure stored at rest.
type encryptedTokenEnvelope struct {
	// Nonce is the AES-GCM nonce (12 bytes).
	Nonce []byte `json:"nonce"`
	// Ciphertext is the AES-256-GCM encrypted token JSON with appended
	// authentication tag.
	Ciphertext []byte `json:"ciphertext"`
}

// SecureTokenStore implements TokenStore using AES-256-GCM encryption.
// Tokens are stored in the brain's Store at:
//
//	connector/<name>/oauth-token.enc.json
type SecureTokenStore struct {
	store brain.Store
	key   [32]byte
}

// NewSecureTokenStore creates a SecureTokenStore. The passphrase is
// derived into a 256-bit key using SHA-256. The passphrase must be at
// least 16 bytes long.
//
// V1 limitation: single-iteration SHA-256 is fast and acceptable when
// the passphrase is a long, randomly generated environment variable.
// Before supporting user-chosen passphrases, upgrade to PBKDF2
// (>=100,000 iterations) or scrypt/argon2 for brute-force resistance.
func NewSecureTokenStore(store brain.Store, passphrase []byte) (*SecureTokenStore, error) {
	if len(passphrase) < minEncryptionKeyLength {
		return nil, fmt.Errorf(
			"%w: passphrase must be at least %d bytes, got %d",
			ErrInvalidEncryptionKey, minEncryptionKeyLength, len(passphrase),
		)
	}
	key := sha256.Sum256(passphrase)
	return &SecureTokenStore{store: store, key: key}, nil
}

func tokenPath(connectorName, brainID string) brain.Path {
	return brain.Path(fmt.Sprintf("connector/%s/%s/oauth-token.enc.json", connectorName, brainID))
}

// Save encrypts and persists the token.
func (s *SecureTokenStore) Save(ctx context.Context, connectorName, brainID string, token OAuth2Token) error {
	plaintext, err := json.Marshal(token)
	if err != nil {
		return fmt.Errorf("connector: token marshal: %w", err)
	}

	ciphertext, nonce, err := encryptAES256GCM(s.key[:], plaintext)
	if err != nil {
		return fmt.Errorf("connector: token encrypt: %w", err)
	}

	envelope := encryptedTokenEnvelope{
		Nonce:      nonce,
		Ciphertext: ciphertext,
	}
	data, err := json.MarshalIndent(envelope, "", "  ")
	if err != nil {
		return fmt.Errorf("connector: envelope marshal: %w", err)
	}

	if err := s.store.Write(ctx, tokenPath(connectorName, brainID), data); err != nil {
		return fmt.Errorf("connector: token store write: %w", err)
	}
	return nil
}

// Load retrieves and decrypts the token.
func (s *SecureTokenStore) Load(ctx context.Context, connectorName, brainID string) (OAuth2Token, bool, error) {
	data, err := s.store.Read(ctx, tokenPath(connectorName, brainID))
	if err != nil {
		if errors.Is(err, brain.ErrNotFound) {
			return OAuth2Token{}, false, nil
		}
		return OAuth2Token{}, false, fmt.Errorf("connector: token store read: %w", err)
	}

	var envelope encryptedTokenEnvelope
	if err := json.Unmarshal(data, &envelope); err != nil {
		return OAuth2Token{}, false, fmt.Errorf("connector: envelope unmarshal: %w", err)
	}

	plaintext, err := decryptAES256GCM(s.key[:], envelope.Nonce, envelope.Ciphertext)
	if err != nil {
		return OAuth2Token{}, false, fmt.Errorf("%w: %v", ErrDecryptionFailed, err)
	}

	var token OAuth2Token
	if err := json.Unmarshal(plaintext, &token); err != nil {
		return OAuth2Token{}, false, fmt.Errorf("connector: token unmarshal: %w", err)
	}
	return token, true, nil
}

// Delete removes the stored token.
func (s *SecureTokenStore) Delete(ctx context.Context, connectorName, brainID string) error {
	err := s.store.Delete(ctx, tokenPath(connectorName, brainID))
	if err != nil && !errors.Is(err, brain.ErrNotFound) {
		return fmt.Errorf("connector: token store delete: %w", err)
	}
	return nil
}

// encryptAES256GCM encrypts plaintext using AES-256-GCM. Returns the
// ciphertext (with appended auth tag) and the randomly generated nonce.
func encryptAES256GCM(key, plaintext []byte) (ciphertext, nonce []byte, err error) {
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, nil, err
	}

	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, nil, err
	}

	nonce = make([]byte, gcm.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return nil, nil, fmt.Errorf("nonce generation: %w", err)
	}

	ciphertext = gcm.Seal(nil, nonce, plaintext, nil)
	return ciphertext, nonce, nil
}

// decryptAES256GCM decrypts ciphertext produced by encryptAES256GCM.
func decryptAES256GCM(key, nonce, ciphertext []byte) ([]byte, error) {
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}

	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}

	plaintext, err := gcm.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return nil, err
	}
	return plaintext, nil
}

// timingSafeEqual performs a constant-time comparison of two byte slices.
// This prevents timing attacks when comparing authentication tags or HMAC
// values.
func timingSafeEqual(a, b []byte) bool {
	return subtle.ConstantTimeCompare(a, b) == 1
}

var _ TokenStore = (*SecureTokenStore)(nil)
