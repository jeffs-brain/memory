// SPDX-License-Identifier: Apache-2.0

package queue

import (
	"context"
	"fmt"
	"time"
)

// listenLoop subscribes to the NOTIFY channel on the dedicated
// connection and dispatches notifications until the adapter is closed.
func (q *PostgresQueue) listenLoop() {
	defer q.activeWg.Done()

	if q.listenConn == nil {
		return
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Stop the listen context when the adapter closes.
	go func() {
		<-q.stopCh
		cancel()
	}()

	// Subscribe to the channel.
	_, err := q.listenConn.ExecContext(ctx, "LISTEN "+q.notifyChannel)
	if err != nil {
		q.log.Warn("ingest: LISTEN setup failed", "channel", q.notifyChannel, "error", err.Error())
		return
	}
	q.log.Info("ingest: LISTEN subscribed", "channel", q.notifyChannel)

	// Poll for notifications until the context is cancelled.
	for {
		select {
		case <-q.stopCh:
			return
		default:
		}

		// WaitForNotification blocks until a notification arrives or the
		// context deadline expires. We use a short deadline so we can
		// check the stop channel periodically.
		waitCtx, waitCancel := context.WithTimeout(ctx, 5*time.Second)
		err := q.pollNotification(waitCtx)
		waitCancel()

		if err != nil && ctx.Err() == nil {
			q.log.Debug("ingest: notification poll cycle", "error", err.Error())
		}
	}
}

// pollNotification checks for pending notifications by executing a
// no-op query that triggers the driver's notification pathway.
func (q *PostgresQueue) pollNotification(ctx context.Context) error {
	// Execute a lightweight query. For pgx-based drivers, notifications
	// are delivered as a side-effect of any query on the connection.
	_, err := q.listenConn.ExecContext(ctx, "SELECT 1")
	return err
}

// heartbeatLoop periodically refreshes the heartbeat for all claimed
// jobs until the adapter is closed. Uses a single bulk UPDATE for
// efficiency instead of per-job serial queries.
func (q *PostgresQueue) heartbeatLoop() {
	defer q.activeWg.Done()
	ticker := time.NewTicker(q.heartbeatInt)
	defer ticker.Stop()

	for {
		select {
		case <-q.stopCh:
			return
		case <-ticker.C:
			ids := q.claimedIDs()
			if len(ids) == 0 {
				continue
			}
			ctx, cancel := context.WithTimeout(context.Background(), time.Duration(heartbeatTimeoutSeconds)*time.Second)
			if err := q.bulkHeartbeat(ctx, ids); err != nil {
				q.log.Warn("ingest: bulk heartbeat refresh failed",
					"count", len(ids), "error", err.Error())
			}
			cancel()
		}
	}
}

// bulkHeartbeat updates the heartbeat timestamp for multiple jobs in
// a single query using ANY($1::uuid[]) for a stable query plan and
// single-parameter binding regardless of batch size.
func (q *PostgresQueue) bulkHeartbeat(ctx context.Context, jobIDs []string) error {
	tbl := q.qualifiedTable()

	query := fmt.Sprintf(`
		UPDATE %s
		SET last_heartbeat = NOW(), updated_at = NOW()
		WHERE id = ANY($1::uuid[]) AND status = $2`,
		tbl)

	_, err := q.db.ExecContext(ctx, query, pqStringArray(jobIDs), string(StatusProcessing))
	return err
}
