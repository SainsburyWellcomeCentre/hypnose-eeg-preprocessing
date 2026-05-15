# Restart and Recovery Notes

## Goal
Resume continuous EEG processing without reprocessing or data loss.

## Procedure
1. On startup, load checkpoint from configured checkpoint URI.
2. Re-open source at the checkpoint cursor.
3. Skip already processed records up to `last_offset`.
4. Continue processing and writing to sink.
5. Flush checkpoint every N batches and on graceful shutdown.

## Failure handling
- If sink write fails, do not advance checkpoint.
- If processing fails, log bad chunk and apply retry policy.
- If retries are exhausted, park the chunk in a dead-letter path.
