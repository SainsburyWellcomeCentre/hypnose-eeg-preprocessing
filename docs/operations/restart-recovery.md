# Restart and Recovery Notes

## Goal
Resume EDF recording processing without reprocessing completed recordings.

## Procedure
1. On startup, load checkpoint from configured checkpoint URI.
2. Re-open source at the checkpoint cursor.
3. Skip EDF files up to and including `last_file`.
4. Continue processing the next whole recording and writing to sink.
5. Flush checkpoint after each recording and on graceful shutdown.

## Failure handling
- If sink write fails, do not advance checkpoint.
- If processing fails, log the bad recording and apply retry policy.
- If retries are exhausted, park the recording in a dead-letter path.
