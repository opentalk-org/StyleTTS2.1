ALTER TABLE `logs` DROP COLUMN `logger`;
ALTER TABLE `runs`
  MODIFY COLUMN `status` Enum8('running' = 1, 'succeeded' = 2, 'failed' = 3, 'cancelled' = 4, 'queued' = 5),
  DROP COLUMN `started_at`,
  DROP COLUMN `ended_at`,
  ADD COLUMN `data_config` JSON AFTER `status`,
  ADD COLUMN `train_config` JSON AFTER `data_config`,
  ADD COLUMN `started_at_unix_ms` Int64 AFTER `train_config`,
  ADD COLUMN `ended_at_unix_ms` Int64 AFTER `started_at_unix_ms`;
