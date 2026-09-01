-- Create "artifacts" table
CREATE TABLE `artifacts` (
  `run_id` UUID,
  `step` UInt64,
  `timestamp_unix_ms` Int64,
  `name` String,
  `path` String,
  `content_type` LowCardinality(String),
  `size_bytes` UInt64
) ENGINE = MergeTree
PRIMARY KEY (`run_id`, `path`, `step`, `name`, `timestamp_unix_ms`) ORDER BY (`run_id`, `path`, `step`, `name`, `timestamp_unix_ms`) SETTINGS index_granularity = 8192;
-- Create "logs" table
CREATE TABLE `logs` (
  `run_id` UUID,
  `timestamp_unix_ms` Int64,
  `level` Enum8('debug' = 1, 'info' = 2, 'warning' = 3, 'error' = 4, 'critical' = 5),
  `logger` LowCardinality(String),
  `message` String
) ENGINE = MergeTree
PRIMARY KEY (`run_id`, `timestamp_unix_ms`) ORDER BY (`run_id`, `timestamp_unix_ms`) PARTITION BY (`run_id`) SETTINGS index_granularity = 8192;
-- Create "metrics" table
CREATE TABLE `metrics` (
  `run_id` UUID,
  `step` UInt64,
  `timestamp_unix_ms` Int64,
  `name` LowCardinality(String),
  `value` Float32
) ENGINE = MergeTree
PRIMARY KEY (`run_id`, `name`, `step`, `timestamp_unix_ms`) ORDER BY (`run_id`, `name`, `step`, `timestamp_unix_ms`) PARTITION BY (`run_id`) SETTINGS index_granularity = 8192;
-- Create "projects" table
CREATE TABLE `projects` (
  `id` UUID,
  `name` String,
  `description` String,
  `created_at` DateTime64(3, 'UTC'),
  `version` UInt64
) ENGINE = ReplacingMergeTree(version)
PRIMARY KEY (`id`) ORDER BY (`id`) SETTINGS index_granularity = 8192;
-- Create "run_params" table
CREATE TABLE `run_params` (
  `run_id` UUID,
  `key` String,
  `value` String
) ENGINE = ReplacingMergeTree
PRIMARY KEY (`run_id`, `key`) ORDER BY (`run_id`, `key`) SETTINGS index_granularity = 8192;
-- Create "run_tags" table
CREATE TABLE `run_tags` (
  `run_id` UUID,
  `key` String,
  `value` String,
  `version` UInt64
) ENGINE = ReplacingMergeTree(version)
PRIMARY KEY (`run_id`, `key`) ORDER BY (`run_id`, `key`) SETTINGS index_granularity = 8192;
-- Create "runs" table
CREATE TABLE `runs` (
  `id` UUID,
  `project_id` UUID,
  `name` String,
  `status` Enum8('running' = 1, 'succeeded' = 2, 'failed' = 3, 'cancelled' = 4),
  `started_at` DateTime64(3, 'UTC'),
  `ended_at` Nullable(DateTime64(3, 'UTC')),
  `version` UInt64
) ENGINE = ReplacingMergeTree(version)
PRIMARY KEY (`project_id`, `id`) ORDER BY (`project_id`, `id`) SETTINGS index_granularity = 8192;
