-- Create "array_metrics" table
CREATE TABLE `array_metrics` (
  `timestamp` DateTime64(9),
  `run_id` UUID,
  `step` UInt64,
  `name` LowCardinality(String),
  `value` Array(Float32)
) ENGINE = MergeTree
PRIMARY KEY (`run_id`, `name`, `step`, `timestamp`)
ORDER BY (`run_id`, `name`, `step`, `timestamp`)
PARTITION BY (toYYYYMM(timestamp))
SETTINGS index_granularity = 8192;
