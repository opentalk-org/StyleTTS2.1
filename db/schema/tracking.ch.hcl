table "projects" {
  schema = schema.default
  engine = sql("ReplacingMergeTree(version)")

  column "id" {
    type = UUID
  }
  column "name" {
    type = String
  }
  column "description" {
    type = String
  }
  column "created_at" {
    type = DateTime64(3, "UTC")
  }
  column "version" {
    type = UInt64
  }

  primary_key {
    columns = [column.id]
  }
  sort {
    columns = [column.id]
  }
}

table "runs" {
  schema = schema.default
  engine = sql("ReplacingMergeTree(version)")

  column "id" {
    type = UUID
  }
  column "project_id" {
    type = UUID
  }
  column "name" {
    type = String
  }
  column "status" {
    type = sql("Enum8('running' = 1, 'succeeded' = 2, 'failed' = 3, 'cancelled' = 4)")
  }
  column "started_at" {
    type = DateTime64(3, "UTC")
  }
  column "ended_at" {
    null = true
    type = sql("Nullable(DateTime64(3, 'UTC'))")
  }
  column "version" {
    type = UInt64
  }

  primary_key {
    columns = [column.project_id, column.id]
  }
  sort {
    columns = [column.project_id, column.id]
  }
}

table "run_params" {
  schema = schema.default
  engine = sql("ReplacingMergeTree")

  column "run_id" {
    type = UUID
  }
  column "key" {
    type = String
  }
  column "value" {
    type = String
  }

  primary_key {
    columns = [column.run_id, column.key]
  }
  sort {
    columns = [column.run_id, column.key]
  }
}

table "metrics" {
  schema = schema.default
  engine = MergeTree

  column "run_id" {
    type = UUID
  }
  column "step" {
    type = UInt64
  }
  column "timestamp_unix_ms" {
    type = Int64
  }
  column "name" {
    type = sql("LowCardinality(String)")
  }
  column "value" {
    type = Float32
  }

  partition {
    columns = [column.run_id]
  }
  primary_key {
    columns = [column.run_id, column.name, column.step, column.timestamp_unix_ms]
  }
  sort {
    columns = [column.run_id, column.name, column.step, column.timestamp_unix_ms]
  }
}

table "logs" {
  schema = schema.default
  engine = MergeTree

  column "run_id" {
    type = UUID
  }
  column "timestamp_unix_ms" {
    type = Int64
  }
  column "level" {
    type = sql("Enum8('debug' = 1, 'info' = 2, 'warning' = 3, 'error' = 4, 'critical' = 5)")
  }
  column "logger" {
    type = sql("LowCardinality(String)")
  }
  column "message" {
    type = String
  }

  partition {
    columns = [column.run_id]
  }
  primary_key {
    columns = [column.run_id, column.timestamp_unix_ms]
  }
  sort {
    columns = [column.run_id, column.timestamp_unix_ms]
  }
}

table "artifacts" {
  schema = schema.default
  engine = MergeTree

  column "run_id" {
    type = UUID
  }
  column "step" {
    type = UInt64
  }
  column "timestamp_unix_ms" {
    type = Int64
  }
  column "name" {
    type = String
  }
  column "path" {
    type = String
  }
  column "content_type" {
    type = sql("LowCardinality(String)")
  }
  column "size_bytes" {
    type = UInt64
  }

  primary_key {
    columns = [column.run_id, column.path, column.step, column.name, column.timestamp_unix_ms]
  }
  sort {
    columns = [column.run_id, column.path, column.step, column.name, column.timestamp_unix_ms]
  }
}
