schema "default" {
  engine = sql("Shared")
}

table "projects" {
  schema = schema.default
  engine = sql("ReplacingMergeTree(updated_at)")

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
    type = DateTime64(6)
  }
  column "updated_at" {
    type = DateTime64(6)
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
  engine = MergeTree

  column "id" {
    type = UUID
  }
  column "project_id" {
    type = UUID
  }
  column "name" {
    type = String
  }
  column "data_config" {
    type = JSON
  }
  column "train_config" {
    type = JSON
  }

  primary_key {
    columns = [column.project_id, column.id]
  }
  sort {
    columns = [column.project_id, column.id]
  }
}

table "run_status" {
  schema = schema.default
  engine = MergeTree

  column "timestamp" {
    type = DateTime64(9)
  }
  column "run_id" {
    type = UUID
  }
  column "status" {
    type = sql("Enum8('running' = 1, 'succeeded' = 2, 'failed' = 3, 'cancelled' = 4, 'queued' = 5)")
  }

  sort {
    columns = [column.run_id, column.timestamp]
  }
  primary_key {
    columns = [column.run_id, column.timestamp]
  }
}

table "metrics" {
  schema = schema.default
  engine = MergeTree

  column "timestamp" {
    type = DateTime64(9)
  }
  column "run_id" {
    type = UUID
  }
  column "step" {
    type = UInt64
  }
  column "name" {
    type = sql("LowCardinality(String)")
  }
  column "value" {
    type = Float32
  }

  partition {
    on {
      expr = "toYYYYMM(timestamp)"
    }
  }
  primary_key {
    columns = [column.run_id, column.name, column.step, column.timestamp]
  }
  sort {
    columns = [column.run_id, column.name, column.step, column.timestamp]
  }
}

table "array_metrics" {
  schema = schema.default
  engine = MergeTree

  column "timestamp" {
    type = DateTime64(9)
  }
  column "run_id" {
    type = UUID
  }
  column "step" {
    type = UInt64
  }
  column "name" {
    type = sql("LowCardinality(String)")
  }
  column "value" {
    type = sql("Array(Float32)")
  }

  partition {
    on {
      expr = "toYYYYMM(timestamp)"
    }
  }
  primary_key {
    columns = [column.run_id, column.name, column.step, column.timestamp]
  }
  sort {
    columns = [column.run_id, column.name, column.step, column.timestamp]
  }
}

table "logs" {
  schema = schema.default
  engine = MergeTree

  column "run_id" {
    type = UUID
  }
  column "timestamp" {
    type = DateTime64(9)
  }
  column "message" {
    type = String
  }

  partition {
    on {
      expr = "toYYYYMM(timestamp)"
    }
  }
  primary_key {
    columns = [column.run_id, column.timestamp]
  }
  sort {
    columns = [column.run_id, column.timestamp]
  }
}

table "artifacts" {
  schema = schema.default
  engine = MergeTree

  column "id" {
    type = UUID
  }
  column "run_id" {
    type = UUID
  }
  column "step" {
    type = UInt64
  }
  column "timestamp" {
    type = DateTime64(9)
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
    columns = [column.id]
  }
  sort {
    columns = [column.id]
  }
}

table "audio_files" {
  schema = schema.default
  engine = MergeTree

  column "id" {
    type = UUID
  }
  column "updated_at" {
    type = DateTime64(9)
  }
  column "name" {
    type = String
  }
  column "bucket_file_id" {
    type = sql("Nullable(UUID)")
  }
  column "byte_offset" {
    type = UInt64
  }
  column "duration" {
    type = Float64
  }
  column "byte_length" {
    type = UInt64
  }
  column "score" {
    type = sql("Nullable(Float32)")
  }
  column "language" {
    type = sql("LowCardinality(Nullable(String))")
  }
  column "style_prompt" {
    type = sql("Nullable(String)")
  }
  column "voice_prompt" {
    type = sql("Nullable(String)")
  }
  column "virtual" {
    type = Bool
  }
  column "storage_kind" {
    type = sql("Enum8('packed' = 1, 'external' = 2)")
  }
  column "storage_ref" {
    type = sql("Nullable(JSON)")
  }
  column "metadata" {
    type = sql("JSON")
  }

  primary_key {
    columns = [column.id]
  }
  sort {
    columns = [column.id, column.updated_at]
  }
}

table "audio_segments" {
  schema = schema.default
  engine = MergeTree

  column "id" {
    type = String
  }
  column "audio_file_id" {
    type = UUID
  }
  column "updated_at" {
    type = DateTime64(9)
  }
  column "position" {
    type = UInt32
  }
  column "start_seconds" {
    type = Float64
  }
  column "end_seconds" {
    type = Float64
  }
  column "text" {
    type = String
  }
  column "phon" {
    type = String
  }
  column "kind" {
    type = sql("LowCardinality(String)")
  }
  column "accuracy" {
    type = sql("Nullable(Float32)")
  }
  column "speaker_id" {
    type = sql("Nullable(String)")
  }
  column "metadata" {
    type = sql("JSON")
  }
  column "alignment" {
    type = sql("JSON")
  }

  primary_key {
    columns = [column.audio_file_id, column.id]
  }
  sort {
    columns = [column.audio_file_id, column.id, column.updated_at]
  }
}

table "dataset_audio_files" {
  schema = schema.default
  engine = sql("ReplacingMergeTree(updated_at)")

  column "dataset_id" {
    type = UUID
  }
  column "audio_file_id" {
    type = UUID
  }
  column "updated_at" {
    type = DateTime64(6)
  }

  primary_key {
    columns = [column.dataset_id, column.audio_file_id]
  }
  sort {
    columns = [column.dataset_id, column.audio_file_id]
  }
}

table "datasets" {
  schema = schema.default
  engine = sql("ReplacingMergeTree(updated_at)")

  column "id" {
    type = UUID
  }
  column "updated_at" {
    type = DateTime64(6)
  }
  column "name" {
    type = String
  }

  primary_key {
    columns = [column.id]
  }
  sort {
    columns = [column.id]
  }
}

table "bucket_files" {
  schema = schema.default
  engine = MergeTree

  column "id" {
    type = UUID
  }
  column "kind" {
    type = sql("Enum8('audio' = 1, 'waveform' = 2)")
  }
  column "path" {
    type = String
  }
  column "size" {
    type = UInt64
  }

  primary_key {
    columns = [column.id]
  }
  sort {
    columns = [column.id]
  }
}

table "assets" {
  schema = schema.default
  engine = sql("ReplacingMergeTree(updated_at)")

  column "id" {
    type = UUID
  }
  column "updated_at" {
    type = DateTime64(6)
  }
  column "kind" {
    type = sql("Enum8('checkpoint' = 1, 'file' = 2)")
  }
  column "name" {
    type = String
  }
  column "path" {
    type = String
  }
  column "size" {
    type = UInt64
  }
  column "content_hash" {
    type = FixedString(64)
  }
  column "type" {
    type = sql("LowCardinality(String)")
  }
  column "metadata" {
    type = sql("JSON")
  }
  column "run_id" {
    type = sql("Nullable(UUID)")
  }

  primary_key {
    columns = [column.id]
  }
  sort {
    columns = [column.id]
  }
}
