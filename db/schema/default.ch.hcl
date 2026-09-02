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
    columns = [column.run_id, column.path, column.step, column.name, column.timestamp]
  }
  sort {
    columns = [column.run_id, column.path, column.step, column.name, column.timestamp]
  }
}
