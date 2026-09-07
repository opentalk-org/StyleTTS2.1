import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";

import { query } from "@/server/clickhouse";
import { uuidSchema } from "@/shared/ids";
import type { RunLog } from "@/shared/types";

const PAGE_SIZE = 1_000;
const logCursorSchema = z.object({
  timestamp: z.string(),
  runId: uuidSchema,
  message: z.string(),
});
const logsInputSchema = z.object({
  runIds: z.array(uuidSchema),
  before: logCursorSchema.nullable(),
});

interface LogRow {
  runId: string;
  cursorTimestamp: string;
  timestampMs: string;
  message: string;
}

export interface LogCursor {
  timestamp: string;
  runId: string;
  message: string;
}

export interface LogPage {
  rows: RunLog[];
  nextCursor: LogCursor | null;
}

export const getLogs = createServerFn({ method: "POST" })
  .validator(logsInputSchema)
  .handler(async ({ data }): Promise<LogPage> => {
    const before = data.before === null
      ? ""
      : "AND tuple(timestamp, run_id, message) < tuple({before:DateTime64(9)}, {before_run:UUID}, {before_message:String})";
    const rows = await query<LogRow>(`
      SELECT toString(run_id) AS runId, toString(timestamp) AS cursorTimestamp,
        toUnixTimestamp64Milli(timestamp) AS timestampMs, message
      FROM logs
      WHERE run_id IN {run_ids:Array(UUID)} ${before}
      ORDER BY timestamp DESC, run_id DESC, message DESC
      LIMIT {limit:UInt32}`, {
      run_ids: data.runIds,
      before: data.before?.timestamp,
      before_run: data.before?.runId,
      before_message: data.before?.message,
      limit: PAGE_SIZE + 1,
    });
    const pageRows = rows.slice(0, PAGE_SIZE);
    const last = pageRows.at(-1);
    return {
      rows: pageRows.map((row) => ({
        runId: row.runId,
        timestamp: Number(row.timestampMs),
        message: row.message,
      })),
      nextCursor: rows.length > PAGE_SIZE && last !== undefined
        ? { timestamp: last.cursorTimestamp, runId: last.runId, message: last.message }
        : null,
    };
  });
