import type { MetricPoint, PlotQueryResult, PlotRow } from "@/shared/types";

/**
 * Stand-in for ClickHouse. The real backend would forward the SQL untouched; this
 * evaluates the documented subset locally so the viewer can be driven by a query
 * without a server. It understands:
 *
 *   SELECT <col> AS plot, run_id, <col> AS x, <col> AS y
 *   FROM metrics(names => [...], runs => selected())
 *   [WHERE x BETWEEN a AND b]
 *   [LIMIT n]
 *
 * Anything outside that shape comes back as an error naming what it could not read,
 * rather than silently plotting nothing.
 */

export class QueryError extends Error {}

/** Columns a row of `metrics(...)` exposes to the SELECT list. */
type SourceColumn = "name" | "run_id" | "step" | "timestamp" | "value";

const SOURCE_COLUMNS: SourceColumn[] = ["name", "run_id", "step", "timestamp", "value"];

export interface QueryContext {
  /** Run ids `selected()` resolves to. */
  selectedRunIds: string[];
  /** Run ids `all_runs()` resolves to. */
  projectRunIds: string[];
  /** Loads the points behind `metrics(...)`. */
  loadPoints: (runIds: string[], names: string[]) => MetricPoint[];
}

interface ParsedQuery {
  projections: Record<"plot" | "x" | "y", SourceColumn>;
  names: string[];
  runIds: string[];
  range: { min: number; max: number } | null;
  limit: number | null;
}

export function runPlotQuery(sql: string, context: QueryContext): PlotQueryResult {
  const started = performance.now();
  const parsed = parse(sql, context);
  const points = context.loadPoints(parsed.runIds, parsed.names);

  const rows: PlotRow[] = [];
  for (const point of points) {
    const x = numberOf(point, parsed.projections.x);
    if (parsed.range !== null && (x < parsed.range.min || x > parsed.range.max)) continue;
    rows.push({
      plot: String(valueOf(point, parsed.projections.plot)),
      runId: point.runId,
      x,
      y: numberOf(point, parsed.projections.y),
    });
    if (parsed.limit !== null && rows.length >= parsed.limit) break;
  }

  return {
    rows,
    elapsedMs: Math.round(performance.now() - started),
    readRows: points.length,
    xLabel: parsed.projections.x === "timestamp" ? "wall time" : parsed.projections.x,
  };
}

function valueOf(point: MetricPoint, column: SourceColumn): string | number {
  if (column === "name") return point.name;
  if (column === "run_id") return point.runId;
  if (column === "step") return point.step;
  if (column === "timestamp") return point.timestamp;
  return point.value;
}

function numberOf(point: MetricPoint, column: SourceColumn): number {
  const value = valueOf(point, column);
  if (typeof value !== "number") {
    throw new QueryError(`Column "${column}" is not numeric and cannot be plotted.`);
  }
  return value;
}

function parse(sql: string, context: QueryContext): ParsedQuery {
  const normalised = sql.replace(/--[^\n]*/g, " ").replace(/\s+/g, " ").trim();
  if (!/^select\b/i.test(normalised)) {
    throw new QueryError("A plots query has to start with SELECT.");
  }

  const from = /\bfrom\b/i.exec(normalised);
  if (from === null) throw new QueryError("Missing a FROM clause.");

  const projections = parseProjections(normalised.slice(6, from.index));
  const source = normalised.slice(from.index + 4);

  const call = /metrics\s*\(([\s\S]*?)\)\s*(?:where|order|limit|$)/i.exec(`${source} `);
  if (call === null) {
    throw new QueryError("FROM must read from metrics(names => [...], runs => selected()).");
  }

  return {
    projections,
    names: parseNames(call[1]),
    runIds: parseRuns(call[1], context),
    range: parseRange(source),
    limit: parseLimit(source),
  };
}

/** `expr AS alias` pairs; plot, x and y are required, run_id is implicit. */
function parseProjections(list: string): ParsedQuery["projections"] {
  const found: Partial<Record<string, SourceColumn>> = {};
  for (const item of splitTopLevel(list)) {
    const aliased = /^(.+?)\s+as\s+([a-z_]+)$/i.exec(item.trim());
    if (aliased === null) continue;
    const source = aliased[1].trim().toLowerCase();
    const alias = aliased[2].toLowerCase();
    if (!SOURCE_COLUMNS.includes(source as SourceColumn)) {
      throw new QueryError(
        `Unknown column "${source}". Available: ${SOURCE_COLUMNS.join(", ")}.`,
      );
    }
    found[alias] = source as SourceColumn;
  }
  for (const alias of ["plot", "x", "y"]) {
    if (found[alias] === undefined) {
      throw new QueryError(`The SELECT list needs a column aliased AS ${alias}.`);
    }
  }
  return found as ParsedQuery["projections"];
}

function parseNames(args: string): string[] {
  const list = /names\s*=>\s*\[([^\]]*)\]/i.exec(args);
  if (list === null) throw new QueryError("metrics(...) needs names => ['metric', …].");
  const names = [...list[1].matchAll(/'([^']*)'/g)].map((match) => match[1]).filter(Boolean);
  if (names.length === 0) throw new QueryError("names => [] does not select any metric.");
  return names;
}

function parseRuns(args: string, context: QueryContext): string[] {
  const runs = /runs\s*=>\s*([\s\S]*?)(?:,\s*[a-z_]+\s*=>|$)/i.exec(args);
  const expression = (runs?.[1] ?? "selected()").trim();
  if (/^selected\s*\(\s*\)$/i.test(expression)) return context.selectedRunIds;
  if (/^all_runs\s*\(\s*\)$/i.test(expression)) return context.projectRunIds;
  const explicit = [...expression.matchAll(/'([^']*)'/g)].map((match) => match[1]);
  if (explicit.length > 0) return explicit;
  throw new QueryError(`Cannot read runs => ${expression}. Use selected(), all_runs() or a list.`);
}

function parseRange(source: string): ParsedQuery["range"] {
  const between = /\bwhere\s+x\s+between\s+(-?[\d.e]+)\s+and\s+(-?[\d.e]+)/i.exec(source);
  if (between === null) return null;
  return { min: Number(between[1]), max: Number(between[2]) };
}

function parseLimit(source: string): number | null {
  const limit = /\blimit\s+(\d+)/i.exec(source);
  return limit === null ? null : Number(limit[1]);
}

/** Splits a SELECT list on commas that are not inside brackets or quotes. */
function splitTopLevel(list: string): string[] {
  const parts: string[] = [];
  let depth = 0;
  let quoted = false;
  let current = "";
  for (const character of list) {
    if (character === "'") quoted = !quoted;
    if (!quoted && (character === "(" || character === "[")) depth += 1;
    if (!quoted && (character === ")" || character === "]")) depth -= 1;
    if (character === "," && depth === 0 && !quoted) {
      parts.push(current);
      current = "";
      continue;
    }
    current += character;
  }
  parts.push(current);
  return parts;
}
