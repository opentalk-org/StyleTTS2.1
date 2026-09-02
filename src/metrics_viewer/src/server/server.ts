import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { extname, resolve, sep } from "node:path";

import { Database } from "./database.js";

const database = new Database({
  url: required("CLICKHOUSE_HTTP_URL"),
  username: required("CLICKHOUSE_USER"),
  password: required("CLICKHOUSE_PASSWORD"),
});
const metricsDir = resolve(required("METRICS_DIR"));
const port = Number(process.env.METRICS_VIEWER_PORT ?? "8182");

createServer(async (request, response) => {
  try {
    await route(request, response);
  } catch (error) {
    console.error(error);
    json(response, 500, { detail: error instanceof Error ? error.message : String(error) });
  }
}).listen(port, "0.0.0.0", () => {
  console.log(`metrics viewer backend listening on http://0.0.0.0:${port}`);
});

async function route(request: IncomingMessage, response: ServerResponse) {
  const url = new URL(request.url ?? "/", `http://${request.headers.host}`);
  if (request.method === "GET" && url.pathname === "/api/projects") {
    return json(response, 200, await database.projects());
  }
  const runs = url.pathname.match(/^\/api\/projects\/([^/]+)\/runs$/);
  if (request.method === "GET" && runs) {
    return json(response, 200, await database.runs(decodeURIComponent(runs[1])));
  }
  const plots = url.pathname.match(/^\/api\/projects\/([^/]+)\/plots$/);
  if (request.method === "POST" && plots) {
    const body = await requestBody(request) as { sql: string; runIds: string[] };
    const sql = body.sql.trim().replace(/;+$/, "");
    if (!/^(SELECT|WITH)\b/i.test(sql)) return json(response, 400, { detail: "Plot query must start with SELECT or WITH" });
    return json(response, 200, await database.plots(
      sql,
      decodeURIComponent(plots[1]),
      body.runIds,
    ));
  }
  if (request.method === "GET" && url.pathname === "/api/artifacts") {
    const runIds = (url.searchParams.get("run_ids") ?? "").split(",").filter(Boolean);
    const rows = await database.artifacts(runIds);
    return json(response, 200, rows.map((row) => ({
      id: `${row.runId}-${row.name}-${row.step}`,
      runId: row.runId,
      name: row.name,
      step: Number(row.step),
      timestamp: Number(row.timestamp),
      kind: artifactKind(row.contentType, row.name),
      contentType: row.contentType,
      sizeBytes: Number(row.sizeBytes),
      source: `/api/artifacts/content?run_id=${encodeURIComponent(row.runId)}&path=${encodeURIComponent(row.path)}`,
    })));
  }
  if (request.method === "GET" && url.pathname === "/api/artifacts/content") {
    return artifact(response, url.searchParams.get("run_id") ?? "", url.searchParams.get("path") ?? "");
  }
  const modelGraph = url.pathname.match(/^\/api\/runs\/([^/]+)\/model-graph$/);
  if (request.method === "GET" && modelGraph) {
    const row = await database.modelGraphArtifact(decodeURIComponent(modelGraph[1]));
    if (row === undefined) return json(response, 404, { detail: "Not found" });
    return artifact(response, row.runId, row.path);
  }
  const arrayNames = url.pathname.match(/^\/api\/runs\/([^/]+)\/array-metrics\/names$/);
  if (request.method === "GET" && arrayNames) {
    return json(response, 200, await database.arrayMetricNames(decodeURIComponent(arrayNames[1])));
  }
  const arraySeries = url.pathname.match(/^\/api\/runs\/([^/]+)\/array-metrics$/);
  if (request.method === "GET" && arraySeries) {
    const name = url.searchParams.get("name");
    if (name === null) return json(response, 400, { detail: "name is required" });
    return json(response, 200, await database.arrayMetric(decodeURIComponent(arraySeries[1]), name));
  }
  json(response, 404, { detail: "Not found" });
}

async function requestBody(request: IncomingMessage) {
  const chunks: Buffer[] = [];
  for await (const chunk of request) chunks.push(Buffer.from(chunk));
  return JSON.parse(Buffer.concat(chunks).toString("utf8")) as unknown;
}

async function artifact(response: ServerResponse, runId: string, path: string) {
  const root = resolve(metricsDir, runId);
  const target = resolve(root, path);
  if (!target.startsWith(`${root}${sep}`)) return json(response, 404, { detail: "Not found" });
  const metadata = await stat(target);
  response.writeHead(200, {
    "content-length": metadata.size,
    "content-type": contentType(target),
  });
  createReadStream(target).pipe(response);
}

function json(response: ServerResponse, status: number, body: unknown) {
  response.writeHead(status, { "content-type": "application/json" });
  response.end(JSON.stringify(body));
}

function artifactKind(contentType: string, name: string) {
  if (contentType.startsWith("audio/")) return "audio";
  if (contentType.startsWith("image/")) return "image";
  if (contentType.includes("json") || contentType.includes("plotly") || name.endsWith(".plot")) return "plot";
  return "text";
}

function contentType(path: string) {
  return ({
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".json": "application/json",
  } as Record<string, string>)[extname(path).toLowerCase()] ?? "text/plain; charset=utf-8";
}

function required(name: string) {
  const value = process.env[name];
  if (value === undefined) throw new Error(`${name} is required`);
  return value;
}
