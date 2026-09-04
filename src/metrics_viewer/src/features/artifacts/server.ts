import { createReadStream } from "node:fs";
import { readFile, stat } from "node:fs/promises";
import { extname, resolve, sep } from "node:path";
import { Readable } from "node:stream";

export async function artifactResponse(runId: string, path: string) {
  const target = artifactPath(runId, path);
  const metadata = await stat(target);
  const stream = Readable.toWeb(createReadStream(target));
  return new Response(stream as ReadableStream, {
    headers: {
      "content-length": String(metadata.size),
      "content-type": contentType(target),
    },
  });
}

export async function readArtifactJson<T>(runId: string, path: string) {
  return JSON.parse(await readFile(artifactPath(runId, path), "utf8")) as T;
}

export function artifactKind(contentType: string, name: string) {
  if (contentType.startsWith("audio/")) return "audio";
  if (contentType.startsWith("image/")) return "image";
  if (contentType.includes("json") || contentType.includes("plotly") || name.endsWith(".plot")) return "plot";
  return "text";
}

function artifactPath(runId: string, path: string) {
  const root = resolve(process.env.METRICS_DIR!, runId);
  const target = resolve(root, path);
  if (!target.startsWith(`${root}${sep}`)) throw new Error("Artifact path is outside its run directory");
  return target;
}

function contentType(path: string) {
  const contentTypes: Record<string, string> = {
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac", ".ogg": "audio/ogg",
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
    ".svg": "image/svg+xml", ".json": "application/json",
  };
  return contentTypes[extname(path).toLowerCase()] ?? "text/plain; charset=utf-8";
}
