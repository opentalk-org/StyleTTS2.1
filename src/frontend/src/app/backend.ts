import { useNav } from "./navStore";

function backendUrl(path: string, base: string = useNav.getState().backendUrl): string {
  const baseUrl = new URL(base);
  return new URL(path, baseUrl.origin).toString();
}

export function backendResourceUrl(path: string, base?: string): string {
  return backendUrl(path, base);
}

export function backendWebSocketUrl(path: string, base?: string): string {
  const url = new URL(backendUrl(path, base));
  if (url.protocol === "https:") {
    url.protocol = "wss:";
    return url.toString();
  }
  if (url.protocol === "http:") {
    url.protocol = "ws:";
    return url.toString();
  }
  throw new Error(`Unsupported backend protocol: ${url.protocol}`);
}

export function backendFetch(path: string, init?: RequestInit): Promise<Response> {
  return fetch(backendUrl(path), init);
}

export async function backendRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await backendFetch(path, init);
  if (!response.ok) throw new Error(`Backend request failed: ${response.status}`);
  return response.json() as Promise<T>;
}
