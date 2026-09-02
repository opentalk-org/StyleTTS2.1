
export function metricsApiUrl(): string {
  const configured = import.meta.env.VITE_METRICS_API_URL;
  if (configured) return configured;

  return "";
}


export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${metricsApiUrl()}${path}`, {
    headers: { "content-type": "application/json" },
    ...init,
  });
  if (!response.ok) throw new Error(await errorMessage(response));
  return (await response.json()) as T;
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") return body.detail;
  } catch {

  }
  return `${response.status} ${response.statusText}`;
}
