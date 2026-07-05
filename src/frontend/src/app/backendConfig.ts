export function defaultBackendUrl(): string {
  const configuredUrl = import.meta.env.VITE_BACKEND_URL;
  if (configuredUrl) return configuredUrl;
  if (import.meta.env.DEV) return "http://127.0.0.1:8000";
  return window.location.origin;
}
