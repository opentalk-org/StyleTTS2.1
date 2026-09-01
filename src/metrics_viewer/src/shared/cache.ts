const PREFIX = "runflow.metrics.cache.v1:";
const TTL_MS = 30 * 60 * 1000;
const MAX_BYTES = 3_700_000;

interface CacheEntry<T> { createdAt: number; accessedAt: number; value: T; }

export async function cached<T>(key: string, load: () => Promise<T>): Promise<T> {
  const storageKey = PREFIX + key;
  const existing = read<T>(storageKey);
  if (existing !== null && Date.now() - existing.createdAt < TTL_MS) {
    existing.accessedAt = Date.now();
    write(storageKey, existing);
    return existing.value;
  }
  localStorage.removeItem(storageKey);
  const value = await load();
  write(storageKey, { createdAt: Date.now(), accessedAt: Date.now(), value });
  evict();
  return value;
}

function read<T>(key: string): CacheEntry<T> | null {
  const serialized = localStorage.getItem(key);
  if (serialized === null) return null;
  try { return JSON.parse(serialized) as CacheEntry<T>; }
  catch { localStorage.removeItem(key); return null; }
}

function write<T>(key: string, entry: CacheEntry<T>): void {
  try { localStorage.setItem(key, JSON.stringify(entry)); }
  catch { evict(true); }
}

function evict(force = false): void {
  const entries = Object.keys(localStorage).filter((key) => key.startsWith(PREFIX)).map((key) => ({ key, value: localStorage.getItem(key) ?? "" })).sort((a, b) => accessTime(a.value) - accessTime(b.value));
  let bytes = entries.reduce((sum, entry) => sum + entry.key.length + entry.value.length, 0) * 2;
  for (const entry of entries) {
    if (!force && bytes <= MAX_BYTES) break;
    localStorage.removeItem(entry.key);
    bytes -= (entry.key.length + entry.value.length) * 2;
    force = false;
  }
}

function accessTime(value: string): number {
  try { return (JSON.parse(value) as CacheEntry<unknown>).accessedAt; }
  catch { return 0; }
}
