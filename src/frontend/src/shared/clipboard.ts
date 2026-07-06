/**
 * Copy text to the clipboard. Uses the async Clipboard API when available and
 * falls back to a hidden-textarea + execCommand for non-secure contexts (e.g.
 * the app served over a LAN IP rather than localhost/https), where
 * `navigator.clipboard` is undefined. Returns whether the copy succeeded so
 * callers only report success when something was actually copied.
 */
export async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // Permission denied or insecure context — fall through to the legacy path.
  }
  return legacyCopy(text);
}

function legacyCopy(text: string): boolean {
  try {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.top = "-9999px";
    document.body.appendChild(textarea);
    textarea.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(textarea);
    return ok;
  } catch {
    return false;
  }
}
