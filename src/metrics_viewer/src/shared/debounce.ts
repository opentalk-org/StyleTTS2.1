import { useEffect, useRef, useState } from "react";

/**
 * Local draft of a value whose commit is expensive (every commit redraws all charts).
 * The draft updates instantly for the control; `commit` runs once the input is idle.
 */
export function useDebouncedCommit<T>(value: T, commit: (value: T) => void, delay: number): [T, (next: T) => void] {
  const [draft, setDraft] = useState(value);
  const timer = useRef<number | null>(null);
  const latest = useRef(commit);
  latest.current = commit;

  useEffect(() => {
    setDraft(value);
  }, [value]);

  useEffect(
    () => () => {
      if (timer.current !== null) window.clearTimeout(timer.current);
    },
    [],
  );

  function update(next: T) {
    setDraft(next);
    if (timer.current !== null) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => {
      timer.current = null;
      latest.current(next);
    }, delay);
  }

  return [draft, update];
}
