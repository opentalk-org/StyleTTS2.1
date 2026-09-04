import { Dialog, Kbd } from "@/shared/ui";

const SHORTCUTS: { keys: string[]; action: string }[] = [
  { keys: ["/"], action: "Focus the run search" },
  { keys: ["j", "k"], action: "Move between runs" },
  { keys: ["Space"], action: "Add or remove the focused run" },
  { keys: ["Enter"], action: "Show only the focused run (⌘/Ctrl adds it)" },
  { keys: ["Shift", "Click"], action: "Select a range of runs" },
  { keys: ["a"], action: "Select every run that matches the filter" },
  { keys: ["Esc"], action: "Clear the selection while the run list is focused" },
  { keys: ["Drag"], action: "Zoom into a region of a chart (scroll wheel zooms in the expanded chart)" },
  { keys: ["Double-click"], action: "Reset a chart's zoom" },
  { keys: ["⌘", "Enter"], action: "Run the SQL query" },
  { keys: ["←", "→"], action: "Previous / next chart or media step in a dialog" },
  { keys: ["?"], action: "Show this list" },
];

export function HelpDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  return (
    <Dialog open={open} onClose={onClose} title="Keyboard shortcuts" size="sm">
      <ul className="m-0 flex list-none flex-col gap-1 p-3">
        {SHORTCUTS.map((shortcut) => (
          <li key={shortcut.action} className="flex h-8 items-center justify-between gap-4 text-[13px] text-fg-secondary">
            <span>{shortcut.action}</span>
            <span className="flex shrink-0 items-center gap-1">
              {shortcut.keys.map((key) => (
                <Kbd key={key}>{key}</Kbd>
              ))}
            </span>
          </li>
        ))}
      </ul>
    </Dialog>
  );
}
