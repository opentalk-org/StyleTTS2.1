import { useState } from "react";

import { IconButton } from "@/shared/ui/IconButton";
import { NodePickerPopover } from "./NodePickerPopover";
import { RuntimeSettingsPopover } from "./RuntimeSettingsPopover";

type Popup = "nodes" | "settings" | null;

export function WorkflowBottomBar() {
  const [popup, setPopup] = useState<Popup>(null);
  const toggle = (next: Popup) => setPopup((current) => (current === next ? null : next));
  return (
    <div className="absolute bottom-4 left-4 z-10 flex gap-2 rounded-md border border-line bg-panel p-2 shadow-lg">
      <IconButton icon="plus" title="Add node" onClick={() => toggle("nodes")} />
      <IconButton icon="settings" title="Runtime settings" onClick={() => toggle("settings")} />
      {popup === "nodes" ? <NodePickerPopover onClose={() => setPopup(null)} /> : null}
      {popup === "settings" ? <RuntimeSettingsPopover /> : null}
    </div>
  );
}
