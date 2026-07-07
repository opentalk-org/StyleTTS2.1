import { useState } from "react";

import { IconButton } from "@/shared/ui/IconButton";
import { useWorkflowStore } from "../store";
import { NodePickerPopover } from "./NodePickerPopover";
import { RuntimeSettingsPopover } from "./RuntimeSettingsPopover";
import { WorkflowLibraryPopover } from "./WorkflowLibraryPopover";

type Popup = "library" | "nodes" | "settings" | null;

export function WorkflowBottomBar() {
  const [popup, setPopup] = useState<Popup>(null);
  const { addPanel, autoLayout, graph, viewport } = useWorkflowStore();
  const toggle = (next: Popup) => setPopup((current) => (current === next ? null : next));
  return (
    <div className="absolute bottom-4 left-4 z-10 flex gap-2 rounded-md border border-line bg-panel p-2 shadow-lg">
      <IconButton icon="folder-open" title="Load or save workflow" onClick={() => toggle("library")} />
      <IconButton icon="plus" title="Add node" onClick={() => toggle("nodes")} />
      <IconButton icon="sort" title="Auto align workflow" disabled={graph.nodes.length === 0} onClick={autoLayout} />
      <IconButton icon="sliders" title="Add control panel" onClick={() => addPanel((120 - viewport.x) / viewport.zoom, (120 - viewport.y) / viewport.zoom)} />
      <IconButton icon="settings" title="Runtime settings" onClick={() => toggle("settings")} />
      {popup === "library" ? <WorkflowLibraryPopover onClose={() => setPopup(null)} /> : null}
      {popup === "nodes" ? <NodePickerPopover onClose={() => setPopup(null)} /> : null}
      {popup === "settings" ? <RuntimeSettingsPopover /> : null}
    </div>
  );
}
