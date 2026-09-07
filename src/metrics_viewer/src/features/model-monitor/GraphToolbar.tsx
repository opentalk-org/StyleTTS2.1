import { Maximize2, Scan } from "lucide-react";

import { Checkbox, IconButton, Range, SearchInput, Toolbar, Tooltip } from "@/shared/ui";

export interface GraphToolbarProps {
  search: string;
  onSearch: (value: string) => void;
  detail: number;
  maxDetail: number;
  onDetail: (value: number) => void;
  showShapes: boolean;
  onShowShapes: (value: boolean) => void;
  parametersOnly: boolean;
  onParametersOnly: (value: boolean) => void;
  boxCount: number;
  moduleCount: number;
  onFit: () => void;
}

export function GraphToolbar(props: GraphToolbarProps) {
  const depth = Math.min(props.detail, props.maxDetail);
  return (
    <>
      <Toolbar
        start={
          <>
            <SearchInput
              label="Search modules"
              value={props.search}
              onValue={props.onSearch}
              placeholder="Search modules by name or type"
              className="w-64"
            />
            <Tooltip content="How many levels to nest inside the module on the canvas; click a module to go into it">
              <label className="flex items-center gap-2 text-xs text-fg-muted">
                Depth
                <Range
                  aria-label="Detail level"
                  min={1}
                  max={props.maxDetail}
                  step={1}
                  value={depth}
                  onValue={props.onDetail}
                  className="w-24"
                  disabled={props.maxDetail < 2}
                />
                <span className="w-8 font-mono tabular-nums">
                  {depth}/{props.maxDetail}
                </span>
              </label>
            </Tooltip>
          </>
        }
        end={
          <>
            <Tooltip content="Hide activations, dropout and traced tensor operations, wiring the layers that hold weights straight together">
              <Checkbox
                checked={props.parametersOnly}
                onChange={(event) => props.onParametersOnly(event.target.checked)}
              >
                Weights only
              </Checkbox>
            </Tooltip>
            <Checkbox checked={props.showShapes} onChange={(event) => props.onShowShapes(event.target.checked)}>
              Shapes
            </Checkbox>
            <span className="font-mono text-xs tabular-nums text-fg-muted">
              {props.boxCount} shown · {props.moduleCount} modules
            </span>
            <IconButton label="Fit to view" onClick={props.onFit}>
              <Scan size={14} />
            </IconButton>
            <IconButton label="Fullscreen" onClick={() => void document.documentElement.requestFullscreen()}>
              <Maximize2 size={14} />
            </IconButton>
          </>
        }
      />
    </>
  );
}
