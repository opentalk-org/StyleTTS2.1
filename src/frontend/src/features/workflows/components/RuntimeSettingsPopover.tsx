import { Input } from "@/shared/ui/Input";
import { numberRecord, resourceLimitsForGraph, resourceRequirementsForGraph, runtimeConfigForGraph } from "../logic";
import { useWorkflowStore } from "../store";

export function RuntimeSettingsPopover() {
  const { schema, graph, runtimeConfig, setRuntimeConfig } = useWorkflowStore();
  if (!schema) return null;
  const requirements = resourceRequirementsForGraph(schema, graph);
  const resources = resourceLimitsForGraph(schema, graph, numberRecord(runtimeConfig["resources"], "runtime resources"));
  const rows = Object.keys(requirements).sort((left, right) => left.localeCompare(right));
  const setResource = (key: string, value: number) => {
    setRuntimeConfig(runtimeConfigForGraph(schema, graph, { ...runtimeConfig, resources: { ...resources, [key]: value } }));
  };

  return (
    <div className="absolute bottom-14 left-[64px] z-20 w-[360px] rounded-md border border-line bg-panel p-3 shadow-xl">
      <div className="mb-3 text-[13px] font-bold text-txt">Global runtime settings</div>
      <div className="grid gap-2">
        {rows.map((key) => (
          <label key={key} className="grid grid-cols-[minmax(0,1fr)_112px] items-center gap-3">
            <span className="truncate font-mono text-[12.5px] font-semibold text-txt">{key}</span>
            <Input
              filled
              aria-label={`${key} resource limit`}
              className="h-9 text-right font-mono tabular-nums"
              type="number"
              step="0.25"
              value={resources[key]}
              onChange={(event) => setResource(key, Number.isNaN(event.target.valueAsNumber) ? 0 : event.target.valueAsNumber)}
            />
          </label>
        ))}
      </div>
    </div>
  );
}
