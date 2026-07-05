import { SchemaForm } from "@/shared/schema-form/SchemaForm";
import { useWorkflowStore } from "../store";

export function RuntimeSettingsPopover() {
  const { schema, runtimeConfig, setRuntimeConfig } = useWorkflowStore();
  if (!schema) return null;
  return (
    <div className="absolute bottom-14 left-[64px] z-20 w-[360px] rounded-md border border-line bg-panel p-3 shadow-xl">
      <div className="mb-3 text-[13px] font-bold text-txt">Global runtime settings</div>
      <SchemaForm schema={schema.runtime_config} values={runtimeConfig} onChange={setRuntimeConfig} />
    </div>
  );
}
