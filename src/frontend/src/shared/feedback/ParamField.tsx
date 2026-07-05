import type { ReactNode } from "react";

import { Icon } from "../icons";
import { Input } from "../ui/Input";
import { Select } from "../ui/Select";
import { Field } from "../ui/form/Field";
import { NumberInput } from "../ui/form/NumberInput";
import { RadioGroup } from "../ui/form/RadioGroup";
import { Toggle } from "../ui/form/Toggle";
import type { ParamField, ParamValues } from "./ParamModal";

/** Renders one field of the parameter modal based on its declared type. */
export function ParamFieldRenderer({
  field,
  values,
  set,
}: {
  field: ParamField;
  values: ParamValues;
  set: (key: string, value: ParamValues[string]) => void;
}) {
  if (field.type === "info")
    return (
      <div className="col-span-2 flex items-start gap-2.5 rounded-md bg-blue-50 px-3 py-2.5">
        <span className="mt-px text-blue-600">
          <Icon name="alert" size={15} strokeWidth={2} />
        </span>
        <span className="text-xs font-medium leading-normal text-blue-700">{field.label}</span>
      </div>
    );
  if (field.type === "drop")
    return (
      <div className="col-span-2 flex flex-col items-center justify-center gap-2 rounded-[9px] border-2 border-dashed border-line-2 bg-panel-2 p-6 text-center hover:border-blue-500">
        <span className="text-txt-mute">
          <Icon name="upload" size={22} strokeWidth={2} />
        </span>
        <div className="text-[13px] font-semibold text-txt">{field.label}</div>
        {field.hint ? <div className="text-[11.5px] text-txt-mute">{field.hint}</div> : null}
      </div>
    );
  if (field.type === "toggle")
    return (
      <label className="col-span-2 flex cursor-pointer items-center gap-3 py-1">
        <span className="flex-1">
          <div className="text-[13px] font-semibold text-txt">{field.label}</div>
          {field.hint ? <div className="text-[11.5px] text-txt-mute">{field.hint}</div> : null}
        </span>
        <Toggle checked={Boolean(values[field.key])} onChange={(v) => set(field.key, v)} />
      </label>
    );

  const full = field.type !== "number";
  let control: ReactNode;
  switch (field.type) {
    case "number":
      control = (
        <NumberInput
          value={Number(values[field.key])}
          onChange={(v) => set(field.key, v)}
          min={field.min}
          max={field.max}
          step={field.step}
        />
      );
      break;
    case "select":
      control = (
        <Select value={String(values[field.key])} options={field.options} onChange={(v) => set(field.key, v)} />
      );
      break;
    case "radio":
      control = (
        <RadioGroup value={String(values[field.key])} options={field.options} onChange={(v) => set(field.key, v)} />
      );
      break;
    case "text":
      control = (
        <Input
          filled
          className="h-10"
          value={String(values[field.key] ?? "")}
          placeholder={field.placeholder}
          onChange={(e) => set(field.key, e.target.value)}
        />
      );
      break;
  }

  return (
    <div className={full ? "col-span-2" : ""}>
      <Field label={field.label} hint={field.hint}>
        {control}
      </Field>
    </div>
  );
}
