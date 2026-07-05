import { useState } from "react";
import { create } from "zustand";

import { Icon, type IconName } from "../icons";
import { Button } from "../ui/Button";
import { Input } from "../ui/Input";
import { Select, type Option } from "../ui/Select";
import { NumberInput, RadioGroup, Toggle } from "../ui/controls";
import { Field } from "../ui/Field";
import { Modal } from "./Modal";

export type ParamValues = Record<string, string | number | boolean>;

export type ParamField =
  | { key: string; type: "number"; label: string; default: number; min?: number; max?: number; step?: number; hint?: string; showIf?: (v: ParamValues) => boolean }
  | { key: string; type: "text"; label: string; default?: string; placeholder?: string; hint?: string; showIf?: (v: ParamValues) => boolean }
  | { key: string; type: "select" | "radio"; label: string; default: string; options: Option[]; hint?: string; showIf?: (v: ParamValues) => boolean }
  | { key: string; type: "toggle"; label: string; default: boolean; hint?: string; showIf?: (v: ParamValues) => boolean }
  | { key?: string; type: "info" | "drop"; label: string; hint?: string; showIf?: (v: ParamValues) => boolean };

export type ParamSchema = {
  icon?: IconName;
  title: string;
  desc?: string;
  danger?: boolean;
  submitLabel?: string;
  fields: ParamField[];
  onSubmit: (values: ParamValues) => void;
};

type ParamStore = {
  schema: ParamSchema | null;
  open: (schema: ParamSchema) => void;
  close: () => void;
};

export const useParamModal = create<ParamStore>((set) => ({
  schema: null,
  open: (schema) => set({ schema }),
  close: () => set({ schema: null }),
}));

/** Imperative helper to open the shared parameter modal from a mock action. */
export function openParamModal(schema: ParamSchema) {
  useParamModal.getState().open(schema);
}

function initialValues(fields: ParamField[]): ParamValues {
  const out: ParamValues = {};
  for (const f of fields) {
    if (f.type === "info" || f.type === "drop") continue;
    out[f.key] = f.default ?? (f.type === "text" ? "" : f.type === "toggle" ? false : 0);
  }
  return out;
}

export function ParamModalHost() {
  const { schema, close } = useParamModal();
  if (!schema) return <ParamModalInner key="none" schema={null} onClose={close} />;
  return <ParamModalInner key={schema.title} schema={schema} onClose={close} />;
}

function ParamModalInner({
  schema,
  onClose,
}: {
  schema: ParamSchema | null;
  onClose: () => void;
}) {
  const [values, setValues] = useState<ParamValues>(() =>
    schema ? initialValues(schema.fields) : {},
  );
  if (!schema) return null;
  const set = (key: string, value: ParamValues[string]) =>
    setValues((v) => ({ ...v, [key]: value }));
  const visible = schema.fields.filter((f) => !f.showIf || f.showIf(values));

  return (
    <Modal
      icon={schema.icon ?? "sliders"}
      title={schema.title}
      desc={schema.desc}
      danger={schema.danger}
      onClose={onClose}
      footer={
        <>
          <div className="flex-1" />
          <Button variant="secondary" onClick={onClose}>
            Cancel
          </Button>
          <Button
            variant={schema.danger ? "danger" : "primary"}
            icon={schema.danger ? "trash" : "bolt"}
            onClick={() => {
              schema.onSubmit(values);
              onClose();
            }}
          >
            {schema.submitLabel ?? "Confirm"}
          </Button>
        </>
      }
    >
      <div className="grid grid-cols-2 gap-3.5">
        {visible.map((f, i) => (
          <FieldRenderer key={f.key ?? `f${i}`} field={f} values={values} set={set} />
        ))}
      </div>
    </Modal>
  );
}

function FieldRenderer({
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

  const full = field.type === "radio" || field.type === "select" || field.type === "text";
  let control;
  if (field.type === "number")
    control = (
      <NumberInput
        value={Number(values[field.key])}
        onChange={(v) => set(field.key, v)}
        min={field.min}
        max={field.max}
        step={field.step}
      />
    );
  else if (field.type === "select")
    control = (
      <Select
        value={String(values[field.key])}
        options={field.options}
        onChange={(v) => set(field.key, v)}
      />
    );
  else if (field.type === "radio")
    control = (
      <RadioGroup
        value={String(values[field.key])}
        options={field.options}
        onChange={(v) => set(field.key, v)}
      />
    );
  else
    control = (
      <Input
        filled
        className="h-10"
        value={String(values[field.key] ?? "")}
        placeholder={field.placeholder}
        onChange={(e) => set(field.key, e.target.value)}
      />
    );

  return (
    <div className={full ? "col-span-2" : ""}>
      <Field label={field.label} hint={field.hint}>
        {control}
      </Field>
    </div>
  );
}
