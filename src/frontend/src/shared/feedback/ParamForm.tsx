import { useState } from "react";

import { Button } from "../ui/Button";
import { Modal } from "./Modal";
import { ParamFieldRenderer } from "./ParamField";
import type { ParamField, ParamSchema, ParamValues } from "./ParamModal";

function initialValues(fields: ParamField[]): ParamValues {
  const out: ParamValues = {};
  for (const f of fields) {
    switch (f.type) {
      case "number":
      case "select":
      case "radio":
      case "toggle":
        out[f.key] = f.default;
        break;
      case "text":
        out[f.key] = f.default ?? "";
        break;
      case "drop":
        if (f.key) out[f.key] = [];
        break;
    }
  }
  return out;
}

/** Stateful body of the parameter modal: holds field values and submits them. */
export function ParamForm({
  schema,
  onClose,
}: {
  schema: ParamSchema;
  onClose: () => void;
}) {
  const [values, setValues] = useState<ParamValues>(() => initialValues(schema.fields));
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
          <ParamFieldRenderer key={f.key ?? `f${i}`} field={f} values={values} set={set} />
        ))}
      </div>
    </Modal>
  );
}
