import { create } from "zustand";

import type { IconName } from "../icons";
import type { Option } from "../ui/Select";
import { ParamForm } from "./ParamForm";

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

/** Mounts the active parameter modal, remounting per schema to reset field state. */
export function ParamModalHost() {
  const { schema, close } = useParamModal();
  if (!schema) return null;
  return <ParamForm key={schema.title} schema={schema} onClose={close} />;
}
