import { askConfirm } from "../../shared/feedback/ConfirmDialog";
import { openParamModal } from "../../shared/feedback/ParamModal";
import { showToast } from "../../shared/feedback/Toast";
import { Icon } from "../../shared/icons";
import { Badge } from "../../shared/ui/Badge";
import { Button } from "../../shared/ui/Button";
import { Card } from "../../shared/ui/Card";
import { IconButton } from "../../shared/ui/IconButton";
import { SearchInput } from "../../shared/ui/SearchInput";
import { Select } from "../../shared/ui/Select";
import { SectionTitle } from "../../shared/ui/Card";
import { useJobs } from "../jobs/store";
import type { Checkpoint } from "../../mock/types";
import { CATALOG, TYPE_TONE, groupCheckpoints } from "./logic";
import { useCheckpoints } from "./store";

const COLS = "minmax(180px,1.4fr) 116px 1fr 120px";

export function CheckpointsScreen() {
  const store = useCheckpoints();
  const startJob = useJobs((s) => s.startJob);
  const groups = groupCheckpoints(store);

  const upload = () =>
    openParamModal({
      icon: "upload",
      title: "Upload checkpoint",
      desc: "Accepted files and validation depend on the checkpoint type.",
      submitLabel: "Upload",
      fields: [
        { key: "type", type: "select", label: "Checkpoint type", default: "styletts2", options: [
          { value: "styletts2", label: "StyleTTS2" },
          { value: "asr", label: "ASR aligner" },
          { value: "f0", label: "F0 model" },
          { value: "plbert", label: "PL-BERT" },
        ] },
        { type: "drop", label: "Drop a .pth / .t7 file", hint: "Validated against the chosen type" },
        { key: "name", type: "text", label: "Name", default: "uploaded_checkpoint" },
      ],
      onSubmit: (v) => showToast(`Uploading "${String(v.name)}"…`),
    });

  return (
    <div className="mx-auto max-w-[1080px] px-7 pb-16 pt-5">
      <div className="mb-4 flex flex-wrap items-center gap-2.5">
        <SearchInput value={store.query} onChange={(v) => store.set({ query: v })} placeholder="Search checkpoints…" />
        <Select
          variant="mini"
          value={store.type}
          onChange={(v) => store.set({ type: v })}
          options={[
            { value: "all", label: "All types" },
            { value: "styletts2", label: "StyleTTS2" },
            { value: "asr", label: "ASR" },
            { value: "f0", label: "F0" },
            { value: "plbert", label: "PL-BERT" },
          ]}
        />
        <div className="flex-1" />
        <Button variant="primary" icon="upload" onClick={upload}>
          Upload checkpoint
        </Button>
      </div>

      <div className="flex flex-col gap-[18px]">
        {Object.entries(groups).map(([job, items]) => (
          <div key={job}>
            <div className="mb-2 flex items-center gap-2 text-xs font-semibold text-txt-dim">
              <Icon name="activity" size={14} strokeWidth={2} className="text-txt-mute" />
              {job === "—" ? "Imported / uploaded" : "From job"}
              {job !== "—" ? <span className="font-mono text-blue-600">{job}</span> : null}
            </div>
            <Card className="overflow-hidden">
              <div className="overflow-x-auto">
                <div className="min-w-[640px]">
                  {items.map((c) => (
                    <CheckpointRow key={c.id} checkpoint={c} />
                  ))}
                </div>
              </div>
            </Card>
          </div>
        ))}
      </div>

      <div className="mt-6">
        <SectionTitle className="mb-3">Pretrained catalog</SectionTitle>
        <div className="grid grid-cols-[repeat(auto-fill,minmax(280px,1fr))] gap-3">
          {CATALOG.map((c) => (
            <Card key={c.file} className="flex items-center gap-3 p-4">
              <div className="flex h-[38px] w-[38px] flex-none items-center justify-center rounded-[9px] bg-blue-50 text-blue-600">
                <Icon name="box" size={18} strokeWidth={2.2} />
              </div>
              <div className="min-w-0 flex-1">
                <div className="text-[13px] font-bold text-txt">{c.name}</div>
                <div className="font-mono text-[11px] text-txt-mute">{c.file} · {c.size}</div>
              </div>
              <IconButton
                icon="download"
                title="Download"
                onClick={() => { startJob("upload", `Download ${c.file}`); showToast("Download queued", c.file); }}
              />
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}

function CheckpointRow({ checkpoint: c }: { checkpoint: Checkpoint }) {
  const { rename, remove } = useCheckpoints();
  const del = () =>
    askConfirm({
      title: "Delete checkpoint?",
      desc: `Delete "${c.name}". This permanently removes the artifact.`,
      danger: true,
      label: "Delete checkpoint",
      onConfirm: () => { remove(c.id); showToast("Checkpoint deleted", undefined, "error"); },
    });
  const doRename = () => {
    const n = window.prompt("Rename checkpoint", c.name);
    if (n && n.trim()) rename(c.id, n.trim());
  };

  return (
    <div
      className="grid items-center gap-3.5 border-b border-line px-4 py-3 last:border-b-0"
      style={{ gridTemplateColumns: COLS }}
    >
      <div className="flex min-w-0 items-center gap-2.5">
        <div className="flex h-[30px] w-[30px] flex-none items-center justify-center rounded-[7px] bg-panel-2 text-txt-dim">
          <Icon name="box" size={15} strokeWidth={2} />
        </div>
        <span className="truncate font-mono text-[13px] font-semibold text-txt">{c.name}</span>
      </div>
      <div>
        <Badge tone={TYPE_TONE[c.type]}>{c.type}</Badge>
      </div>
      <div className="flex flex-wrap items-center gap-2.5 text-[11.5px] text-txt-mute">
        {c.spkMode !== "—" ? <span>spk: {c.spkMode}</span> : null}
        {c.decoder !== "—" ? <span>dec: {c.decoder}</span> : null}
        {c.symbols ? (
          <button
            onClick={() => showToast(`${c.symbols} symbols copied`)}
            className="inline-flex items-center gap-1 rounded bg-panel-2 px-1.5 py-0.5 font-mono text-[11px] text-txt-dim cursor-pointer"
          >
            <Icon name="copy" size={11} strokeWidth={2} />
            {c.symbols} sym
          </button>
        ) : null}
      </div>
      <div className="flex justify-end gap-0.5">
        <IconButton icon="edit" title="Rename" onClick={doRename} />
        <IconButton icon="download" title="Download" onClick={() => showToast(`Downloading ${c.name}.pth`)} />
        <IconButton icon="trash" danger title="Delete" onClick={del} />
      </div>
    </div>
  );
}
