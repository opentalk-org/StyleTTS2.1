import { useState } from "react";

import { EmptyState } from "@/shared/ui/EmptyState";
import { Badge } from "@/shared/ui/Badge";
import { Button } from "@/shared/ui/Button";
import { Input } from "@/shared/ui/Input";
import { useCreateRunnerMutation, useRunnersQuery } from "./query";

export function ClusterScreen() {
  const runners = useRunnersQuery();
  const createRunner = useCreateRunnerMutation();
  const rows = runners.data?.rows ?? [];
  const [name, setName] = useState("external-runner");
  const [hostname, setHostname] = useState("127.0.0.1");
  const [port, setPort] = useState("0");
  const submit = () => createRunner.mutate({
    name,
    hostname,
    port: Number(port),
    gpu_index: null,
    resources: {},
  });

  return (
    <div className="mx-auto flex h-full max-w-[980px] flex-col px-7 py-6">
      <div className="mb-4">
        <div className="text-lg font-bold text-txt">Runners</div>
        <div className="text-[13px] text-txt-mute">Backend-connected workflow workers and manually registered extras.</div>
      </div>
      <div className="mb-4 grid grid-cols-[1fr_1fr_120px_auto] gap-2 rounded-md border border-line bg-panel p-3">
        <Input filled value={name} onChange={(event) => setName(event.target.value)} />
        <Input filled value={hostname} onChange={(event) => setHostname(event.target.value)} />
        <Input filled value={port} onChange={(event) => setPort(event.target.value)} />
        <Button icon="plus" onClick={submit} disabled={!name || !hostname || createRunner.isPending}>Add</Button>
      </div>
      {!rows.length ? (
        <div className="flex flex-1 items-center justify-center">
          <EmptyState icon="server" title="No runners registered" description="Start the dev stack or add an external runner." />
        </div>
      ) : (
        <div className="grid gap-3">
          {rows.map((runner) => (
            <article key={runner.id} className="rounded-md border border-line bg-panel p-4">
              <div className="flex items-center gap-3">
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <div className="text-[14px] font-bold text-txt">{runner.name}</div>
                    <Badge tone={runner.online ? "emerald" : runner.stale ? "amber" : "gray"}>{runner.online ? "online" : runner.stale ? "stale" : "manual"}</Badge>
                    {runner.busy ? <Badge tone="blue">busy</Badge> : null}
                  </div>
                  <div className="text-[12px] text-txt-mute">
                    {runner.hostname}:{runner.port} {runner.process_id ? `· pid ${runner.process_id}` : ""}
                  </div>
                </div>
                <div className="text-[12px] font-semibold text-txt-dim">GPU {runner.gpu_index ?? "auto"}</div>
              </div>
              <div className="mt-3 flex flex-wrap gap-2">
                {Object.entries(runner.resources).map(([key, value]) => (
                  <span key={key} className="rounded-md bg-panel-2 px-2 py-1 text-[11px] font-semibold text-txt-dim">{key}: {value}</span>
                ))}
              </div>
              <div className="mt-3 grid gap-1 text-[12px] text-txt-mute">
                <div>Active runs: {runner.active_run_ids.length ? runner.active_run_ids.join(", ") : "none"}</div>
                <div>Last heartbeat: {runner.last_seen_at ? new Date(runner.last_seen_at).toLocaleTimeString() : "not seen"}</div>
              </div>
            </article>
          ))}
        </div>
      )}
    </div>
  );
}
