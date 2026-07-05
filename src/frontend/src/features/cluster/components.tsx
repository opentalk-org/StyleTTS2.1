import { EmbeddedDashboard } from "../../shared/EmbeddedDashboard";

export function ClusterScreen() {
  return (
    <EmbeddedDashboard
      toolbarIcon="server"
      toolbarLabel="Ray cluster dashboard"
      status="5 nodes · healthy"
      openLabel="Opening Ray dashboard in a new tab…"
      title="Embedded Ray dashboard"
      description="The Ray dashboard renders here in an iframe — cluster utilization, node CPU/GPU/memory, the actor and task timeline, and per-worker logs for every running job."
    />
  );
}
