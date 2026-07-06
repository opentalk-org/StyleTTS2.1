import { EmbeddedDashboard } from "@/shared/EmbeddedDashboard";

export function RunsScreen() {
  return (
    <EmbeddedDashboard
      toolbarIcon="activity"
      toolbarLabel="Aim experiment tracker"
      status="Connected"
      openLabel="Opening Aim in a new tab…"
      title="Embedded run dashboard"
      description="The Aim UI for the selected job renders here in an iframe — loss curves, audio samples, and gradient norms, deep-linked to the run."
    />
  );
}
