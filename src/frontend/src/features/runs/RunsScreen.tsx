import { defaultAimUrl } from "@/app/backendConfig";
import { EmbeddedDashboard } from "@/shared/EmbeddedDashboard";

export function RunsScreen() {
  return (
    <EmbeddedDashboard
      src={defaultAimUrl()}
      toolbarIcon="activity"
      toolbarLabel="Aim experiment tracker"
      status="Connected"
      openLabel="Opening Aim in a new tab…"
      title="Embedded run dashboard"
      description="The Aim UI renders here in an iframe — loss curves, audio samples, and gradient norms for your training runs."
    />
  );
}
