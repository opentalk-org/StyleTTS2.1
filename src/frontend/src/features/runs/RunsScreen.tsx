import { defaultAimUrl } from "@/app/backendConfig";
import { useIntegrationSettingsQuery } from "@/features/settings/query";

export function RunsScreen() {
  const integration = useIntegrationSettingsQuery();
  const aimUrl = integration.data?.aim_url || defaultAimUrl();

  return <iframe src={aimUrl} title="Aim experiment tracker" className="block h-full w-full border-0" />;
}
