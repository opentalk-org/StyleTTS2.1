import { defaultWandbUrl } from "@/app/backendConfig";
import { useIntegrationSettingsQuery } from "@/features/settings/query";

export function RunsScreen() {
  const integration = useIntegrationSettingsQuery();
  const wandbUrl = integration.data?.wandb_url || defaultWandbUrl();

  return <iframe src={wandbUrl} title="Weights & Biases experiment tracker" className="block h-full w-full border-0" />;
}
