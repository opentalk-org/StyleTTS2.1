import { defaultMlflowUrl } from "@/app/backendConfig";
import { useIntegrationSettingsQuery } from "@/features/settings/query";

export function RunsScreen() {
  const integration = useIntegrationSettingsQuery();
  const mlflowUrl = integration.data?.mlflow_url || defaultMlflowUrl();

  return <iframe src={mlflowUrl} title="MLflow experiment tracker" className="block h-full w-full border-0" />;
}
