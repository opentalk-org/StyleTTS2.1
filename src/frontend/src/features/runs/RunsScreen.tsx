import { defaultAimUrl } from "@/app/backendConfig";

export function RunsScreen() {
  return <iframe src={defaultAimUrl()} title="Aim experiment tracker" className="block h-full w-full border-0" />;
}
