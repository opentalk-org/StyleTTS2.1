import { showToast } from "./feedback/Toast";
import { Icon, type IconName } from "./icons";
import { Button } from "./ui/Button";

/**
 * Frame around an external tool that renders in an iframe (Aim, Ray dashboard).
 * With a `src` it embeds the live tool; without one it shows a placeholder. The
 * open-in-new-tab button is the fallback when the embedded tool blocks framing.
 */
export function EmbeddedDashboard({
  src,
  toolbarIcon,
  toolbarLabel,
  status,
  openLabel,
  title,
  description,
}: {
  src?: string;
  toolbarIcon: IconName;
  toolbarLabel: string;
  status: string;
  openLabel: string;
  title: string;
  description: string;
}) {
  const open = () => {
    if (src) {
      window.open(src, "_blank", "noopener,noreferrer");
      return;
    }
    showToast(openLabel);
  };
  return (
    <div className="flex h-full flex-col px-5 pb-5 pt-4">
      <div className="mb-3 flex items-center gap-2.5">
        <div className="flex items-center gap-2 text-[13px] text-txt-dim">
          <Icon name={toolbarIcon} size={16} />
          {toolbarLabel}
        </div>
        <div className="flex-1" />
        <span className="flex items-center gap-1.5 text-xs font-semibold text-emerald-700">
          <span className="h-[7px] w-[7px] rounded-full bg-emerald-500" />
          {status}
        </span>
        <Button variant="secondary" icon="external" onClick={open}>
          Open dashboard
        </Button>
      </div>
      <div className="relative flex flex-1 overflow-hidden rounded-xl border border-line bg-panel">
        {src ? (
          <iframe src={src} title={toolbarLabel} className="h-full w-full border-0" />
        ) : (
          <div className="flex flex-1 items-center justify-center">
            <div className="max-w-[380px] px-6 text-center">
              <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-blue-50 text-blue-600">
                <Icon name={toolbarIcon} size={28} strokeWidth={2} />
              </div>
              <div className="mb-1.5 text-base font-bold text-txt">{title}</div>
              <div className="text-[13px] leading-relaxed text-txt-mute">{description}</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
