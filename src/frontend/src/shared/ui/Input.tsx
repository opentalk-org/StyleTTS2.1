import type { InputHTMLAttributes } from "react";

import { cn } from "./cn";

/**
 * Text input. `filled` uses the muted in-form treatment (panel-2, transparent
 * border, blue focus ring); the default uses the bordered toolbar treatment.
 */
export function Input({
  filled = false,
  className,
  ...rest
}: { filled?: boolean } & InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={cn(
        "h-9 w-full rounded-md px-3 text-[13px] text-txt outline-none transition-colors",
        filled
          ? "bg-panel-2 border-2 border-transparent focus:border-blue-500"
          : "bg-panel border border-line focus:border-blue-500",
        className,
      )}
      {...rest}
    />
  );
}
