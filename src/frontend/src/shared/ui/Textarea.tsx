import { forwardRef, type TextareaHTMLAttributes } from "react";

import { cn } from "./cn";

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement>>(
  function Textarea({ className, ...rest }, ref) {
    return (
      <textarea
        ref={ref}
        className={cn(
          "w-full min-h-20 rounded-md bg-panel-2 border-2 border-transparent px-3 py-2.5 text-sm leading-relaxed text-txt outline-none resize-y focus:border-blue-500",
          className,
        )}
        {...rest}
      />
    );
  },
);
