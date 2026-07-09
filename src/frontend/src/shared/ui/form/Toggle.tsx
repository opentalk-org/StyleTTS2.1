import { cn } from "../cn";

export function Toggle({
  checked,
  onChange,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <button
      onClick={() => onChange(!checked)}
      className={cn(
        "relative h-[26px] w-11 flex-none rounded-full border-0 cursor-pointer transition-colors",
        checked ? "bg-blue-500" : "bg-gray-300",
      )}
    >
      <span
        className={cn(
          "absolute top-[3px] h-5 w-5 rounded-full bg-white transition-[left]",
          checked ? "left-[21px]" : "left-[3px]",
        )}
      />
    </button>
  );
}
