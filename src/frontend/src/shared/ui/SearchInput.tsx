import { Icon } from "../icons";
import { Input } from "./Input";
import { cn } from "./cn";

export function SearchInput({
  value,
  onChange,
  placeholder,
  className,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  className?: string;
}) {
  return (
    <div className={cn("relative min-w-[200px] max-w-[320px] flex-1", className)}>
      <span className="absolute left-3 top-1/2 -translate-y-1/2 text-txt-mute">
        <Icon name="search" size={16} />
      </span>
      <Input
        className="pl-9"
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
      />
    </div>
  );
}
