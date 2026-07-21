export function SpeakerSkeleton() {
  return (
    <div className="grid grid-cols-[repeat(auto-fill,minmax(220px,1fr))] gap-3.5 p-3.5">
      {Array.from({ length: 6 }).map((_, i) => (
        <div
          key={i}
          className="h-[92px] rounded-[10px] border border-line"
          style={{
            background:
              "linear-gradient(90deg,var(--color-panel) 0px,var(--color-panel-2) 200px,var(--color-panel) 400px)",
            backgroundSize: "800px 100%",
            animation: "shimmer 1.3s infinite linear",
          }}
        />
      ))}
    </div>
  );
}
