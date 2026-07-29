import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt

EVENT_PATTERN = re.compile(r"PROFILE_EVENT (\{.*\})")


class ProfileEvent(TypedDict):
    name: str
    path: list[str]
    depth: int
    step: int | None
    rank: int
    start_ns: int
    end_ns: int
    duration_ms: float
    start_allocated_bytes: int
    end_allocated_bytes: int


def parse_events(paths: list[Path]) -> list[ProfileEvent]:
    events: list[ProfileEvent] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            match = EVENT_PATTERN.search(line)
            if match is not None:
                events.append(json.loads(match.group(1)))
    if not events:
        raise ValueError("no PROFILE_EVENT records found")
    return events


def write_tree(
    events: list[ProfileEvent],
    output: Path,
    warmup: int,
) -> None:
    grouped: dict[tuple[str, ...], list[ProfileEvent]] = defaultdict(list)
    for event in events:
        step = event["step"]
        root = event["path"][0]
        if (
            root not in ("model_loading", "checkpoint")
            and step is not None
            and step >= warmup
        ):
            grouped[tuple(event["path"])].append(event)
    roots = sorted(path for path in grouped if len(path) == 1)
    root_total = sum(
        sum(event["duration_ms"] for event in grouped[root])
        for root in roots
    )
    lines = [
        "# Training profile",
        "",
        f"> Warmup excluded: **{warmup} training step(s) per rank**.",
        "",
        "## Top-level summary",
        "",
        "| Operation | Average | Calls | Recorded time | Share |",
        "|---|---:|---:|---:|---:|",
    ]
    for root in roots:
        records = grouped[root]
        total = sum(event["duration_ms"] for event in records)
        average = total / len(records)
        percentage = 100 * total / root_total
        lines.append(
            f"| `{root[-1]}` | {_format_duration(average)} | "
            f"{len(records)} | {_format_duration(total)} | "
            f"**{percentage:.1f}%** |"
        )
    lines.extend(
        [
            "",
            "## Nested timing tree",
            "",
            "Every percentage is a share of total post-warmup top-level "
            "recorded time.",
            "",
        ]
    )
    for root in roots:
        _append_tree(lines, root, grouped, root_total)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_tree(
    lines: list[str],
    path: tuple[str, ...],
    grouped: dict[tuple[str, ...], list[ProfileEvent]],
    root_total: float,
) -> None:
    records = grouped[path]
    if records:
        total = sum(event["duration_ms"] for event in records)
        average = total / len(records)
        percentage = 100 * total / root_total
        indent = "  " * (len(path) - 1)
        bar = "█" * max(1, min(20, round(percentage / 5)))
        lines.append(
            f"{indent}- **`{path[-1]}`** — {_format_duration(average)} avg · "
            f"**{percentage:.1f}%** · {len(records)} call(s) `{bar}`"
        )
    children = sorted(candidate for candidate in grouped
                      if len(candidate) == len(path) + 1
                      and candidate[:-1] == path)
    for child in children:
        _append_tree(lines, child, grouped, root_total)


def _format_duration(milliseconds: float) -> str:
    if milliseconds >= 1000:
        return f"{milliseconds / 1000:.2f} s"
    if milliseconds >= 1:
        return f"{milliseconds:.2f} ms"
    return f"{milliseconds * 1000:.1f} µs"


def write_vram_tree(
    events: list[ProfileEvent],
    output: Path,
    warmup: int,
) -> None:
    grouped: dict[tuple[str, ...], list[ProfileEvent]] = defaultdict(list)
    for event in events:
        step = event["step"]
        if (
            event["path"][0] not in ("model_loading", "checkpoint")
            and step is not None
            and step >= warmup
        ):
            grouped[tuple(event["path"])].append(event)
    roots = sorted(path for path in grouped if len(path) == 1)
    lines = [
        "# Training VRAM allocation profile",
        "",
        f"> Warmup excluded: **{warmup} training step(s) per rank**.",
        "",
        "Added is the net allocated-memory change from scope entry to exit. "
        "Total is allocated memory at scope exit. Intermediate peaks inside a "
        "scope appear in its nested entries.",
        "",
        "## Nested allocation tree",
        "",
        "| Operation | Avg added | Max added | Avg total | Max total | Calls |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for root in roots:
        _append_vram_tree(lines, root, grouped)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_vram_tree(
    lines: list[str],
    path: tuple[str, ...],
    grouped: dict[tuple[str, ...], list[ProfileEvent]],
) -> None:
    records = grouped[path]
    added = [
        event["end_allocated_bytes"] - event["start_allocated_bytes"]
        for event in records
    ]
    totals = [event["end_allocated_bytes"] for event in records]
    branch = r"\|---" * (len(path) - 1)
    label = f"{branch}`{path[-1]}`"
    lines.append(
        f"| {label} | {_format_bytes(sum(added) / len(added), signed=True)} | "
        f"{_format_bytes(max(added), signed=True)} | "
        f"{_format_bytes(sum(totals) / len(totals))} | "
        f"**{_format_bytes(max(totals))}** | {len(records)} |"
    )
    children = sorted(candidate for candidate in grouped
                      if len(candidate) == len(path) + 1
                      and candidate[:-1] == path)
    for child in children:
        _append_vram_tree(lines, child, grouped)


def _format_bytes(byte_count: float, signed: bool = False) -> str:
    sign = "+" if signed and byte_count > 0 else ""
    absolute = abs(byte_count)
    if absolute >= 1024**3:
        return f"{sign}{byte_count / 1024**3:.2f} GiB"
    if absolute >= 1024**2:
        return f"{sign}{byte_count / 1024**2:.1f} MiB"
    return f"{sign}{byte_count / 1024:.1f} KiB"


def write_vram_plot(events: list[ProfileEvent], output: Path) -> None:
    events = [
        event
        for event in events
        if event["path"][0] not in ("model_loading", "checkpoint")
    ]
    origin = min(event["start_ns"] for event in events)
    figure, (axis, timeline) = plt.subplots(
        2,
        1,
        figsize=(20, 9),
        height_ratios=(12, 1),
        sharex=True,
        layout="constrained",
    )
    ranks = sorted({event["rank"] for event in events})
    for rank in ranks:
        points = []
        for event in events:
            if event["rank"] != rank:
                continue
            points.append((event["start_ns"], event["start_allocated_bytes"]))
            points.append((event["end_ns"], event["end_allocated_bytes"]))
        points.sort()
        axis.plot(
            [(timestamp - origin) / 1e9 for timestamp, _ in points],
            [allocated / 1024**3 for _, allocated in points],
            marker=".",
            label=f"rank {rank}",
        )
    root_events = sorted(
        (event for event in events if event["depth"] == 0),
        key=lambda item: item["start_ns"],
    )
    root_colors = {
        "data_collection": "#8c8c8c",
        "train_step": "#4c78a8",
        "validation": "#f58518",
    }
    for event in root_events:
        start = (event["start_ns"] - origin) / 1e9
        end = (event["end_ns"] - origin) / 1e9
        color = root_colors[event["name"]]
        axis.axvline(start, color=color, alpha=0.28, linewidth=0.8)
        timeline.barh(
            0,
            end - start,
            left=start,
            height=0.65,
            color=color,
            edgecolor="none",
        )
        if event["name"] == "train_step":
            timeline.text(
                start,
                0.48,
                f"step {event['step']}",
                rotation=90,
                va="bottom",
                ha="center",
                fontsize=7,
            )
        elif event["name"] == "validation":
            timeline.text(
                (start + end) / 2,
                0,
                "validation",
                va="center",
                ha="center",
                fontsize=8,
                color="white",
            )
    axis.set_ylabel("CUDA allocated VRAM (GiB)")
    axis.set_title("Training VRAM timeline")
    axis.grid(alpha=0.2)
    axis.legend()
    timeline.set_xlabel("Elapsed time (seconds)")
    timeline.set_yticks([])
    timeline.set_ylim(-0.55, 1.35)
    timeline.spines[["left", "right", "top"]].set_visible(False)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--tree", type=Path, default=Path("profile.md"))
    parser.add_argument("--vram-tree", type=Path, default=Path("profile_vram.md"))
    parser.add_argument("--plot", type=Path, default=Path("profile_vram.png"))
    parser.add_argument("--warmup", type=int, default=1)
    arguments = parser.parse_args()
    events = parse_events(arguments.logs)
    write_tree(events, arguments.tree, arguments.warmup)
    write_vram_tree(events, arguments.vram_tree, arguments.warmup)
    write_vram_plot(events, arguments.plot)
    print(f"parsed {len(events)} events")
    print(arguments.tree)
    print(arguments.vram_tree)
    print(arguments.plot)


if __name__ == "__main__":
    main()
