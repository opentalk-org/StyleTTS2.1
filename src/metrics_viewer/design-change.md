# Metrics viewer — UI/UX redesign

Scope: `src/metrics_viewer` only. This document is the plan, the design system and the
layout specification. It is written against the code as of branch `metrics-viewer-backend`
and the screenshots in `.tmp/design-refs/current/` (captured headless at 1600×1000 and
1100×800). Reference material from W&B, MLflow, ClearML and Pluto lives in
`.tmp/design-refs/inspiration/`; it was used to sanity-check conventions, not to copy.

---

## 1. Audit of the current UI

### 1.1 Colour and contrast

| Problem | Where | Measurement / evidence |
|---|---|---|
| Near-black canvas with a full-white text colour | `styles/theme.css` `--color-base:#09090b`, `--color-fg:#f4f4f5` | 19:1 contrast. Far beyond what is needed; on OLED/dark rooms this produces halation and eye strain. Every serious tracker (Pluto, ClearML dark, W&B dark) sits on a dark *grey* (#141518-#1e2229), never #09090b. |
| Muted text fails AA and is used at 10 px | `--color-fg-muted:#71717a`, `GroupLabel` (`text-[10px] uppercase tracking-widest`) | 4.1:1 on `#09090b`; WCAG AA for small text is 4.5:1. This colour carries column headers, chart axis text, counts, captions, breadcrumbs. |
| Borders are invisible | `--color-line: rgb(255 255 255 / 0.07)` | 1.15:1 against the canvas. Cards, table header and split handle all vanish; the eye cannot find edges, which is the main reason the UI reads as "randomly joined". |
| Alpha-white surfaces | `--color-surface: rgb(255 255 255 / 0.035)` | Surfaces differ from the canvas by ~2 % luminance; hover states and selected rows are barely perceptible. |
| One accent hue does five jobs | indigo `#6366f1`/`#818cf8` | Selection, active tab, links, "running" status, *and* series 1-3 and 5 of the chart palette. A selected row, the running badge and the first three run lines all look the same. |
| Chart palette is monochrome | `shared/chart.ts` `SERIES_COLORS` = `#818cf8 #6366f1 #a5b4fc #7dd3fc #c4b5fd #94a3b8` | Four of six are indigo/violet tints. Runs cannot be told apart on a chart; this defeats the purpose of a comparison tool. |
| Hard-coded colours bypass the theme | `model-monitor/ModelMonitor.tsx` (`#1357dc`, `#d4d4d8`, `#111318`, `#b8bbc4`), `HistogramCard.tsx` (`#2563eb`, `#090a0d`, `#07080a`), `chart.ts` hover label | The model graph uses bright white-grey node borders on black (glare), a different blue than the rest of the app, and cannot follow a theme switch. |
| Body gradient | `styles/base.css` `body { background: linear-gradient(...) fixed }` | Adds banding and a visible seam behind fixed panes for no benefit. |
| No light theme | `:root { color-scheme: dark }` | Users who find dark UIs tiring have no way out. |

### 1.2 Sizing, spacing and typography

| Problem | Evidence |
|---|---|
| Four stacked bars before any content | App header `h-14` (56) → runs toolbar `h-13` (52) → selection bar `h-9` (36) → column header 32 px = 176 px of chrome on the left; header 56 → analysis nav `h-13` → query card `min-h-11` → filter row 32 on the right. |
| Heights are inconsistent | `h-14`, `h-13`, `h-12`, `h-9`, `h-8`, `h-7`, `h-6`, `min-h-11`, `min-h-14`, `min-h-10` all appear for toolbar-like rows. Card headers are `h-12` in `Surface.tsx` but overridden to `h-14` in `MetricChart.tsx`. |
| Padding is inconsistent | `px-3`, `px-3.5`, `px-4`, `px-5`, `p-2`, `p-3`, `p-4`, `px-2.5`, `py-1.5`, `py-3` across sibling components with no rule. |
| Radii are inconsistent | `rounded-md`, `rounded-lg`, `rounded-xl`, `rounded-2xl`, `rounded-[6px]`, `rounded-[4px]`, `rounded-full` used without a scale. |
| Type sizes below the legibility floor | `text-[10px]` in GroupLabel, StatusBadge, CountPill, legend chips, media captions; `text-[11px]` in SQL preview, footers. Chart tick font 10 px. |
| Uppercase-tracked mono labels everywhere | Column headers, status badges, section eyebrows, run names in artifact cards. Uppercase + `tracking-widest` + 10 px + muted colour is the least legible combination possible. |
| Root font is 14 px but body text is `text-xs`/`text-sm` | Effective UI text is 12-13 px with 14 px controls; nothing is set at the base size, so the scale is meaningless. |

### 1.3 Layout and interaction (UX)

| Problem | Where | Effect |
|---|---|---|
| **Charts below the fold are unreachable** | `Analysis.tsx`: `<div className="@container min-h-0 flex-1 p-4">` has no `overflow-auto`; the SplitPane end pane is `overflow-hidden` and `<main>` is `h-dvh overflow-hidden`. | With more than ~2 rows of charts the rest is clipped. The `sticky top-14` nav only makes sense in a scrolling container; in the screenshots the query card is already half hidden under the nav. This is a functional bug, not just polish. |
| Three counters for the same fact | Header `1 / 1 runs`, table bar `1 selected · 1 runs`, analysis nav `1 selected`. | Noise, and the eye reads the header counter as navigation. |
| Two "Clear" buttons | Header `Clear` and table bar `Clear 1`. | Duplicate primary-looking action in the top bar. |
| Back button is overloaded | `Viewer.tsx`: same arrow closes the model graph *or* leaves the project. | Users cannot predict where "back" goes. |
| Model graph replaces the whole workspace | `modelMonitorOpen` swaps the SplitPane out. | Losing the run list while inspecting a run breaks the mental model "runs on the left, detail on the right". |
| Layout toggles are three cryptic icons in the header | `PanelLeftClose / Rows2 / PanelRightClose` group. | Collapsing the runs pane hides the only way to change the selection; orientation toggle is rarely needed and costs prime header space. |
| Random default columns | `shared/metrics.ts` `defaultRunColumns` shuffles metric columns with `Math.random()`. | The table looks different on every load. Users learn nothing. |
| Selection model is checkbox-only | Every row click toggles inclusion; there is no "focus one run" affordance. | To look at a single run you must deselect everything else. Star, colour swatch, check and name are jammed into one 160 px cell. |
| Legend is per-chart and hides runs per-chart | `MetricChart.tsx` `hiddenRuns` is local state. | Hiding a run in one chart does nothing in the others; 20 charts × N chips repeats the same names 20 times. |
| Plot settings is a 1400 px modal with draft/save/cancel | `PlotSettingsDialog` | A modal with a confirm dialog on close, for two toggles and a slider. Settings apply to one chart only; there is no global smoothing or x-axis choice (W&B, ClearML, Pluto all have a global control). |
| SQL query is the first thing in the plots tab | `QueryBar` at the top of `Plots`. | Advanced feature in the most valuable position; the collapsed state shows a truncated SQL fragment as a "summary". |
| Stale copy | Empty state: "The query above resolves `selected()` against them…" | `selected()` no longer exists; the query uses `{run_ids:Array(UUID)}`. |
| Native `window.prompt` / `window.confirm` | `ViewsDialog.tsx`, `MetricChart.tsx` | Unstyled, blocks the tab, not themeable. |
| Views are stored per browser with no URL state | `store.ts` localStorage; project/run selection not in URL. | Reload loses the project and selection; links cannot be shared. |
| Inspector overlays the graph | `MonitorInspector.tsx` `absolute left-3 w-[840px]` | Covers the node you clicked; histogram cards have hard-coded chart colours and a bare `<input type=range>`. |
| Projects page has a header with only the word "Projects" | `Projects.tsx` | 56 px of chrome for a title; the table is a card floating in a gradient. |
| Responsive behaviour | 1100 px screenshot: header actions crowd the title; pane ratio persists at 0.38 regardless of width. | No breakpoint switches to stacked rows or a drawer. |

### 1.4 What already works and should be kept

- Virtualised run table with keyboard navigation, shift-range selection and sort.
- Plotly with LTTB downsampling in ClickHouse; `scattergl` switch at 350 points.
- Namespace grouping of plots and artifacts, collapsible groups.
- Media panel with a step scrubber and lightbox; audio waveform player.
- Params diff highlighting.
- Saved views concept (contents are right; the UI around it is wrong).
- Feature folder structure and the `shared/ui` primitive set.

---

## 2. Design principles for the redesign

1. **Runs left, panels right, always.** The two-pane workspace never disappears; every
   detail view (charts, compare, media, model graph) is a tab of the right pane.
2. **One accent, one job.** The accent colour marks *interactive selection* only. Status
   has its own semantic colours; runs have a categorical palette that never overlaps the
   accent.
3. **Edges come from borders, not from luminance tricks.** Every surface has a visible
   border; alpha-white overlays are removed.
4. **Progressive disclosure, not hiding.** Frequent controls (search, smoothing, x-axis,
   columns) are always visible in a single toolbar. Rare controls (SQL, per-chart
   settings, layout) are one click away behind clearly labelled buttons. Nothing lives
   behind an icon-only button without a tooltip.
5. **One number per fact.** Selection count appears once, in the runs panel footer.
6. **Deterministic defaults.** Same project → same columns, same colours, same layout.
7. **12-13 px is the floor.** No text under 11 px; captions are 11 px only for numeric
   tick labels.
8. **State in the URL.** Project, selected runs, active tab and view id are search params.

---

## 3. Design system

All tokens go in `src/styles/theme.css` (`@theme`) and are consumed as Tailwind utilities.
Light and dark are both first-class; the dark palette is the default and light is chosen
with `data-theme="light"` on `<html>` (toggle in the top bar, persisted in localStorage,
initial value from `prefers-color-scheme`).

### 3.1 Colour tokens

Values are hex for direct use; the notes give the OKLCH intent so the scale can be
extended consistently.

#### Neutral surfaces

| Token | Dark | Light | Role |
|---|---|---|---|
| `--color-canvas` | `#141518` | `#f5f5f6` | Page background. Dark is L≈0.20, a warm-neutral grey, not black. |
| `--color-surface` | `#1b1c20` | `#ffffff` | Panels, table body, cards. |
| `--color-surface-raised` | `#222328` | `#ffffff` | Popovers, dialogs, sticky toolbars. Light mode uses shadow instead of a lighter fill. |
| `--color-surface-inset` | `#101114` | `#ececee` | Inputs, code/SQL editor, chart plot area. Slightly *darker* than the surface in dark mode so inputs read as wells. |
| `--color-surface-hover` | `#26272d` | `#f0f0f2` | Row/button hover. Opaque, not alpha. |
| `--color-surface-selected` | `#1e2a3f` | `#e6eefc` | Selected row background (accent-tinted). |
| `--color-border` | `#2c2e34` | `#e1e2e6` | Hairline dividers and card edges. ≥1.5:1 on canvas. |
| `--color-border-strong` | `#5c5f67` | `#b9bcc4` | Input and checkbox outlines, split handle. ≥3:1 on canvas (WCAG non-text). |

#### Text

| Token | Dark | Light | Contrast on surface | Role |
|---|---|---|---|---|
| `--color-text` | `#e4e5e9` | `#1c1d21` | 13.5:1 / 15:1 | Primary. Deliberately not white. |
| `--color-text-secondary` | `#b2b5bc` | `#4a4d55` | 8.4:1 / 8.6:1 | Table cells, descriptions. |
| `--color-text-muted` | `#8b8e96` | `#6b6f78` | 5.4:1 / 5.1:1 | Captions, column headers, axis labels. AA at 11 px. |
| `--color-text-disabled` | `#5f626a` | `#a4a7ae` | — | Disabled controls only. |

#### Accent (interactive only)

| Token | Dark | Light | Role |
|---|---|---|---|
| `--color-accent` | `#5b9dff` | `#2266d3` | Primary buttons, active tab underline, links, focus ring, selected checkbox. |
| `--color-accent-hover` | `#7ab1ff` | `#1a55b3` | |
| `--color-accent-text` | `#ffffff` | `#ffffff` | Text on accent. |
| `--color-accent-subtle` | `#1e2a3f` | `#e6eefc` | Same as `surface-selected`; used for active segmented options. |

Blue is chosen because it is the conventional "interactive" hue and is the *only* hue
excluded from the run palette below, so a selected row can never be confused with a run
colour.

#### Status (semantic)

| Status | Dark fg / bg | Light fg / bg | Mark |
|---|---|---|---|
| running | `#5b9dff` / `#1e2a3f` | `#2266d3` / `#e6eefc` | pulsing dot |
| succeeded | `#4fc38a` / `#16302a` | `#1d7a4d` / `#e3f5ea` | ✓ |
| failed | `#f0716c` / `#3a1f1f` | `#c0392b` / `#fbe7e5` | ✕ |
| queued | `#e2b04a` / `#332a17` | `#946200` / `#fbf1d8` | ○ |
| cancelled | `#8b8e96` / `#26272d` | `#6b6f78` / `#ececee` | — |

Status badges are 12 px, sentence case, with the mark as a leading icon; never uppercase
mono.

#### Run / series palette (categorical, 12 steps)

Built at constant OKLCH lightness (~0.74 dark, ~0.55 light) and chroma (~0.15) with hues
spaced so that adjacent runs differ by hue, not by tint. Blue at 250° is intentionally
absent (reserved for the accent).

| # | Dark | Light | Hue |
|---|---|---|---|
| 1 | `#f2994a` | `#c7641a` | orange |
| 2 | `#4fc38a` | `#1f8f5a` | green |
| 3 | `#e57cc5` | `#b3428f` | magenta |
| 4 | `#59c2d8` | `#1a8ea6` | cyan |
| 5 | `#d9c04a` | `#9a8300` | yellow |
| 6 | `#a78bfa` | `#6d4fd6` | violet |
| 7 | `#f0716c` | `#c43d38` | red |
| 8 | `#7fd0a8` | `#3f9a6d` | mint |
| 9 | `#e6a1d0` | `#a85f8f` | pink |
| 10 | `#8fb8e8` | `#4d7fb3` | steel (low-chroma blue, clearly distinct from accent) |
| 11 | `#c9a27a` | `#8a6535` | tan |
| 12 | `#b5b8c0` | `#6f727a` | grey |

Run colour is assigned deterministically by *position in the run list ordered by start
time within the project*, not by position in the current selection, so a run keeps its
colour when other runs are added or removed. Overrides in `runColors` still win.

#### Chart tokens

| Token | Dark | Light |
|---|---|---|
| `--chart-grid` | `#26272d` | `#e8e9ec` |
| `--chart-axis-text` | `--color-text-muted` | same |
| `--chart-cursor` | `--color-text-secondary` at 60 % | same |
| `--chart-hover-bg` | `--color-surface-raised` | same |
| `--chart-raw-opacity` | 0.25 | 0.3 |

`shared/chart.ts` reads these from `getComputedStyle(document.documentElement)` once per
theme change instead of embedding hex.

### 3.2 Typography

| Role | Size / line | Weight | Family | Where |
|---|---|---|---|---|
| Display | 18 / 24 | 600 | sans | Page title in projects list only |
| Title | 14 / 20 | 600 | sans | Panel titles, chart titles, dialog titles |
| Body | 13 / 20 | 400 | sans | Default UI text, table cells, descriptions |
| Body strong | 13 / 20 | 500 | sans | Run names, tab labels, button labels |
| Caption | 12 / 16 | 400 | sans | Secondary rows, badges, column headers (500, sentence case) |
| Numeric | 12 / 16 | 400 | mono, `tabular-nums` | Metric values, durations, counts, step numbers |
| Code | 12.5 / 20 | 400 | mono | SQL editor, ids, artifact paths |
| Axis | 11 / 14 | 400 | mono | Chart ticks only |

Rules: root `font-size: 13px`. Sans is Inter → system UI. Mono is used only for values
that benefit from alignment (numbers, ids, SQL), never for labels or names. No
`uppercase` / `tracking-widest` anywhere except the optional column-header style, which
uses 12 px sentence case instead.

### 3.3 Spacing and sizing

4 px grid. Tailwind utilities map directly (`1` = 4 px).

| Token | px | Use |
|---|---|---|
| `space-1` | 4 | icon-to-label gap, inside badges |
| `space-2` | 8 | gap between controls in a toolbar, cell padding x |
| `space-3` | 12 | card padding, gap between cards, toolbar padding x |
| `space-4` | 16 | page/pane padding, dialog padding |
| `space-6` | 24 | section gap in panels |

Fixed heights (single source of truth in `theme.css` as `--size-*`):

| Element | px |
|---|---|
| App top bar | 44 |
| Pane toolbar (runs, charts) | 40 |
| Table column header | 28 |
| Table row | 32 (comfortable), 28 (compact toggle) |
| Control, default | 28 |
| Control, large (primary CTA, search in top bar) | 32 |
| Control, small (inside table rows, chart cards) | 24 |
| Chart card header | 36 |
| Chart plot area (grid) | 220 (1 col: 320) |
| Runs pane width | default 400, min 300, max 640, collapsed rail 44 |

Radius: `--radius-sm` 4 (checkbox, badge), `--radius` 6 (controls, inputs, chart cards),
`--radius-lg` 8 (popovers, dialogs), `--radius-full` for status dots and swatches only.
Cards inside panes do **not** use large radii; large radii on dense data cards are why the
current UI looks "floating".

Shadows: none on cards (borders do the work). `--shadow-popover` `0 4px 16px rgb(0 0 0 / .35)` dark,
`0 4px 16px rgb(0 0 0 / .12)` light. `--shadow-dialog` `0 12px 40px rgb(0 0 0 / .5)` / `.18`.

Focus: `outline: 2px solid var(--color-accent); outline-offset: 1px` on `:focus-visible`.
No box-shadow rings.

Motion: 120 ms ease-out for colour/opacity; 160 ms for popover enter; none for layout.
`prefers-reduced-motion` respected as today.

### 3.4 Component primitives (`src/shared/ui`)

Existing primitives are kept and re-skinned; new ones are added where the layout needs
them. Sizes are the three control sizes above (`sm` 24, `md` 28, `lg` 32).

| Primitive | Change |
|---|---|
| `Button` | Variants `primary`, `secondary` (bordered, surface), `ghost`. Remove `active:scale`. Icon buttons must pass `label` and render a `Tooltip`. |
| `IconButton` | Always wrapped in `Tooltip`; `active` state uses `accent-subtle` bg + `accent` icon. |
| `Badge` / `StatusBadge` | 12 px sentence case, semantic tokens, leading mark icon. Remove `CountPill`; use `Badge tone="neutral"` for counts. |
| `SegmentedControl` | Height 28, selected option is `surface-raised` with `border-strong` (neutral), text `text`. Accent is *not* used for segmented selection so it does not compete with row selection. |
| `SearchInput` | Height 28/32, `surface-inset` bg, `border-strong` border, accent border on focus. Optional `shortcut` prop rendering a `Kbd`. |
| `Checkbox` | 14 px box, `border-strong`, accent fill when checked. Indeterminate state added for "select all". |
| `Range` | Themed track/thumb using tokens; shows value in a `Numeric` label. |
| `Popover` | Always portal. `surface-raised`, `border`, `shadow-popover`, radius 8, padding 4. Adds `title` and `width` props. |
| `Modal` → `Dialog` | Max widths `sm` 420, `md` 640, `lg` 960. Header 44 px with title + close; footer for actions. Never wider than 960; the plot settings modal goes away. |
| `Tooltip` (new) | 12 px, `surface-raised`, delay 300 ms, supports `shortcut`. |
| `Tabs` (new) | Underline tabs, 28 px tall, label + optional count badge, disabled state with tooltip reason. |
| `Toolbar` (new) | 40 px flex row, `px-3`, `gap-2`, bottom border; `start`/`end` slots. |
| `Sheet` (new) | Right-docked panel (360-480 px) inside a pane, used for SQL editor and model-graph inspector. Not an overlay; it pushes content. |
| `Kbd` (new) | Keycap for shortcuts in tooltips and the help sheet. |
| `Skeleton` (new) | Loading rows/cards. |
| `SplitPane` | Keep; handle becomes 6 px with `border-strong` line, accent on hover/drag. |
| `EmptyState` | Keep; no dashed border, icon 20 px, title 13/500, description 12 muted, optional action button. |
| `Surface`/`Card` | Radius 6, `surface` bg, `border`; `CardHeader` 36 px. |

---

## 4. Layout specification

### 4.1 App shell

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ ◀ Projects / styletts2_finetune ▾            [Views ▾] [⟳ auto] [☀/☾] [?]      │ 44
├───────────────────────────┬──────────────────────────────────────────────────┤
│ Runs pane (400, resizable)│ Panels pane                                       │
│                           │ Tabs: Charts · Compare · Media · Model graph      │
└───────────────────────────┴──────────────────────────────────────────────────┘
```

- **Top bar (44 px)**: breadcrumb `Projects / <project>`; the project crumb is a menu
  listing other projects so switching does not require going "back". Right side: `Views`
  menu, auto-refresh toggle (for running runs), theme toggle, keyboard-help `?`.
  Nothing else. No counters, no clear button, no layout toggles.
- **Layout toggles** move to a small `⋯` menu on the runs pane toolbar (`Collapse runs`,
  `Stack panes`, `Reset layout`). Collapsing leaves a 44 px rail with a `▶ 3 runs` chip
  that re-opens it.
- **URL**: `/p/:projectId?runs=id1,id2&tab=charts&view=<viewId>`; the runs pane search
  and sort are *not* in the URL.
- Below 1100 px the SplitPane switches to rows automatically (runs on top, 40 % height);
  below 800 px the runs pane becomes a left drawer opened from a `Runs (3)` button in the
  top bar.

### 4.2 Projects page

Full-width page, `canvas` background, no card wrapper:

```
Projects                                                       [Search ⌘K]
────────────────────────────────────────────────────────────────────────────
Name                          Runs   Running   Last activity      Created
styletts2_finetune            71      —         3 d ago            Aug 5
beetle2_training              48     ● 12       35 d ago           Jul 28
```

- Rows 40 px, name 13/500, description on a second line only when present (row grows to
  52). Running count uses the status badge. Whole row is a link. Sorted by last activity.
- Top bar stays the same shell (breadcrumb shows `Projects`).

### 4.3 Runs pane

```
┌─────────────────────────────────────────────────────┐
│ [🔍 Search runs        /]  [Status ▾] [Columns ▾] [⋯]│ 40  toolbar
├──┬────┬────────────────┬─────────┬────────┬─────────┤
│☐ │    │ Run          ⇅ │ Status  │ Started│ loss    │ 28  header (sticky)
├──┼────┼────────────────┼─────────┼────────┼─────────┤
│☑ │ ●  │ run-name-1     │ ● Runn… │ 2 h    │ 0.4321  │ 32  row (selected)
│☐ │ ○  │ run-name-2     │ ✓ Succ… │ 3 d    │ 0.4012  │
│…                                                     │ virtualised
├─────────────────────────────────────────────────────┤
│ 3 of 71 selected · [Clear]            [Select all]  │ 32  footer (sticky)
└─────────────────────────────────────────────────────┘
```

- **Columns**: checkbox (28), colour swatch (24), name (flex, min 160), then user
  columns. Star moves *into* the swatch cell as a hover action; starred runs show a small
  ★ before the name and sort to the top as today.
- **Swatch**: click opens the colour picker (presets plus a free colour input). Selection alone
  decides which runs are drawn; there is no separate hide toggle.
- **Selection**: checkbox click toggles inclusion (shift = range). Clicking the row *name*
  focuses that run: it becomes the only selected run **unless** ⌘/Ctrl is held, which adds
  it. This gives a one-click "look at this run" path. Keyboard: `j/k` move, `space`
  toggle, `enter` focus, `a` select all filtered, `esc` clear.
- **Status column** is a 12 px badge with mark + text; the filter chip in the toolbar
  filters by status.
- **Default columns** (deterministic): `Run, Status, Started, Duration` + the first two
  summary metrics whose name contains `loss` or `val`, else the first two alphabetically.
  Column chooser is the existing `SearchOptionList` in a popover; columns can be
  reordered by drag in a later phase.
- **Cells**: numbers right-aligned mono 12 px; dates relative (`2 h`, `3 d`) with the
  absolute date in a tooltip; duration `1h 12m`.
- **Footer** is the *only* selection counter in the app.
- Loading uses 8 skeleton rows; empty states use `EmptyState` inside the body.

### 4.4 Panels pane — Charts tab

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Charts · Compare · Media · Model graph                                       │ 36 tabs
├──────────────────────────────────────────────────────────────────────────────┤
│ [🔍 Filter metrics] [X: Step ▾] [Smoothing ──●── 0.6] [⊞ 2 ▾] [SQL ▾] [⋯]   │ 40 toolbar (sticky)
├──────────────────────────────────────────────────────────────────────────────┤  scrollable ↓
│ ▾ train  (6)                                                         [pin]   │ section header 32
│ ┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐    │
│ │ train/loss   ⤢ ⚙ ⊘   │ │ train/mel_loss       │ │ …                    │    │ card header 36
│ │  ┌─chart 220px─────┐ │ │                      │ │                      │    │
│ └──────────────────────┘ └──────────────────────┘ └──────────────────────┘    │
│ ▾ val  (4)                                                                   │
│ …                                                                            │
└──────────────────────────────────────────────────────────────────────────────┘
```

- The content area is `overflow-auto`; the toolbar is `sticky top-0` *inside* it. This
  fixes the clipping bug.
- **Toolbar (global controls)**: metric filter (`/` to focus), x-axis (`Step`, `Relative
  time`, `Wall time`) — this rewrites the default SQL's `x` column and is disabled with a
  tooltip when a custom SQL is active; smoothing slider 0-0.99 applying EMA to *all*
  charts (per-chart settings can override); grid columns `1/2/3/Auto`; `SQL` button that
  opens the query `Sheet` (badge shows `edited` when dirty and a red dot on error);
  `⋯` menu with `Show hidden charts (n)`, `Collapse all`, `Expand all`, `Reset chart
  settings`.
- **Section header**: chevron, namespace, count badge, optional pin (pinned sections go
  first; stored in the view).
- **Chart card**: header 36 px with title (13/500, full metric name, truncates with
  tooltip) and hover-only actions: expand `⤢`, settings `⚙`, hide `⊘`. No "n series · n
  points" line; that goes in the expand view. Plot area 220 px, `surface-inset`
  background, grid lines `--chart-grid`, y ticks inside-left, x ticks 11 px mono. No
  legend chips under the chart; hovering shows the unified tooltip with run colour + name +
  value, and the runs pane swatches are the legend. Cursor sync across charts stays.
- **Expanded chart** (`⤢`, or click the title): a `Dialog lg` with the chart at 480 px, a
  legend list on the right (colour, run name, last / min / max with step, eye toggle that
  writes to the global hidden set), and the per-chart settings inline below the legend.
  `←`/`→` move to the previous/next chart; `esc` closes.
- **Per-chart settings** (`⚙`): a `Popover` (280 px) with `Y scale lin/log`, `X scale
  lin/log`, `Smoothing: inherit | none | EMA α | rolling n`, `Display line/points/both`,
  `Reset`. Changes apply immediately; no draft/save.
- **SQL Sheet** (right-docked, 480 px, resizable): editor (`surface-inset`, mono 12.5,
  min 8 rows), `Run ⌘⏎` primary + `Reset to default` ghost, result summary line
  (`14 plots · 12,400 points · 38 ms`) and the contract help collapsed under a
  `Reference` disclosure. Errors render inline under the editor. When open, the charts
  grid narrows; it is not an overlay.
- Empty state (no runs selected): icon, `Select runs to compare`, one sentence, and a
  `Select the latest run` button.

### 4.5 Compare tab (compare mode)

- Rows are the runs ticked in the run list; there is no second run picker.
- Columns are chosen with the `Columns` popover (parameters, final metrics and run fields);
  the default is the parameters that differ plus the most interesting metrics. `Only
  differences` hides columns that are identical across the rows. Headers sort; long cells
  clip at 25 characters with the full value in a tooltip.
- `Add plot` adds a plot card above the table. Each card picks an X (run name, or any
  parameter or metric of the selected runs) and a numeric Y, independently of the table
  columns: categorical X gives grouped bars per run, numeric X a scatter with one point per
  run in the run's colour. Any number of plots; all of it is saved in views.

### 4.6 Media tab

- Same section grouping as charts. A single **linked step scrubber** in the tab toolbar
  (`Step ◀ 1200 ▶ ──●──`) drives every media group; a group can unlink and keep its own.
- Media card: 36 px header (kind icon, name), grid of run tiles with the run swatch + name
  (12 px, sentence case), image 176 px, audio player, text block. Lightbox unchanged in
  behaviour, restyled with `Dialog lg`.

### 4.7 Model graph tab

- Enabled only when exactly one run is selected; otherwise the tab is disabled with the
  tooltip `Select exactly one run`.
- Toolbar: module search, `Fit`, `Expand all`, `Collapse all`, fullscreen.
- Canvas uses theme tokens: node = `surface` bg, `border` edge, 6 px radius, 13/500
  module type + 12 mono id in muted; selected node = `border-strong` + `accent` 2 px
  outline; edges `border-strong`; dots background `--color-border`.
- Inspector is a right `Sheet` (420 px) instead of a floating overlay: header with module
  type/id and parameter count, then histogram cards stacked. Histogram cards use the chart
  tokens, a proper `Range`, and a `Bins` segmented control instead of a cycling icon
  button.

### 4.8 Views

- `Views ▾` in the top bar opens a popover: list of saved views (name, `n runs · n
  columns`, relative time) with `Load`, inline rename on double-click, and a trash icon
  with an inline `Delete? Yes / No` confirm. Bottom row: `Save current as…` with an inline
  name input. No `window.prompt`/`confirm`.
- A view stores what it stores today plus: pinned sections, global smoothing/x-axis,
  hidden runs, active tab, layout. Loading a view updates the URL.

### 4.9 States, feedback, help

- Loading: skeletons for table rows and chart cards; the tab bar shows a thin accent
  progress line while queries are fetching (replaces the `loading` text).
- Errors: inline, in place (SQL sheet, chart card body), never a toast for query errors.
- Keyboard help: `?` opens a `Dialog sm` listing shortcuts.
- Every icon-only control has a tooltip; every disabled control has a reason.

---

## 5. Implementation plan

Each phase leaves the app working. Files stay under 300 lines; new components split into
`components/` folders where a feature grows.

### Phase 1 — Tokens and base (theme, no layout changes)
1. Rewrite `styles/theme.css` with the token tables in §3 (dark on `:root`, light under
   `:root[data-theme="light"]`); add `--size-*`, `--radius-*`, `--shadow-*`.
2. Rewrite `styles/base.css`: root 13 px, remove body gradient, outline-based focus,
   themed scrollbar and range input.
3. `shared/chart.ts`: read grid/axis/hover colours from CSS variables; replace
   `SERIES_COLORS` and `RUN_COLOR_PALETTE` with the 12-step palette; make `runColor`
   deterministic by project run order.
4. Remove hard-coded colours from `ModelMonitor.tsx` and `HistogramCard.tsx`.
5. Add the theme toggle (Zustand `ui` slice + `data-theme` on `<html>`).

### Phase 2 — Primitives
1. Re-skin `Button`, `IconButton`, `Badge`, `StatusBadge`, `SegmentedControl`,
   `SearchInput`, `Checkbox`, `Range`, `Popover`, `Modal→Dialog`, `EmptyState`,
   `Surface`, `SplitPane` to the sizes and tokens in §3.
2. Add `Tooltip`, `Tabs`, `Toolbar`, `Sheet`, `Kbd`, `Skeleton`.
3. Delete `GroupLabel` and `CountPill`; replace call sites with `Caption`/`Badge`.

### Phase 3 — Shell and runs pane
1. `Viewer.tsx`: new top bar (breadcrumb with project menu, Views, refresh, theme, help);
   move layout toggles into the runs pane `⋯` menu; collapsed rail.
2. Router: `/p/$projectId` route with `runs`, `tab`, `view` search params; store hydrates
   from and writes to the URL.
3. `RunTable.tsx` → split into `runs/components/{Toolbar,Header,Row,Footer}.tsx`:
   swatch column with global hidden set, focus-vs-toggle selection, status filter,
   deterministic default columns, footer counter, skeletons.

### Phase 4 — Charts tab
1. Fix the scroll container; sticky toolbar with global x-axis, smoothing, columns, SQL,
   `⋯` menu. Global settings live in the store; per-chart settings become overrides.
2. `MetricChart.tsx`: 36 px header with hover actions, 220 px plot, no inline legend;
   settings `Popover`; remove draft/save/cancel and `window.confirm`.
3. New `ChartDialog.tsx` (expanded chart with legend/stats table and prev/next).
4. `QueryBar.tsx` → `QuerySheet.tsx`; fix the stale `selected()` copy.

### Phase 5 — Compare, Media, Model graph
1. `ParamsPanel.tsx` → `ComparePanel.tsx` with params + summary groups, sticky column,
   `Only differences`.
2. `MediaPanel.tsx`: linked step scrubber in the tab toolbar, restyled tiles and lightbox.
3. Model graph becomes a tab; inspector becomes a `Sheet`; histogram card controls.

### Phase 6 — Views, keyboard, projects page
1. `ViewsDialog.tsx` → `ViewsMenu.tsx` popover with inline save/rename/delete.
2. Keyboard shortcuts and the `?` help dialog.
3. `Projects.tsx` full-width list, relative times with tooltips, running badge.

### Phase 7 — Verification
1. Contrast check script (throwaway, `.tmp/`) computing WCAG ratios for every text/surface
   pair in `theme.css` for both themes; all body text ≥ 4.5:1, control borders ≥ 3:1.
2. Headless screenshots at 1600×1000, 1280×800 and 1100×800 for: projects, empty
   workspace, 5 runs selected (charts), expanded chart, compare, media, model graph +
   inspector, SQL sheet open, views menu, light theme. Compare against
   `.tmp/design-refs/current/`.
3. Manual pass: 70+ runs virtualised table, 20+ charts scroll, keyboard navigation, reload
   restores URL state.

---

## 6. Open questions (defaults chosen; change if you disagree)

| Question | Default in this plan |
|---|---|
| Keep Plotly or move to a lighter charting library? | Keep Plotly; the palette/tokens changes are independent of the library. |
| Should views be stored server-side? | Out of scope; stays in localStorage but the view id is in the URL. |
| Per-chart x-axis (time vs step)? | Global only in this pass; per-chart later if needed. |
| Light theme default? | Dark remains the default; light follows `prefers-color-scheme` on first load. |

---

## 8. Implementation status

Applied on 2026-09-03 in one pass (phases 1-7). Everything in §3 and §4 is in the code with these
deliberate simplifications:

- URL state uses search params on `/` (`?project=…&runs=…&tab=…`) instead of a `/p/:id` route,
  so the generated route tree did not need a new file.
- Below 1100 px the panes stack vertically; the sub-800 px drawer was not built.
- Column reordering by drag in the runs table is not built (the column chooser is).
- `base.css` element rules live in `@layer base` so Tailwind utilities keep precedence; this is
  required, not optional — unlayered `button { color: inherit }` silently overrode every
  `text-*` class on buttons.
- Plot containers carry an explicit height because Plotly's responsive handler sizes the SVG
  from the parent box on every window resize; without it charts collapse to 0 px after a resize.

Verification: `npm run build` (tsc + vite) passes; headless screenshots for both themes are in
`.tmp/design-refs/new2/` and `.tmp/design-refs/light/`.
