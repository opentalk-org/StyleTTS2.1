# StyleTTS Studio — frontend

Visual workbench for fine-tuning and auditioning StyleTTS2 voice models. This is
a **UI scaffold**: every screen renders from **mock data** and **mock actions**;
no real backend calls yet.

## Stack

- **Vite + React + TypeScript**
- **Tailwind CSS v4** — config-less. The Blok color palette and design tokens
  live in an `@theme` block in `src/index.css` (there is no `tailwind.config.js`),
  exposed as normal utilities: surfaces `bg-panel / bg-panel-2 / bg-app`, lines
  `border-line / border-line-2`, text `text-txt / text-txt-dim / text-txt-mute`,
  and the `blue / emerald / amber / red / gray` scales. Font: Outfit (`font-sans`).
- **TanStack Query** — server-state/cache seam (mock for now).
- **Zustand** — local UI/client state and the mock "backends".
- **@tanstack/react-virtual** — windowed tables (see below).
- **lucide-react** — icons (wrapped by `shared/icons.tsx`).

## Structure

```
src/
  app/        App frame: shell, sidebar, header, connect screen, screen router, nav store
  shared/     Reusable, domain-agnostic building blocks
    ui/         primitives (Button, Input, Select, Badge, Card, Tabs, …)
    ui/form/    form controls (Field, NumberInput, Toggle, RadioGroup, Slider)
    feedback/   Modal, ConfirmDialog, Toast, ProgressBar, ParamModal
    data/       VirtualTable, Pager
    media/      WaveformBars, WaveformPlayer
    icons.tsx, format.ts, EmbeddedDashboard.tsx
  features/   One folder per feature (datasets, voices, audio, statistics,
              training, testing, checkpoints, jobs, runs, cluster, settings,
              workflows). Each keeps its own store.ts / logic.ts / actions.ts and
              its components as one-file-per-component.
```

### Conventions

- **One component per file.** The file name matches the component
  (`DatasetCard.tsx` exports `DatasetCard`). Non-component code goes in
  `store.ts` (Zustand), `logic.ts` (pure helpers), or `actions.ts` (mock actions).
- **Imports:** use **relative imports for nearby files** (same feature folder or
  the same shared sub-tree) and **absolute imports (`@/…`) for app-level / shared
  code** — anything under `@/shared`, `@/app`, or another
  `@/features/*`. The `@` alias maps to `src/` (configured in `tsconfig.json` and
  `vite.config.ts`).
- Separate **logic from components**: rendering in `*.tsx`, state/derivation in
  `store.ts` / `logic.ts`.
- Files stay under 300 lines; folders under 16 files.

## Scale & virtualization

Large dataset, audio, and job lists are expected to reach millions of rows.
Render them through `shared/data/VirtualTable`, which keeps only the visible rows
in the DOM. Never map a multi-million-row array into JSX — always window it.
Toolbar filtering/sorting over that scale is a server concern.

## Mock data & actions

- Temporary scaffold data belongs inside the feature that consumes it and behind
  that feature's `api.ts` / `query.ts` seam.
- Actions are mocked through shared helpers: `showToast`, `askConfirm`,
  `openParamModal` (the one reusable parameter-form/modal), and
  `useJobs().startJob(...)` which enqueues a job that animates to completion.

## Run

```bash
npm install
npm run dev      # vite dev server
npm run build    # tsc --noEmit && vite build
```

## Not yet built

- **Workflows** and **Jobs** tabs — empty placeholders (later passes); only the
  nav entries are wired.
- **Runs** / **Cluster** — framed iframe placeholders (Trackio / Ray dashboards).
- Real charts data and backend wiring are stubbed. The segment-editor timeline
  is windowed (minimap navigation), lays out overlapping segments in stacked
  lanes, and supports drag-to-move and edge-resize of segment start/end.
