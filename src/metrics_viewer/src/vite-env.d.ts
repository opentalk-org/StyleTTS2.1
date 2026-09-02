interface ImportMetaEnv {
  readonly VITE_METRICS_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

declare module "*.css";
declare module "plotly.js-basic-dist-min" {
  const Plotly: typeof import("plotly.js");
  export default Plotly;
}
