import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "@/app/App";
import "@/styles/index.css";

const root = document.querySelector("#root");
if (root === null) throw new Error("Root element #root was not found");
createRoot(root).render(<StrictMode><App /></StrictMode>);
