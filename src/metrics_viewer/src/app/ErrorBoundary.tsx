import { AlertTriangle } from "lucide-react";
import { Component, type ErrorInfo, type ReactNode } from "react";

import { Button } from "@/shared/ui";

interface State {
  error: Error | null;
}

export class ErrorBoundary extends Component<{ children: ReactNode }, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("Metrics viewer crashed", error, info.componentStack);
  }

  render(): ReactNode {
    const { error } = this.state;
    if (error === null) return this.props.children;
    return (
      <main className="grid min-h-dvh place-items-center bg-canvas p-6">
        <div className="flex max-w-lg flex-col items-center gap-3 rounded-lg border border-line bg-surface p-6 text-center">
          <AlertTriangle size={20} className="text-failed" />
          <h1 className="text-sm font-semibold text-fg">Something broke while rendering</h1>
          <p className="font-mono text-xs leading-relaxed break-words text-fg-muted">{error.message}</p>
          <div className="mt-1 flex items-center gap-2">
            <Button onClick={() => this.setState({ error: null })}>Try again</Button>
            <Button variant="primary" onClick={() => window.location.reload()}>
              Reload
            </Button>
          </div>
        </div>
      </main>
    );
  }
}
