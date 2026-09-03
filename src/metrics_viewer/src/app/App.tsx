import { ErrorBoundary } from "@/app/ErrorBoundary";
import { Viewer } from "@/features/viewer/Viewer";

export function App() {
  return (
    <ErrorBoundary>
      <Viewer />
    </ErrorBoundary>
  );
}
