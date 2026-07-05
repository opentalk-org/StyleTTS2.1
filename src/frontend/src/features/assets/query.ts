import { useQuery } from "@tanstack/react-query";

import { fetchFileAssets } from "./api";

const KEY = "file-assets";

export function useFileAssetsQuery(type?: string) {
  return useQuery({
    queryKey: [KEY, type ?? "all"],
    queryFn: () => fetchFileAssets(type),
  });
}
