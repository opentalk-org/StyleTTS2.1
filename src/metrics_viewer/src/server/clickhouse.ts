import { createClient, type ClickHouseClient } from "@clickhouse/client";

let client: ClickHouseClient | undefined;

export function clickhouse() {
  client ??= createClient({
    url: process.env.CLICKHOUSE_HTTP_URL!,
    username: process.env.CLICKHOUSE_USER!,
    password: process.env.CLICKHOUSE_PASSWORD!,
  });
  return client;
}

export async function query<T>(
  sql: string,
  queryParams: Record<string, unknown> = {},
  settings: Record<string, string | number> = {},
) {
  const result = await clickhouse().query({
    query: sql,
    query_params: queryParams,
    clickhouse_settings: settings,
    format: "JSONEachRow",
  });
  return result.json<T>();
}
