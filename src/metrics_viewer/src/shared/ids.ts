import { z } from "zod";

/**
 * ClickHouse UUIDs in this project are not all RFC 4122 (migrated ids such as
 * 00000000-0000-0000-0000-000000000004 have no version/variant nibbles), so the strict
 * `z.uuid()` rejects them. Validate the shape only.
 */
export const uuidSchema = z
  .string()
  .regex(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i, "Invalid UUID");
