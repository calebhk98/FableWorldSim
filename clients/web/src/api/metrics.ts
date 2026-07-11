/** GET /metrics (api/app.py `_register_observability`): server counters + telemetry. */

import { getJson } from "./http";
import type { Metrics } from "./types";

export function getMetrics(): Promise<Metrics> {
  return getJson("/metrics");
}
