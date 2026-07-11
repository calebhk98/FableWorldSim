/**
 * Thin fetch wrapper: JSON in, JSON out, one error type. Every REST call in
 * this client goes through here so the "talk only to the real API" rule is
 * enforced in one place — no module reaches for a bare `fetch`.
 */

import { API_BASE_URL } from "../config";

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly path: string,
    public readonly detail: unknown,
  ) {
    super(`API ${status} on ${path}: ${JSON.stringify(detail)}`);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: { "content-type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => response.statusText);
    throw new ApiError(response.status, path, detail);
  }
  // 204s and the like never occur in this API, but stay defensive.
  const text = await response.text();
  return (text ? JSON.parse(text) : undefined) as T;
}

export function getJson<T>(path: string): Promise<T> {
  return request<T>(path, { method: "GET" });
}

export function postJson<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, { method: "POST", body: JSON.stringify(body ?? {}) });
}

export function putJson<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, { method: "PUT", body: JSON.stringify(body) });
}
