/**
 * Single source for backend API base URL.
 * Used by Next.js API routes (server-side) and by client code when calling backend.
 * Set NEXT_PUBLIC_API_URL in .env.local (e.g. http://localhost:8000) so frontend and API routes use the same URL.
 */
export function getBackendUrl(): string {
  const url =
    (typeof process !== 'undefined' && process.env.NEXT_PUBLIC_API_URL) ||
    (typeof process !== 'undefined' && process.env.BACKEND_URL) ||
    'http://localhost:8000';
  return url.replace(/\/$/, '');
}
