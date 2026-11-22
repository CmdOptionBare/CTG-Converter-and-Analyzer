/**
 * Shared LIVE-PREVIEW OAuth client identity (server-only — NEVER import from the client).
 *
 * The sandbox serves each live preview on a dynamic `https://*.grok-sandbox.com`
 * URL, which can't be pre-registered per app. The broker instead exposes ONE
 * shared "preview" client that accepts any
 * `https://*.grok-sandbox.com/api/auth/oauth2/callback/*`.
 *
 * The preview client **secret** is not stored in this repository. Set
 * `GROK_AUTH_CLIENT_SECRET` (and optionally `GROK_AUTH_CLIENT_ID`) in the host
 * environment when federated sign-in is required. Without that secret, auth
 * stays off unless a deployer injects per-app `GROK_AUTH_*` credentials.
 *
 * When deployed, the deployer injects a per-app `GROK_AUTH_*` that overrides
 * these defaults (see `server.ts`).
 */
export const PREVIEW_CLIENT_ID = "grok_preview";

/** The shared auth broker issuer (OIDC discovery lives under it). */
export const GROK_ISSUER_DEFAULT = "https://auth.grok.me";

/**
 * Host patterns whose callbacks the preview client accepts. Better Auth derives
 * the live preview's real origin from the request host and validates it against
 * this list (wildcard-matched), so the OAuth `redirect_uri` becomes the concrete
 * `https://<preview-host>/api/auth/oauth2/callback/...` the broker allows.
 */
export const PREVIEW_ALLOWED_HOSTS = ["*.grok-sandbox.com"] as const;
