# Security notes

This is a development project; the threat model is "don't accidentally
expose admin endpoints to the open internet, and don't ship API keys in
git." The notes below are advisory.

## Admin endpoints

`/admin/reset`, `/admin/rebuild-stats`, and `/admin/status` are gated by
the `CONNECT4_ADMIN_TOKEN` env var:

- **Unset (default):** routes are open. Fine for single-user local dev.
- **Set:** every request must include `X-Admin-Token: <value>`. The
  frontend prompts for the value the first time it hits a 401 and caches
  it in `localStorage` under `admin_token`. There's a "Clear admin token"
  button on the `/admin` page.

To rotate the token: change `CONNECT4_ADMIN_TOKEN` in the backend env,
restart the backend, click "Clear admin token" in the UI (or
`localStorage.removeItem('admin_token')` from devtools).

Set the token in any deployment that reaches a non-trivial network.

## API keys

LLM provider keys live in `backend/.env`:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `DEEPSEEK_API_KEY`
- `MISTRAL_API_KEY`

`backend/.env` is in `.gitignore` (verified — no secret has ever been
committed) but lives in plaintext on the host filesystem and is mounted
into the backend container. Treat the host as the trust boundary.

**Rotate periodically and any time:**

- A key has been pasted into a chat / paste site / shared screen.
- The host has been shared or its disk imaged.
- A maintainer leaves the project.
- You're not sure when it was last rotated.

Rotation steps:

1. Generate a new key in the provider's console.
2. Update the corresponding line in `backend/.env`.
3. `docker-compose restart backend` (or `kill -HUP` the uvicorn worker if
   running locally).
4. Revoke the old key in the provider console.

## Database

The Postgres container exposed on host `5433` uses the dev defaults
`user`/`password`. That's fine for a dev box, never fine for a public IP.
For deployment, set `POSTGRES_USER` / `POSTGRES_PASSWORD` to non-default
values via `docker-compose.yml` env or override file.

## What's NOT in scope here

- Rate limiting per IP (FastAPI has no built-in; use a reverse proxy or
  `slowapi` if needed).
- HTTPS termination — assumed to be done by whatever fronts the app
  (nginx, Caddy, Cloudflare).
- Multi-user auth on game routes. The current model assumes a trusted
  user pool; player-1/player-2 tokens are per-game session secrets but
  don't tie to identities.

If any of these become relevant, add the gate (see Tier 4 plan) or open
an issue tracking what needs to land first.
