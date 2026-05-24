---
name: testing-helium-app
description: Test the Helium/tiffytime Zig web app locally, including auth, session ownership, traversal, CORS, WebAuthn, and optional AI/search integrations.
---

# Testing Helium app

Use this when validating runtime behavior in `maczkodavid1-tech/app`.

## Devin Secrets Needed

- `HPC_AI_API_KEY`: required to verify real LLM/HPC/OpenAI-compatible streaming.
- `EXA_API_KEY`: required to verify real Exa search integration.

If these secrets are unavailable, still test auth, session ownership, token storage, traversal, CORS, and WebAuthn validation. Mark LLM streaming and Exa runtime behavior as untested.

## Build and run locally

- Build with Zig 0.14:
  ```bash
  /home/linuxbrew/.linuxbrew/opt/zig@0.14/bin/zig build-exe main.zig -O ReleaseSafe --name helium
  ```
- If building from outside the repo, either use the emitted binary path Zig creates or pass an explicit output path, for example:
  ```bash
  /home/linuxbrew/.linuxbrew/opt/zig@0.14/bin/zig build-exe /home/ubuntu/repos/app/main.zig -O ReleaseSafe -femit-bin=/home/ubuntu/repos/app/helium
  ```
- Run against an isolated runtime directory so local auth/session JSON does not pollute the repo:
  ```bash
  mkdir -p /home/ubuntu/app-test-runtime/static /home/ubuntu/app-test-runtime/images
  cp /home/ubuntu/repos/app/index.html /home/ubuntu/app-test-runtime/index.html
  PROJECT_ROOT=/home/ubuntu/app-test-runtime HOST=127.0.0.1 PORT=5017 /home/ubuntu/repos/app/helium
  ```

## Useful fixture

To test session ownership, create `sessions_db.json` before starting the server with one session owned by the UI test email and another owned by a different email. Then register/login with the owner email and verify the other session is hidden and returns 404.

## UI checks

1. Open `http://127.0.0.1:<PORT>/` in Chrome.
2. Open the account modal. The sidebar avatar button can be offscreen on desktop; if needed, use the app’s exposed `window.openAuthModal()` from the browser console only for testing the auth modal.
3. Register a unique email with a password of at least 8 characters.
4. Verify the modal shows `Sikeres regisztráció!`.
5. Verify `sessionStorage.getItem('tiffytime_auth_token')` is present and `localStorage.getItem('tiffytime_auth_token')` is absent.
6. Reopen the account modal and verify it shows `Bejelentkezve` with the test email.
7. Send a chat message. Without `HPC_AI_API_KEY`, the expected authenticated failure is `Hiba: 503 AI client not configured`; a `401` means auth did not work.

## API security checks

Use local runtime test data and a freshly registered test token from `auth_db.json` without printing secret values unnecessarily.

Verify:

- Unauthenticated `GET /api/sessions`, `GET /api/session/:id`, `DELETE /api/session/:id`, and `POST /api/chat` return `401` with `Nincs bejelentkezve`.
- Authenticated `GET /api/sessions` includes only sessions owned by the token email.
- Cross-owner `GET /api/session/:id` and `DELETE /api/session/:id` return `404 {"detail":"Session not found"}`.
- Traversal attempts like `/static/%2e%2e/main.zig` return `400 {"detail":"Invalid path"}`.
- Traversal attempts like `/api/image/%2e%2e/auth_db.json` return `400 {"detail":"Invalid image key"}`.
- CORS preflight with `Origin: https://evil.example` echoes that exact origin, includes `Vary: Origin`, and does not return wildcard `*`.
- WebAuthn register verify should reject mismatched origin: first call `/api/auth/webauthn/register/challenge`, then call `/api/auth/webauthn/register/verify` with `clientDataJSON` set to base64url JSON containing the same challenge but `origin: "https://evil.example"`; expected response is `400 {"error":"Érvénytelen webauthn clientData"}`.

## Recording tips

For browser recordings, maximize Chrome first. On Ubuntu, `wmctrl` is useful:

```bash
sudo apt-get install -y wmctrl 2>/dev/null
wmctrl -r :ACTIVE: -b add,maximized_vert,maximized_horz
```
