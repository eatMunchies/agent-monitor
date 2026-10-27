# Where the proxy sits and how traffic is stopped

The model's judgment is a soft control; the network layout is the hard one. Even if every
model tier were fooled, the sandbox physically has one way out, and that way asks the
gateway first.

```
                        ┌──────────── network "outside" (normal bridge, has internet) ───────────┐
                        │                                                                        │
  your Mac ──127.0.0.1:8080──► gateway ──────────► ollama                    egress-proxy ──────► internet
                        │         │  ▲                                          ▲   │             │
                        └─────────┼──┼──────────────────────────────────────────┼───┼─────────────┘
                                  │  │ POST /check/connection (X-Check-Token)   │   │
                        ┌─────────┼──┼──────────────────────────────────────────┼───┼─────────────┐
                        │         ▼  └──────────────────────────────────────────┘   │             │
                        │      sandbox :9000  ── HTTP(S)_PROXY ─────────────────────┘             │
                        │                                                                         │
                        └──────────── network "sandbox_net" (internal: true, no internet) ────────┘
```

## The pieces

**`sandbox_net` is an internal Docker network.** Docker gives it no gateway to the
internet, so a container whose only network is `sandbox_net` cannot open a connection to
the outside, whatever it runs. The sandbox's only network is `sandbox_net`.

**The egress proxy straddles both networks.** It is mitmproxy (`mitmproxy/mitmproxy:12.2.3`)
running `proxy/egress_addon.py`. The sandbox's `HTTP_PROXY`/`HTTPS_PROXY` point at it, and
the proxy is the only container on `sandbox_net` that can also reach the internet.

For every request it:

1. Reads the request: host, port, method, path, the first 4 KB of the body, and the body
   size. HTTPS is decrypted first (below).
2. Removes the `X-Agent-Session` header the sandbox's `http_get` tool adds, and uses it to
   tell the gateway which run the request belongs to. The header is never forwarded.
3. Asks `POST /check/connection` on the gateway. That runs the same cascade as tool calls,
   with the run's goal and history attached.
4. On a block, returns `403 Blocked by agent monitor: <reason>`. On allow, forwards.

It also enforces three things on its own:

- **`connection_strategy=lazy`:** no upstream connection opens until the request is allowed.
  A blocked request costs not even a TCP handshake.
- **CONNECT only to ports 80 and 443.** Tunnels to ssh, databases and similar get a 403.
- **No raw TCP inside an allowed tunnel**, for example ssh spoken on port 443. Such bytes
  never passed the HTTP check, so they are dropped and the flow is killed.

**HTTPS inspection.**
- mitmproxy creates a CA on first start, in the `mitm_ca` volume.
- The sandbox mounts that volume read-only, and its entrypoint adds the CA to the trust
  bundle (`SSL_CERT_FILE`, `CURL_CA_BUNDLE`, `REQUESTS_CA_BUNDLE`).
- As a result, curl, Python and git inside the sandbox accept the proxy's certificates, and
  the proxy can read the request (path, body) the agent sends over HTTPS.
- A tool that brings its own trust store or pins certificates simply fails, which is the
  safe direction.

**DNS.**
- The sandbox's resolver is set to `127.0.0.1`, where nothing listens, so names outside the
  compose project do not resolve. That closes DNS as a side channel (data encoded in lookups).
- Compose service names (`egress-proxy`) still resolve through Docker's internal DNS.
- The proxy resolves real hostnames itself, after the check.

**The gateway is reachable from the sandbox but useless to it.**
- The gateway must be on `sandbox_net` to call the tool server, so the sandbox can reach it.
- Every gateway endpoint needs a token the sandbox does not have: `GATEWAY_TOKEN` for
  `/v1/*`, `CHECK_TOKEN` for `/check/connection`.
- Through the proxy, single-label hosts such as `gateway` and IP literals are blocked
  outright.

**Sandbox container hardening** (`docker-compose.yml`):
- read-only root filesystem (writable: only `/work` and a tmpfs `/tmp`)
- all Linux capabilities dropped and `no-new-privileges`
- an unprivileged user
- limits of 256 processes, 1 GB memory and 1 CPU
- no docker socket and no ssh

## Verified on 2026-10-16 (Mac, Docker Desktop)

| Test from inside the sandbox | Result |
|---|---|
| `curl --noproxy '*' http://1.1.1.1` (skip the proxy) | fails, no route |
| `getent hosts pastebin.com` (DNS side channel) | fails, does not resolve |
| `curl https://example.com` via proxy (allowlisted, HTTPS) | 200, TLS inspected |
| `curl https://pastebin.com/...` | 403 `denylisted domain pastebin.com` |
| `curl http://gateway:8080/health` via proxy | 403 `internal host gateway` |
| CONNECT to `github.com:22` | 403 `only ports 80 and 443` |
| ssh bytes tunneled over `github.com:443` | dropped, flow killed, no banner returned |
| `docker cp` a file into the sandbox | refused, read-only root filesystem |

Bug found during this testing: before the port check existed, killing the raw TCP flow in
mitmproxy's `tcp_start` hook did **not** stop the data. The ssh banner still came back.
The fix is the CONNECT port allowlist plus dropping bytes in `tcp_message`.

## What this does not cover

- **UDP/QUIC:** the sandbox has no route, so this is blocked, but it is also not inspected.
- **Covert channels inside allowed traffic:** data hidden in the query strings of GET
  requests to allowlisted hosts. The secret and entropy checks catch keys and tokens, not
  slow, encoded leaks. Keep the allowlist short.
- **Escapes from the container itself:** kernel bugs. For stronger isolation, run the
  sandbox under gVisor (`runtime: runsc`) on the Linux server.
