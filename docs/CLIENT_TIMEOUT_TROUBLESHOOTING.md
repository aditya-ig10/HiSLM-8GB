# Client Timeout Troubleshooting (Jetson Orin NX -> AGX Orin FastAPI)

## Summary

The Jetson client reaches the AGX server, but the request does not receive a response body within the configured timeout window.

Observed client error:

```text
curl: (28) Operation timed out after 90001 milliseconds with 0 bytes received
```

This indicates a **read timeout** after successful TCP connection, not a general network outage.

## Environment

- Client device: Jetson Orin NX (8GB)
- Server device: AGX Orin
- API endpoint: `http://172.16.6.21:8000/query`
- Client script: `client.py`

## What Was Verified

### 1) Network path is healthy

From Jetson client side:

```bash
ping -c 2 172.16.6.21
```

Result: packets received, low latency.

### 2) Port is reachable

From Jetson client side:

```bash
nc -vz 172.16.6.21 8000
```

Result: connection to TCP port 8000 succeeded.

### 3) HTTP POST connects but no response bytes are returned in time

From Jetson client side:

```bash
curl -v --connect-timeout 3 --max-time 12 \
  -X POST http://172.16.6.21:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Write one short sentence about Jetson Orin.","sensor":{"temp_c":44.8}}'
```

Observed behavior:

- TCP connect succeeds.
- Request headers/body are sent.
- No response payload is received before timeout.
- Curl exits with code 28.

## Interpretation

The issue is in the request-processing/response path on the AGX server side, not in LAN routing.

Possible server-side reasons:

1. Inference path is too slow for current client timeout.
2. Request handler blocks and never reaches return statement.
3. Exception path logs request receipt but does not send response.
4. A downstream dependency (model load, GPU init, file IO, another service) is stalling.

## Why AGX Logs Can Look "OK" While Client Times Out

Request-arrival logs only confirm that the server accepted the request. They do **not** confirm that a response was completed and flushed to the client.

For confirmation, check AGX access logs for request completion with:

- status code (200/4xx/5xx)
- response time
- bytes sent

If completion logs are missing or appear much later than client timeout, the client timeout is expected.

## Recommended Server-Side Checks (AGX)

1. Test endpoint locally on AGX:

```bash
curl -v --max-time 120 -X POST http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"test","sensor":{"temp_c":44.8}}'
```

2. Add timing logs in endpoint:

- timestamp before inference call
- timestamp after inference call
- timestamp immediately before return

3. Ensure model is loaded once at startup, not per request.

4. Ensure every code path returns a response object, including exception paths.

5. If using async FastAPI route with blocking inference, move blocking work off the event loop (for example via thread offload) so the server stays responsive.

## Recommended Client-Side Controls (Jetson)

1. Keep retries with exponential backoff (already implemented in `client.py`).
2. Increase timeout during diagnostics:

```bash
python client.py --agx-ip 172.16.6.21 --timeout 180 --attempts 3
```

3. Separate connect timeout and read timeout in future client updates for clearer error reporting.

## Quick Decision Tree

1. Ping fails -> fix L3 network first.
2. Ping ok, port closed -> fix AGX bind/firewall/service startup.
3. Port open, POST times out with 0 bytes -> server handler/inference path is blocking or too slow.
4. Local curl on AGX also times out -> server issue is independent of LAN.
5. Local curl on AGX succeeds quickly but remote still times out -> inspect AGX firewall, reverse proxy, or interface binding.

## Current Diagnosis for This Case

Based on current tests, this is a **server response-time/handler completion issue**:

- Connectivity is established.
- Requests are accepted.
- Response is not returned within 90 seconds.

## Next Action Plan

1. Run local AGX curl test and measure completion time.
2. Add endpoint start/end timing logs and verify completion.
3. Optimize or offload blocking inference path.
4. Temporarily set client timeout above measured AGX response duration.
5. After fix, re-test from Jetson using both curl and `client.py`.