#!/usr/bin/env bash
# Smoke test for a running heart-api (container, Kubernetes service or local uvicorn).
# Usage: scripts/smoke_test.sh [base_url]      default: http://localhost:8000
# Exits non-zero on the first failed check, so CI fails loudly.
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE="$HERE/sample_input.json"

pass() { echo "PASS  $1"; }
fail() { echo "FAIL  $1" >&2; exit 1; }

echo "Target: $BASE_URL"
echo "Waiting for the service to become ready (max 60s)..."
ready=0
for _ in $(seq 1 60); do
  if curl -fsS "$BASE_URL/ready" >/dev/null 2>&1; then ready=1; break; fi
  sleep 1
done
[ "$ready" = 1 ] || fail "GET /ready did not return 200 within 60s"
pass "GET /ready"

body="$(curl -fsS "$BASE_URL/health")" || fail "GET /health"
echo "$body" | grep -q '"status":"ok"' || fail "GET /health body unexpected: $body"
pass "GET /health  -> $body"

resp="$(curl -fsS -X POST "$BASE_URL/predict" -H 'Content-Type: application/json' -d @"$SAMPLE")" \
  || fail "POST /predict returned an HTTP error"
echo "$resp" | grep -q '"prediction"' || fail "POST /predict missing 'prediction': $resp"
echo "$resp" | grep -q '"confidence"' || fail "POST /predict missing 'confidence': $resp"
pass "POST /predict -> $resp"

code="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE_URL/predict" \
  -H 'Content-Type: application/json' -d '{"age": 500}')"
[ "$code" = "422" ] || fail "invalid payload should return 422, got $code"
pass "POST /predict with invalid payload -> 422"

curl -fsS "$BASE_URL/metrics" | grep -q '^http_requests_total' || fail "GET /metrics has no http_requests_total"
pass "GET /metrics exposes Prometheus metrics"

echo "All smoke tests passed."
