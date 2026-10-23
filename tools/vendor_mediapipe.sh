#!/usr/bin/env bash
# Vendors MediaPipe Tasks Vision (JS + SIMD wasm) and a model into an app folder,
# so camera apps run from this site with no CDN. Reuse for any face/hand/pose app.
#   tools/vendor_mediapipe.sh chehra face_landmarker/face_landmarker/float16/1/face_landmarker.task
set -euo pipefail
APP=${1:?app folder}; MODEL=${2:?model path under mediapipe-models}; VER=${MP_VERSION:-0.10.14}
ROOT=$(cd "$(dirname "$0")/.." && pwd); OUT="$ROOT/$APP/vendor"; TMP=$(mktemp -d)
mkdir -p "$OUT/wasm"
(cd "$TMP" && npm pack -s "@mediapipe/tasks-vision@$VER" >/dev/null && tar xzf ./*.tgz)
cp "$TMP/package/vision_bundle.mjs" "$OUT/"
cp "$TMP/package/wasm/vision_wasm_internal."{js,wasm} "$OUT/wasm/"
curl -sSfL -o "$OUT/$(basename "$MODEL")" "https://storage.googleapis.com/mediapipe-models/$MODEL"
rm -rf "$TMP"; du -sh "$OUT"
