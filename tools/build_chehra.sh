#!/usr/bin/env bash
# Build the obfuscated Chehra bundle that ships on the public site.
# The readable source lives in chehra/src/ (git-ignored, kept private). Put app.js, track.js
# and audio.js there, then run this. Output: chehra/app.min.js (minified + obfuscated, one file).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); SRC="$ROOT/chehra/src"; TMP="$(mktemp -d)"
[ -f "$SRC/app.js" ] || { echo "put the Chehra source in $SRC first"; exit 1; }
npx -y esbuild@0.24.2 "$SRC/app.js" --bundle --format=esm --minify --legal-comments=none \
  --external:*vision_bundle.mjs --outfile="$TMP/bundle.js"
npx -y javascript-obfuscator@4.1.1 "$TMP/bundle.js" --output "$ROOT/chehra/app.min.js" \
  --compact true --control-flow-flattening false --dead-code-injection false \
  --string-array true --string-array-encoding base64 --string-array-rotate true \
  --identifier-names-generator mangled --self-defending false
rm -rf "$TMP"; echo "built $ROOT/chehra/app.min.js"
