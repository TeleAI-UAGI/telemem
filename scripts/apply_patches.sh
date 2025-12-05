#!/usr/bin/env bash
set -e

PATCH_DIR="overlay/patches"

if [ ! -d "$PATCH_DIR" ]; then
  echo "⚠️ No patch directory found ($PATCH_DIR). Skipping."
  exit 0
fi

for patch in "$PATCH_DIR"/*.patch; do
  [ -f "$patch" ] || continue
  echo "🔧 Applying $patch ..."
  git apply --whitespace=fix "$patch" || {
    echo "❌ Failed to apply $patch"
    exit 1
  }
done

