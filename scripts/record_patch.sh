#!/usr/bin/env bash
set -e

NAME=$1
[ -z "$NAME" ] && { echo "Usage: bash scripts/record_patch.sh <patch-name>"; exit 1; }

PATCH_DIR="overlay/patches"
mkdir -p "$PATCH_DIR"

PATCH_FILE="$PATCH_DIR/${NAME}.patch"
echo "📜 Recording patch: $PATCH_FILE"

git diff vendor/mem0 > "$PATCH_FILE"
echo "- ${NAME}: $(date)" >> PATCHES.md

echo "✅ Patch recorded. Remember to revert changes in vendor/mem0 now!"

