#!/usr/bin/env bash
set -e

REPO_URL=${UPSTREAM_REPO:-"https://github.com/mem0ai/mem0.git"}
BRANCH=${UPSTREAM_REF:-"main"}
PREFIX="vendor/mem0"

echo "⬇️ Pulling latest upstream changes from $REPO_URL@$BRANCH"
git subtree pull --prefix $PREFIX $REPO_URL $BRANCH --squash

echo "♻️ Reapplying patches..."
bash scripts/apply_patches.sh

