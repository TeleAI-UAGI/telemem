#!/usr/bin/env bash
set -e

REPO_URL=${UPSTREAM_REPO:-"https://github.com/mem0ai/mem0.git"}
BRANCH=${UPSTREAM_REF:-"main"}
PREFIX="vendor/mem0"

echo "📦 Adding upstream $REPO_URL@$BRANCH to $PREFIX"
git subtree add --prefix $PREFIX $REPO_URL $BRANCH --squash

