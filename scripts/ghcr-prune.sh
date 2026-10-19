#!/usr/bin/env bash
# scripts/ghcr-prune.sh — one-time cleanup to reclaim GHCR storage.
#
# Why this script exists:
#   image-build.yml pushed a :latest + :sha-<7> on every successful
#   build. After ~60 successful builds the free 0.5 GB tier is full.
#   The prune step in image-build.yml keeps the 5 newest :sha-* tags
#   going forward; this script does the same retroactive cleanup.
#
# Usage:
#   1. Run this script: `bash scripts/ghcr-prune.sh`
#   2. `gh auth refresh -s delete:packages` when prompted — this
#      grants the scope to the existing gh CLI token so the script
#      can list + delete container versions.
#   3. Re-run `gh auth refresh -d delete:packages` afterwards to
#      drop the scope back to least-privilege (your token currently
#      has: gist, read:org, repo, workflow).
#
# Safe to run repeatedly; idempotent.
#
# Output is dry-run by default. Pass `--apply` to actually delete.

set -euo pipefail

# Argument parsing. Flags first, then positional. Default package is
# image-search (the old package, kept published during the migration
# to ghcr.io/isaaclb98/lustre; prune it with this script, then point
# the default at lustre once Phase C7 lands); default keep is 5.
PACKAGE="image-search"
PACKAGE_OWNER="isaaclb98"
KEEP="5"
APPLY="false"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply) APPLY="true"; shift ;;
    --keep)  KEEP="$2"; shift 2 ;;
    --package) PACKAGE="$2"; shift 2 ;;
    --owner) PACKAGE_OWNER="$2"; shift 2 ;;
    --help|-h)
      echo "Usage: $0 [--apply] [--keep N] [--package NAME] [--owner USER]"
      echo ""
      echo "  --apply    actually delete (default: dry-run)"
      echo "  --keep N   number of sha-* tags to keep (default: 5)"
      echo "  --package NAME  container package name (default: image-search)"
      echo "  --owner USER   GitHub user/org owning the package (default: isaaclb98)"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 64 ;;
  esac
done

if [[ "$APPLY" == "false" ]]; then
  echo "DRY RUN: pass --apply to actually delete."
fi

# Make sure the user has the packages scopes. If not, prompt for
# refresh; we don't auto-run `gh auth refresh` because it's a
# permission escalation.
#
# Parse from the text output of `gh auth status`. The relevant line
# looks like:
#   Token scopes: 'delete:packages', 'gist', 'read:org', ...
# We want the full single-quoted list. Approach: strip everything
# before the opening quote, then everything after the closing quote.
# The closing quote is the last `'` on the line — handled by
# matching non-`'` chars greedily from the end via `rev | cut`.
scopes=$(gh auth status 2>&1 \
  | grep "Token scopes:" \
  | head -1 \
  | sed -e "s/.*Token scopes: '//" -e "s/'$//" || echo "")

if [[ -z "$scopes" ]]; then
  echo "Could not read scopes via 'gh auth status'." >&2
  echo "Run this and re-try:" >&2
  echo "  gh auth refresh -s delete:packages -s read:packages" >&2
  exit 1
fi

if [[ "$scopes" != *"delete:packages"* || "$scopes" != *"read:packages"* ]]; then
  echo ""
  echo "Your gh token is missing delete:packages and read:packages."
  echo "Current scopes: $scopes"
  echo "Run this in another terminal to grant them:"
  echo ""
  echo "  gh auth refresh -s delete:packages -s read:packages"
  echo ""
  echo "Then re-run this script."
  exit 1
fi

# List all versions for this container package. The endpoint
# returns the most recent first; we walk it in reverse because
# deleting the N oldest is idempotent if the run is re-triggered.
#
# Note: the `image-build` workflow uses `cache-to: type=gha,mode=max`
# which writes intermediate docker layers to GHCR as **untagged**
# versions. These are cheap to delete individually (the underlying
# blobs are dedup'd and the cache is rebuilt on the next build), but
# killing them wipes the cache — so we explicitly skip untagged
# versions below.
echo "Listing versions for $PACKAGE..."
mapfile -t ALL < <(
  gh api \
    "/users/${PACKAGE_OWNER}/packages/container/${PACKAGE}/versions?per_page=100" \
    --jq '.[] | "\(.id)\t\(.updated_at)\t\(.metadata.container.tags | join(","))"'
)

echo "Total versions: ${#ALL[@]}"
echo ""
printf '%-12s  %-20s  %s\n' "ID" "UPDATED" "TAGS"
for v in "${ALL[@]}"; do
  IFS=$'\t' read -r id updated tags <<<"$v"
  printf '%-12s  %-20s  %s\n' "$id" "${updated:0:10}" "$tags"
done

# Find sha-* tags, keep the KEEP newest. API returns newest-first.
# Skip untagged versions (buildx cache layers — see comment above).
echo ""
echo "=== sha-* tags, sorted by updated_at (newest first) ==="
SHA_VERSIONS=()
for v in "${ALL[@]}"; do
  IFS=$'\t' read -r id updated tags <<<"$v"
  if [[ "$tags" == *"sha-"* ]]; then
    SHA_VERSIONS+=("$v")
  fi
done

KEEP_VERSIONS=("${SHA_VERSIONS[@]:0:$KEEP}")
DELETE_VERSIONS=("${SHA_VERSIONS[@]:$KEEP}")

echo "Will keep: ${#KEEP_VERSIONS[@]} sha-* tags (the KEEP most recent)"
for v in "${KEEP_VERSIONS[@]}"; do
  IFS=$'\t' read -r id updated tags <<<"$v"
  echo "  keep: $tags"
done

echo ""
echo "Will delete: ${#DELETE_VERSIONS[@]} sha-* tags"
echo "(Untagged versions — buildx cache layers — are kept for cache hits.)"
TO_DELETE=()
for v in "${DELETE_VERSIONS[@]}"; do
  IFS=$'\t' read -r id updated tags <<<"$v"
  echo "  delete: $tags (id=$id)"
  TO_DELETE+=("$id")
done

if [[ "$APPLY" == "false" ]]; then
  echo ""
  echo "DRY RUN — pass --apply to delete the above."
  exit 0
fi

# Confirm. Skip the interactive prompt if stdin isn't a tty
# (CI / non-interactive use) — operator passed --apply deliberately,
# that's consent enough.
echo ""
if [[ -t 0 ]]; then
  read -rp "Delete ${#TO_DELETE[@]} versions? [y/N] " confirm
  if [[ "$confirm" != "y" ]]; then
    echo "Cancelled."
    exit 0
  fi
else
  echo "Non-interactive run; --apply flag is consent. Proceeding."
fi

# Delete.
count=0
for id in "${TO_DELETE[@]}"; do
  if gh api -X DELETE "/users/${PACKAGE_OWNER}/packages/container/$PACKAGE/versions/$id"; then
    count=$((count + 1))
  else
    echo "Failed to delete version $id" >&2
  fi
done
echo ""
echo "Deleted $count of ${#TO_DELETE[@]} versions."
echo ""
echo "Optional cleanup: revoke the packages scopes you granted:"
echo "  gh auth refresh -d delete:packages -d read:packages"