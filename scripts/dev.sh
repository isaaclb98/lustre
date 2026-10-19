#!/usr/bin/env bash
#
# scripts/dev.sh — wrapper for the dev stack at ~/docker/lustre-dev/.
#
# Why this exists (rather than calling docker compose directly):
#   - The dev stack uses different container names, volumes, and
#     ports from prod. The point of that is isolation. But a stray
#     `docker compose down` in the wrong directory, or a copy-paste
#     error in a script, could still target prod. This wrapper
#     refuses to run from anywhere except ~/docker/lustre-dev/
#     and refuses to invoke docker compose against any project that
#     isn't `lustre-dev`.
#   - The agent (me) has been told "don't touch prod" many times and
#     has gotten it wrong before. Mechanical guards > discipline.
#
# Usage:
#   scripts/dev.sh up       # docker compose up -d --build
#   scripts/dev.sh down     # docker compose down (keeps volumes)
#   scripts/dev.sh reset    # docker compose down -v (wipes dev data)
#   scripts/dev.sh logs     # docker compose logs -f (both services)
#   scripts/dev.sh status   # docker compose ps
#   scripts/dev.sh shell    # docker compose exec search /bin/bash
#
# Anything else: prints usage.

set -euo pipefail

# ----- Config ---------------------------------------------------------------

DEV_DIR="$HOME/docker/lustre-dev"
DEV_PROJECT="lustre-dev"
PROD_DIR="$HOME/docker/lustre"
PROD_CONTAINERS=("lustre" "lustre-qdrant")

# On this host, the user's effective groups don't include `docker`
# even though /etc/group lists them — so plain `docker` fails with
# "permission denied ... /var/run/docker.sock". Wrap every docker
# invocation in `sg docker -c ...`. If the host IS configured so
# the user can run docker directly (no sg needed), this still
# works — `sg -c "cmd"` is a no-op wrapper when the user already
# has the right groups.
docker() {
  # $@ contains the args AFTER `docker` (the function name itself
  # is consumed by the shell as the command). Re-prepend `docker`
  # so `sg` invokes the real binary.
  local args
  args="$(printf '%q ' "$@")"
  args="${args% }"  # strip trailing space
  sg docker -c "PATH=$PATH docker $args"
}

# ----- Preflight ------------------------------------------------------------

err() { printf '[dev] %s\n' "$*" >&2; }
ok()  { printf '[dev] %s\n' "$*"; }

# Always-visible prod reminder, never blocks. The whole point of
# this script is to keep prod untouched; printing prod's status
# reinforces that awareness every time.
preflight() {
  # Where am I? If this script got invoked from the prod directory
  # somehow (symlink, relative path, accidental copy), refuse.
  local here
  here="$(pwd -P)"
  if [[ "$here" == "$PROD_DIR" ]]; then
    err "refusing to run from prod directory: $here"
    err "this script only operates on the dev stack"
    exit 1
  fi

  if [[ ! -d "$DEV_DIR" ]]; then
    err "dev stack not found at $DEV_DIR"
    err "create ~/docker/lustre-dev/ with docker-compose.yml + .env"
    exit 1
  fi

  # Show prod containers (read-only check). If they're up, just
  # print them so the user/agent is reminded that prod exists and
  # this script is not for prod.
  for c in "${PROD_CONTAINERS[@]}"; do
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -qFx "$c"; then
      ok "prod container '$c' is running (this script will not touch it)"
    fi
  done
}

# After invoking docker compose, sanity-check that the project name
# actually resolved to lustre-dev. Belt-and-braces in case
# the user has weird env overrides or a stale compose project.
verify_dev_project() {
  local projects
  projects=$(docker compose ls --format json 2>/dev/null \
    | python3 -c "import json,sys; [print(p['Name']) for p in json.load(sys.stdin)]" 2>/dev/null || true)
  if ! printf '%s\n' "$projects" | grep -qx "$DEV_PROJECT"; then
    err "dev project '$DEV_PROJECT' not registered with docker compose"
    err "compose ls output was: $(printf '%s' "$projects")"
    exit 1
  fi
}

# ----- Commands -------------------------------------------------------------

cmd="${1:-help}"
shift || true

preflight

case "$cmd" in
  up)
    (cd "$DEV_DIR" && docker compose up -d --build)
    verify_dev_project
    ok "dev stack is up at http://localhost:18000 (Qdrant: 16333)"
    ;;
  down)
    (cd "$DEV_DIR" && docker compose down)
    ok "dev stack stopped (volumes preserved)"
    ;;
  reset)
    err "this will DELETE dev's Qdrant + SQLite + HF cache volumes"
    err "prod is untouched (separate volumes), but dev data WILL be lost"
    read -r -p "[dev] type 'reset' to confirm: " confirm
    if [[ "$confirm" != "reset" ]]; then
      err "aborted"
      exit 1
    fi
    (cd "$DEV_DIR" && docker compose down -v)
    ok "dev stack reset (volumes wiped)"
    ;;
  logs)
    (cd "$DEV_DIR" && docker compose logs -f "$@")
    ;;
  status)
    (cd "$DEV_DIR" && docker compose ps)
    ;;
  shell)
    (cd "$DEV_DIR" && docker compose exec search /bin/bash)
    ;;
  help|--help|-h|"")
    cat <<EOF
usage: scripts/dev.sh <command> [args]

commands:
  up       start dev stack (pulls image if needed)
  down     stop dev stack (preserves volumes)
  reset    stop + WIPE dev volumes (requires typed confirmation)
  logs     tail logs (pass service name to filter: search / qdrant)
  status   show container status
  shell    shell into the search container

this script only operates on the dev stack at $DEV_DIR.
prod is at $PROD_DIR and is NEVER touched by this script.
EOF
    ;;
  *)
    err "unknown command: $cmd"
    err "run 'scripts/dev.sh help' for usage"
    exit 1
    ;;
esac
