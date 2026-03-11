#!/bin/bash
set -euo pipefail

args=()
skip_next=0
for arg in "$@"; do
    if [[ "$skip_next" -eq 1 ]]; then
        skip_next=0
        continue
    fi
    case "$arg" in
        --target=*|-target=*)
            continue
            ;;
        --target|-target)
            skip_next=1
            continue
            ;;
        *)
            args+=("$arg")
            ;;
    esac
done

exec zig cc -target aarch64-linux-musl "${args[@]}"
