#!/usr/bin/env bash
# Source from bash/zsh to use this project's isolated Rust installation.
if [ -n "${BASH_VERSION:-}" ]; then
    nano_rs_env_file="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
    nano_rs_env_file="${(%):-%N}"
else
    echo 'Use bash or zsh to source scripts/env.sh.' >&2
    return 1
fi
nano_rs_project_dir="$(cd -- "$(dirname -- "$nano_rs_env_file")/.." && pwd)"
if [ -x "$nano_rs_project_dir/.tools/cargo/bin/rustup" ]; then
    export CARGO_HOME="$nano_rs_project_dir/.tools/cargo"
    export RUSTUP_HOME="$nano_rs_project_dir/.tools/rustup"
    case ":$PATH:" in
        *":$CARGO_HOME/bin:"*) ;;
        *) export PATH="$CARGO_HOME/bin:$PATH" ;;
    esac
fi
unset nano_rs_env_file nano_rs_project_dir

