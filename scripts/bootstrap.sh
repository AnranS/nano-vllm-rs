#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
export CARGO_HOME="$project_dir/.tools/cargo"
export RUSTUP_HOME="$project_dir/.tools/rustup"
export PATH="$CARGO_HOME/bin:$PATH"
toolchain="$(sed -n 's/^channel = "\([^"]*\)"/\1/p' rust-toolchain.toml)"
if [ -z "$toolchain" ]; then
    echo 'Missing channel in rust-toolchain.toml' >&2
    exit 1
fi
if ! command -v cc >/dev/null; then
    echo 'A C linker is required. On Ubuntu install build-essential first.' >&2
    exit 1
fi
mkdir -p "$project_dir/.tools"
if [ ! -x "$CARGO_HOME/bin/rustup" ]; then
    curl --proto '=https' --tlsv1.2 --fail --show-error --location \
        --connect-timeout 20 --max-time 180 --retry 2 \
        https://sh.rustup.rs --output "$project_dir/.tools/rustup-init.sh"
    sh "$project_dir/.tools/rustup-init.sh" -y --no-modify-path \
        --profile minimal --default-toolchain "$toolchain" \
        --component rustfmt --component clippy
else
    rustup toolchain install "$toolchain" --profile minimal \
        --component rustfmt --component clippy
fi
rustc --version
cargo --version
echo 'Ready. Run: source scripts/env.sh'

