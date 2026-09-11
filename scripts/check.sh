#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
source scripts/env.sh
cargo fmt --all -- --check
cargo clippy --locked --offline --all-targets -- -D warnings
cargo test --locked --offline --lib
cargo build --release --locked --offline

echo 'Standard checks passed. GPU tests are explicit:'
echo 'cargo test --locked --offline --lib gpu::tests -- --ignored --test-threads=1'
