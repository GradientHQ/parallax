#!/bin/bash
# Build the vLLM Rust frontend binary and install it into Parallax's pip scripts dir.
# Usage:
#   ./build_rust.sh [--debug]
#
# By default builds in release mode. Pass --debug for faster compile times
# during development.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

show_help() {
    cat <<'EOF'
Build the vLLM Rust frontend binary and install it into Parallax's pip scripts dir.

Usage:
  ./build_rust.sh [--debug]

Environment:
  VLLM_REF             vLLM git branch/tag to clone. Defaults to main.
  PARALLAX_PYTHON      Python interpreter for the Parallax installation.
EOF
}

resolve_parallax_python() {
    if [[ -n "${PARALLAX_PYTHON:-}" ]]; then
        if [[ ! -x "$PARALLAX_PYTHON" ]]; then
            echo "PARALLAX_PYTHON is not executable: $PARALLAX_PYTHON" >&2
            exit 1
        fi
        echo "$PARALLAX_PYTHON"
        return
    fi

    if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
        echo "$VIRTUAL_ENV/bin/python"
        return
    fi

    if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
        echo "$CONDA_PREFIX/bin/python"
        return
    fi

    if command -v python &>/dev/null; then
        command -v python
        return
    fi

    echo "Unable to find Python for the Parallax installation." >&2
    echo "Activate the Parallax environment or set PARALLAX_PYTHON." >&2
    exit 1
}

resolve_parallax_bin_dir() {
    local parallax_python
    parallax_python="$(resolve_parallax_python)"
    "$parallax_python" - <<'PY'
import sysconfig

print(sysconfig.get_path("scripts"))
PY
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    show_help
    exit 0
fi

if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != "--debug" ) ]]; then
    show_help >&2
    exit 2
fi

VLLM_REF="${VLLM_REF:-main}"
CLONE_PARENT="$(mktemp -d "${TMPDIR:-/tmp}/parallax-vllm-rs.XXXXXX")"
VLLM_CLONE_ROOT="$CLONE_PARENT/vllm"

cleanup_clone() {
    rm -rf "$CLONE_PARENT"
}

trap cleanup_clone EXIT

if ! command -v git &>/dev/null; then
    echo "git not found; install git and rerun this script." >&2
    exit 1
fi

echo "Cloning vLLM from https://github.com/vllm-project/vllm.git (ref: $VLLM_REF)"
git clone --depth 1 --branch "$VLLM_REF" \
    https://github.com/vllm-project/vllm.git \
    "$VLLM_CLONE_ROOT"

RUST_DIR="$VLLM_CLONE_ROOT/rust"
PARALLAX_SCRIPTS_DIR="$(resolve_parallax_bin_dir)"
TARGET_PATH="$PARALLAX_SCRIPTS_DIR/vllm-rs"

if [[ ! -f "$RUST_DIR/Cargo.toml" || ! -f "$VLLM_CLONE_ROOT/rust-toolchain.toml" ]]; then
    echo "Cloned repository does not contain the expected vLLM Rust frontend sources." >&2
    exit 1
fi

# Read the required toolchain from rust-toolchain.toml.
TOOLCHAIN=$(grep '^channel' "$VLLM_CLONE_ROOT/rust-toolchain.toml" | sed 's/.*= *"\(.*\)"/\1/')

if [[ -z "$TOOLCHAIN" ]]; then
    echo "Unable to read Rust toolchain from $VLLM_CLONE_ROOT/rust-toolchain.toml" >&2
    exit 1
fi

# Ensure rustup and the required toolchain are available.
if ! command -v rustup &>/dev/null; then
    echo "rustup not found, installing..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
fi

if ! rustup run "$TOOLCHAIN" rustc --version &>/dev/null; then
    echo "Installing Rust toolchain: $TOOLCHAIN"
    rustup toolchain install "$TOOLCHAIN"
fi

if [[ "${1:-}" == "--debug" ]]; then
    PROFILE_ARGS=()
    PROFILE_DIR="debug"
else
    PROFILE_ARGS=(--release)
    PROFILE_DIR="release"
fi

cargo +"$TOOLCHAIN" build "${PROFILE_ARGS[@]}" \
    --manifest-path "$RUST_DIR/Cargo.toml" \
    --bin vllm-rs \
    --features native-tls-vendored

mkdir -p "$(dirname "$TARGET_PATH")"
cp "$RUST_DIR/target/$PROFILE_DIR/vllm-rs" "$TARGET_PATH"
chmod +x "$TARGET_PATH"
echo "Installed vllm-rs to $TARGET_PATH"
