#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-1.23.0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_ROOT="${CK_ORT_NATIVE_DIR:-$HOME/.cache/ck/onnxruntime/native-openvino}"
LIB_DIR="$INSTALL_ROOT/lib"
WHEEL_PATH="$INSTALL_ROOT/onnxruntime_openvino-${VERSION}.whl"

if ! command -v curl >/dev/null 2>&1; then
  echo "error: curl is required" >&2
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "error: $PYTHON_BIN is required" >&2
  exit 1
fi

mkdir -p "$LIB_DIR"

WHEEL_URL="$($PYTHON_BIN - "$VERSION" <<'PY'
import json
import sys
import urllib.request

version = sys.argv[1]
url = f"https://pypi.org/pypi/onnxruntime-openvino/{version}/json"
with urllib.request.urlopen(url) as r:
    data = json.load(r)

files = data.get("urls", [])
# We only need the bundled native .so files in the wheel.
# Any Linux x86_64 manylinux wheel is acceptable for extraction.
candidates = [
    f["url"]
    for f in files
    if f.get("packagetype") == "bdist_wheel"
    and "manylinux" in f.get("filename", "")
    and "x86_64" in f.get("filename", "")
]

if not candidates:
    raise SystemExit("No suitable onnxruntime-openvino Linux x86_64 wheel found for this version")

print(candidates[0])
PY
)"

echo "Downloading: $WHEEL_URL"
curl -fL "$WHEEL_URL" -o "$WHEEL_PATH"

"$PYTHON_BIN" - "$WHEEL_PATH" "$LIB_DIR" <<'PY'
import os
import pathlib
import sys
import zipfile

wheel_path = pathlib.Path(sys.argv[1])
lib_dir = pathlib.Path(sys.argv[2])
lib_dir.mkdir(parents=True, exist_ok=True)

# clean old libs in target directory
for entry in lib_dir.iterdir():
    if entry.is_file():
        entry.unlink()

count = 0
with zipfile.ZipFile(wheel_path) as zf:
    for name in zf.namelist():
        if not name.startswith("onnxruntime/capi/"):
            continue
        base = os.path.basename(name)
        if ".so" not in base:
            continue
        target = lib_dir / base
        with zf.open(name) as src, open(target, "wb") as dst:
            dst.write(src.read())
        count += 1

# Ensure generic SONAME link exists for ORT_DYLIB_PATH convenience
runtime_candidates = sorted(lib_dir.glob("libonnxruntime.so.*"))
if runtime_candidates:
    soname = lib_dir / "libonnxruntime.so"
    if soname.exists() or soname.is_symlink():
        soname.unlink()
    soname.symlink_to(runtime_candidates[-1].name)

print(f"Extracted {count} shared libraries to {lib_dir}")
PY

cat > "$INSTALL_ROOT/INSTALL_INFO.txt" <<EOF
Installed: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
Version: ${VERSION}
Wheel: ${WHEEL_URL}
LibDir: ${LIB_DIR}
EOF

echo
echo "Native OpenVINO runtime installed to: $LIB_DIR"
echo "ck auto-discovers this path."
echo
echo "Verify with:"
echo "  ck --rebenchmark --model bge-small"
