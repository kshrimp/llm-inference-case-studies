#!/usr/bin/env bash
# CPU bisect: which torch release made BF16 -> float8_e4m3fn casts saturate instead of returning NaN?
# Each version is installed into a throwaway venv (CPU wheel only); nothing touches the project venv.
set -euo pipefail
WORK=${1:-$(mktemp -d)}
for version in 2.9.1 2.10.0 2.12.0 2.13.0 2.14.0 2.14.1; do
  rm -rf "$WORK/venv"
  uv venv -q --python 3.12 "$WORK/venv"
  VIRTUAL_ENV="$WORK/venv" uv pip install -q "torch==$version" --index-url https://download.pytorch.org/whl/cpu
  "$WORK/venv/bin/python" -I -W ignore -c "
import torch
x = torch.tensor([450., 464., 470., 516., -516.], dtype=torch.bfloat16)
print(torch.__version__, x.to(torch.float8_e4m3fn).float().tolist())" 2>&1 | grep -v -i numpy
done
rm -rf "$WORK/venv"
