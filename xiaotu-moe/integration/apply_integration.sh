#!/usr/bin/env bash
# Apply both xiaotu-moe integration patches to a copy of the Lvllmds4-x fork.
#
#  1. routed_experts.swap_xiaotu.patch     : point lk_moe.MOE_* -> xiaotu_moe.MOE_*
#                                            (module constructions + import only;
#                                             self.lk_moe instance attr + its
#                                             cpu_prefill/cpu_decode/gpu_prefill
#                                             method calls are LEFT AS-IS)
#  2. routed_experts.cpu_decode.patch      : bridge _cpu_decode (GPU pointers ->
#                                            pure-CPU: copy GPU->CPU, compute,
#                                            write back the fixed output_gpu buf)
#
# Usage: FORK_DIR=/path/to/Lvllmds4-x runtime/apply_integration.sh
#
# Before running the service, the fork's Python must be able to
# `import xiaotu_moe` (PYTHONPATH to the xiaotu-moe repo root, or pip install -e).
# N.B. this edits the fork's routed_experts.py; review `git diff` after applying
# and stop the running vLLM before restarting the service.
set -euo pipefail
INT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORK_DIR="${FORK_DIR:-$(cd "$INT/.." && pwd)/../Lvllmds4-x}"
cd "$FORK_DIR"
for P in routed_experts.swap_xiaotu.patch routed_experts.cpu_decode.patch; do
  if git apply --check "$INT/$P" 2>/dev/null; then
    git apply "$INT/$P"
    echo ">> applied $P"
  else
    echo ">> $P: could not apply (already applied? not a git worktree?)  -- review git diff"
  fi
done
echo ">> done. Review with: git diff  (paths: vllm/model_executor/layers/fused_moe/routed_experts.py)"
