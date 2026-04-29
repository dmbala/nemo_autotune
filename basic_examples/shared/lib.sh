# Shared bash helpers used by launcher scripts and sbatch files. Source this
# after any `set -euo pipefail`; the helpers don't introduce their own.
#
#   source "${WORKSHOP_ROOT}/shared/lib.sh"

# abs_path <path> — prints the absolute path of its argument.
# Uses `realpath -m` so non-existent targets resolve cleanly (the callers in
# this workshop pass config paths that exist, but ckpt dirs may not yet).
abs_path() {
    local p="${1:?usage: abs_path <path>}"
    realpath -m -- "${p}"
}

# latest_consolidated_ckpt <root>... — prints the newest
# <root>/epoch_*_step_*/model/consolidated across all the given roots.
# Callers may pass a glob that expands to multiple roots before reaching us
# (e.g. `latest_consolidated_ckpt "$CKPT_ROOT/trackB_"*`).
# Prints nothing (exit 0) if no matches — callers should test for empty.
latest_consolidated_ckpt() {
    shopt -s nullglob
    local matches=()
    local root
    for root in "$@"; do
        matches+=("${root}"/epoch_*_step_*/model/consolidated)
    done
    shopt -u nullglob
    if (( ${#matches[@]} == 0 )); then
        return 0
    fi
    ls -td "${matches[@]}" | head -n1
}
