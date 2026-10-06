#!/bin/bash
# ============================================================================
# Hypnose EEG pipeline — SLURM job array (CPU only)
#
# Every argument after the script name is a `hypnose-eeg-pipeline` argument:
# the selection (--subject/--session/--date/--all-sessions) plus any options
# (--model, --stage, --overwrite, --task-unit, ...).  The selection is split
# into tasks with `--list-tasks`; array task N runs `--task-index N` -- line N
# of the list slurm/submit.sh froze at submission, when there is one.  By
# default one task is one subject, its sessions run in order; pass
# `--task-unit session` for one task per session (see slurm/README.md).
#
# Submit (slurm/submit.sh does the counting for you):
#   ARGS=(--subject 66 67 --all-sessions --model my-model)
#   N=$(uv run --no-sync hypnose-eeg-pipeline "${ARGS[@]}" --list-tasks | wc -l)
#   sbatch --array=0-$((N - 1)) slurm/run_pipeline_array.sbatch "${ARGS[@]}"
#
# Resource overrides — the #SBATCH defaults below are the baseline.  Bump any
# of them for one submission by passing the flag before the script name:
#   sbatch --time=48:00:00 --mem=64G --array=0-N slurm/run_pipeline_array.sbatch ...
#   slurm/submit.sh --time 48:00:00 --mem 64G -- --subject 66 --all-sessions ...
# CLI-level sbatch flags take precedence over the #SBATCH directives.
# ============================================================================

#SBATCH --job-name=hypnose_eeg
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/hypnose_eeg_%A_%a.out
#SBATCH --error=slurm/logs/hypnose_eeg_%A_%a.err
# Default array of one task; override with `sbatch --array=0-N` at submit.
#SBATCH --array=0

set -euo pipefail

if (( $# == 0 )); then
    echo "No pipeline arguments: pass the selection after the script name," >&2
    echo "e.g. sbatch ... slurm/run_pipeline_array.sbatch --subject 66 --all-sessions --model my-model" >&2
    exit 2
fi

# ── uv ──────────────────────────────────────────────────────────────────────
# uv comes from the cluster's module (`module load uv`).  If it was loaded in
# the submitting shell, sbatch already passed it on through PATH; otherwise
# load it here.  A standalone install in ~/.local/bin also works.
if ! command -v uv >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    # Module scripts may read unset variables.
    set +u
    module load uv >/dev/null 2>&1 || true
    set -u
fi
export PATH="${PATH}:${HOME}/.local/bin"
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found: \`module load uv\` failed and it is not in ~/.local/bin (slurm/README.md §1)." >&2
    exit 1
fi

# ── Move to the repo root ──────────────────────────────────────────────────
# SLURM stages the batch script into its spool dir, so `realpath "$0"` does
# NOT point at the script inside the repo when running under sbatch.  Prefer
# an explicit override, then $SLURM_SUBMIT_DIR (set by sbatch to the
# directory you submitted from — submit from the repo root), then the
# staged-script fallback (only correct outside SLURM).
: "${REPO_DIR:=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$(realpath "$0")")/.." && pwd)}}"

if [[ ! -f "${REPO_DIR}/src/hypnose_eeg/pipeline/run.py" ]]; then
    echo "REPO_DIR does not look like the repo root (no src/hypnose_eeg/pipeline/run.py): ${REPO_DIR}" >&2
    echo "Submit from the repo root, or pass REPO_DIR=/path/to/repo in --export." >&2
    exit 1
fi
cd "${REPO_DIR}"

# The environment `uv sync` built: $UV_PROJECT_ENVIRONMENT when set (sbatch
# passes the submitting shell's environment through), else the repo's .venv.
VENV="${UV_PROJECT_ENVIRONMENT:-.venv}"
if [[ ! -d "${VENV}" ]]; then
    echo "No uv environment at ${VENV}; run \`uv sync --frozen --no-group notebook\` on a login node first" >&2
    echo "(with the same UV_PROJECT_ENVIRONMENT, see slurm/README.md §1)." >&2
    exit 1
fi

# ── CPU runtime ────────────────────────────────────────────────────────────
# Headless plotting (QC review figures are saved as PNGs).
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen
# Keep numpy/scipy/MNE thread pools within the allocation instead of one
# thread per core of the whole node.
THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"

# Never sync from a job: every array task shares this environment, and a sync
# racing another would rewrite it underneath the running tasks.
PIPELINE=(uv run --no-sync hypnose-eeg-pipeline)

# Each array task submitted by slurm/submit.sh leaves its report here, and a
# follow-up job merges them into one report for the whole array.
REPORTS_ROOT="${REPO_DIR}/slurm/reports"

# ── Merge the array's reports (the follow-up job) ──────────────────────────
# slurm/submit.sh queues this script once more, after every array task has
# ended, with MERGE_REPORTS_FOR set to the array's job ID.
if [[ -n "${MERGE_REPORTS_FOR:-}" ]]; then
    echo "Merging the reports of job array ${MERGE_REPORTS_FOR}"
    MERGE_ARGS=(--merge-reports "${REPORTS_ROOT}/${MERGE_REPORTS_FOR}")
    if [[ -n "${TASK_LIST:-}" && -f "${TASK_LIST}" ]]; then
        MERGE_ARGS+=(--task-file "${TASK_LIST}")
    fi
    exec "${PIPELINE[@]}" "$@" "${MERGE_ARGS[@]}"
fi

# ── Find this task ─────────────────────────────────────────────────────────
# slurm/submit.sh freezes the task list at submit time, and the task runs its
# line of that file (`--task-file`): sessions added to rawdata since then are
# left for the next submission instead of shifting the indices.  Without a
# frozen list (raw sbatch), list the selection as rawdata stands now.
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ -n "${TASK_LIST:-}" ]]; then
    if [[ ! -f "${TASK_LIST}" ]]; then
        echo "The task list frozen at submission is gone: ${TASK_LIST}; resubmit." >&2
        exit 1
    fi
    mapfile -t TASKS < "${TASK_LIST}"
    # After the selection's own arguments, so this --report wins over any
    # given there; that one is where the merged report goes.
    TASK_ARGS=(
        --task-index "${TASK_ID}" --task-file "${TASK_LIST}"
        --report "${REPORTS_ROOT}/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}/task-${TASK_ID}.csv"
    )
else
    mapfile -t TASKS < <("${PIPELINE[@]}" "$@" --list-tasks)
    TASK_ARGS=(--task-index "${TASK_ID}")
fi
N_TASKS=${#TASKS[@]}

if (( N_TASKS == 0 )); then
    echo "No tasks found for: $*" >&2
    exit 1
fi

if (( TASK_ID >= N_TASKS )); then
    echo "Array index ${TASK_ID} >= number of tasks (${N_TASKS}); nothing to do." >&2
    exit 0
fi

echo "─────────────────────────────────────────────────────────────────────────────"
echo "  Job        : ${SLURM_JOB_NAME:-} (${SLURM_JOB_ID:-}) array=${TASK_ID}"
echo "  Node       : $(hostname)"
echo "  CPUs / mem : ${THREADS} / ${SLURM_MEM_PER_NODE:-?}M"
echo "  Started    : $(date -Iseconds)"
echo "  Repo       : ${REPO_DIR} ($(git rev-parse --short HEAD 2>/dev/null || echo 'no git'))"
echo "  Env        : ${VENV}"
echo "  Arguments  : $*"
echo "  Task       : ${TASKS[$TASK_ID]}  (${TASK_ID} / $((N_TASKS - 1)))"
# The checkout's data profile; --rawdata-root/--derivatives-root above, or
# HYPNOSE_EEG_*_ROOT variables, take precedence over it.
echo "  Data profile:"
uv run --no-sync hypnose-eeg-locations --show 2>&1 | sed 's/^/    /' || true
echo "─────────────────────────────────────────────────────────────────────────────"

# ── Run the task ───────────────────────────────────────────────────────────
# Not `exec`: the footer below should report how the task ended.
status=0
"${PIPELINE[@]}" "$@" "${TASK_ARGS[@]}" || status=$?

echo "─────────────────────────────────────────────────────────────────────────────"
echo "  Finished   : $(date -Iseconds)  (exit ${status})"
echo "─────────────────────────────────────────────────────────────────────────────"
exit "${status}"
