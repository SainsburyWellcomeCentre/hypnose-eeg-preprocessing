#!/usr/bin/env bash
# ============================================================================
# Convenience wrapper around `sbatch slurm/run_pipeline_array.sbatch`.
#
# Counts the tasks the pipeline selection splits into and submits exactly
# that many array tasks — so you don't have to compute N and stitch together
# --export/--array by hand.  The task list is frozen at submit time, and each
# array task runs its line of it, so sessions added to rawdata while the
# array is pending are left for the next submission.  A small follow-up job,
# queued to run once every task has ended, merges the tasks' reports into one
# batch report for the whole array.
#
# Usage:
#   slurm/submit.sh [SBATCH_OVERRIDES] [--] PIPELINE_ARGS
#
# Examples:
#   slurm/submit.sh --subject 66 67 --all-sessions --model my-model
#   slurm/submit.sh --subject all --all-sessions --model my-model --task-unit session
#   slurm/submit.sh --subject 66:1,3 67:2-4 --stage qc
#
#   # Bump SLURM resources for this submission only (overrides the matching
#   # #SBATCH directives in run_pipeline_array.sbatch; nothing edited on disk):
#   slurm/submit.sh --time 48:00:00 --mem 64G -- --subject 66 --all-sessions --model my-model
#   slurm/submit.sh --max-running 10 -- --subject all --all-sessions --model my-model
#
# Recognised SBATCH overrides (they go first; everything after them, or
# after `--`, is a hypnose-eeg-pipeline argument):
#   --time, -t                 wall-clock limit per task (e.g. 48:00:00)
#   --mem                      RAM per task (e.g. 64G)
#   --cpus-per-task, -c        CPU cores per task
#   --partition, -p            SLURM partition
#   --max-running N            at most N array tasks at once (--array=0-M%N)
#
# Both `--flag value` and `--flag=value` forms are accepted.  For any other
# sbatch option, edit run_pipeline_array.sbatch in place or use raw `sbatch`.
#
# All paths are resolved against the repo root (the parent of this script),
# so you can run it from anywhere.
# ============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$(realpath "$0")")/.." && pwd)"
cd "${REPO_DIR}"

# uv comes from the cluster's module; load it if this shell has not.  The
# jobs inherit the resulting PATH through --export=ALL below.
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

# The environment the jobs will use; sbatch passes UV_PROJECT_ENVIRONMENT on
# to them through --export=ALL below.
VENV="${UV_PROJECT_ENVIRONMENT:-.venv}"
if [[ ! -d "${VENV}" ]]; then
    echo "No uv environment at ${VENV}; run \`uv sync --frozen --no-group notebook\` first (slurm/README.md §1)." >&2
    exit 1
fi

# Peel off any SBATCH override flags from the front of the argument list.
# They get forwarded to `sbatch` ahead of the script name, where they take
# precedence over the matching #SBATCH directives.
SBATCH_OVERRIDES=()
# The partition also applies to the report-merging job; its other resources
# are its own (see the end of this script).
PARTITION_OVERRIDE=()
MAX_RUNNING=""
while (( $# > 0 )); do
    case "$1" in
        --)
            shift
            break
            ;;
        --time=*|--mem=*|--cpus-per-task=*|--partition=*)
            SBATCH_OVERRIDES+=("$1")
            [[ "$1" == --partition=* ]] && PARTITION_OVERRIDE=("$1")
            shift
            ;;
        --time|-t|--mem|--cpus-per-task|-c|--partition|-p)
            if (( $# < 2 )); then
                echo "Missing value for $1" >&2
                exit 2
            fi
            SBATCH_OVERRIDES+=("$1" "$2")
            [[ "$1" == --partition || "$1" == -p ]] && PARTITION_OVERRIDE=("$1" "$2")
            shift 2
            ;;
        --max-running=*)
            MAX_RUNNING="${1#*=}"; shift
            ;;
        --max-running)
            if (( $# < 2 )); then
                echo "Missing value for $1" >&2
                exit 2
            fi
            MAX_RUNNING="$2"; shift 2
            ;;
        *)
            # The first argument that is not an override starts the
            # pipeline arguments (usually --subject).
            break
            ;;
    esac
done

if (( $# == 0 )); then
    echo "No pipeline arguments given; for example:" >&2
    echo "  slurm/submit.sh --subject 66 --all-sessions --model my-model" >&2
    exit 2
fi

mkdir -p slurm/logs slurm/tasks
TASK_LIST="${REPO_DIR}/slurm/tasks/tasks_$(date +%Y%m%d-%H%M%S)_$$.txt"
# Missing sessions and selection errors print to stderr, shown here as-is.
if ! uv run --no-sync hypnose-eeg-pipeline "$@" --list-tasks > "${TASK_LIST}"; then
    rm -f "${TASK_LIST}"
    echo "Could not list the tasks for: $*" >&2
    exit 1
fi
N=$(wc -l < "${TASK_LIST}")

ARRAY="0-$((N - 1))"
if [[ -n "${MAX_RUNNING}" ]]; then
    ARRAY="${ARRAY}%${MAX_RUNNING}"
fi

echo "Submitting ${N} task(s) (array ${ARRAY}); task list: ${TASK_LIST}"
echo "  environment: ${VENV}"
sed 's/^/  /' "${TASK_LIST}"
if (( ${#SBATCH_OVERRIDES[@]} > 0 )); then
    echo "  sbatch overrides: ${SBATCH_OVERRIDES[*]}"
fi
# `${a[@]+...}` expands an empty array to nothing under `set -u`, even on
# bash < 4.4.  --parsable prints the job ID (`ID` or `ID;cluster`).
JOB_ID=$(sbatch --parsable \
    ${SBATCH_OVERRIDES[@]+"${SBATCH_OVERRIDES[@]}"} \
    --export=ALL,REPO_DIR="${REPO_DIR}",TASK_LIST="${TASK_LIST}" \
    --array="${ARRAY}" \
    "${REPO_DIR}/slurm/run_pipeline_array.sbatch" "$@")
JOB_ID="${JOB_ID%%;*}"
echo "Submitted job array ${JOB_ID}"

# One report for the whole array: once every task has ended, whatever its
# state (afterany), merge the per-task reports in slurm/reports/<job>/.
# Small resources of its own; only the partition follows the overrides.
if MERGE_ID=$(sbatch --parsable \
    ${PARTITION_OVERRIDE[@]+"${PARTITION_OVERRIDE[@]}"} \
    --dependency="afterany:${JOB_ID}" \
    --job-name=hypnose_eeg_report \
    --array=0 --cpus-per-task=1 --mem=4G --time=00:30:00 \
    --output="slurm/logs/hypnose_eeg_${JOB_ID}_report.out" \
    --error="slurm/logs/hypnose_eeg_${JOB_ID}_report.err" \
    --export=ALL,REPO_DIR="${REPO_DIR}",TASK_LIST="${TASK_LIST}",MERGE_REPORTS_FOR="${JOB_ID}" \
    "${REPO_DIR}/slurm/run_pipeline_array.sbatch" "$@"); then
    echo "Submitted report job ${MERGE_ID%%;*} (runs after ${JOB_ID} ends)"
else
    echo "WARNING: could not submit the report job; the per-task reports stay in" >&2
    echo "  slurm/reports/${JOB_ID}/ -- merge them once the array ends with:" >&2
    echo "  uv run hypnose-eeg-pipeline $* --merge-reports slurm/reports/${JOB_ID} --task-file ${TASK_LIST}" >&2
fi
