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
#   slurm/submit.sh [--config FILE] [SBATCH_OVERRIDES] [--] PIPELINE_ARGS
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
#   # Settings from a YAML file (template: slurm/submit.yaml); anything typed
#   # on the command line wins over the file:
#   slurm/submit.sh --config slurm/submit.yaml
#   slurm/submit.sh --config slurm/submit.yaml --mem 64G -- --subject 67 --session 2
#
# Recognised SBATCH overrides (they go first; everything after them, or
# after `--`, is a hypnose-eeg-pipeline argument):
#   --time, -t                 wall-clock limit per task (e.g. 48:00:00)
#   --mem                      RAM per task (e.g. 64G)
#   --cpus-per-task, -c        CPU cores per task
#   --partition, -p            SLURM partition
#   --max-running N            at most N array tasks at once (--array=0-M%N)
#   --config FILE              read resources and pipeline arguments from a YAML
#                              file (slurm/submit.yaml); see
#                              src/hypnose_eeg/pipeline/submit_config.py
#
# Both `--flag value` and `--flag=value` forms are accepted.  For any other
# sbatch option, edit run_pipeline_array.sbatch in place or use raw `sbatch`.
#
# All paths are resolved against the repo root (the parent of this script),
# so you can run it from anywhere.  Logs, frozen task lists and per-task
# reports go to <derivatives>/slurm/ (logs/, tasks/, reports/), where
# <derivatives> is the root the pipeline arguments resolve to.
# ============================================================================
set -euo pipefail

# Where submit.sh was run from, for a relative --config path.
CALLER_DIR="${PWD}"
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

# Peel off submit.sh's own options -- the SLURM resources and --config --
# from the front of the argument list; the rest are pipeline arguments.
TIME="" MEM="" CPUS="" PARTITION="" MAX_RUNNING="" CONFIG=""
while (( $# > 0 )); do
    case "$1" in
        --)
            shift
            break
            ;;
        --*=*)
            name="${1%%=*}"; value="${1#*=}"; consumed=1
            ;;
        --time|-t|--mem|--cpus-per-task|-c|--partition|-p|--max-running|--config)
            if (( $# < 2 )); then
                echo "Missing value for $1" >&2
                exit 2
            fi
            name="$1"; value="$2"; consumed=2
            ;;
        *)
            # The first argument that is not ours starts the pipeline
            # arguments (usually --subject).
            break
            ;;
    esac
    case "${name}" in
        --time|-t) TIME="${value}" ;;
        --mem) MEM="${value}" ;;
        --cpus-per-task|-c) CPUS="${value}" ;;
        --partition|-p) PARTITION="${value}" ;;
        --max-running) MAX_RUNNING="${value}" ;;
        --config) CONFIG="${value}" ;;
        *) break ;;  # a pipeline option written --flag=value
    esac
    shift "${consumed}"
done

# A --config file fills in whatever the command line left out: its sbatch
# values where no flag was given, and its pipeline options except those the
# command line replaces (src/hypnose_eeg/pipeline/submit_config.py).
if [[ -n "${CONFIG}" ]]; then
    [[ "${CONFIG}" == /* ]] || CONFIG="${CALLER_DIR}/${CONFIG}"
    if ! CONFIG_VALUES=$(uv run --no-sync python -m hypnose_eeg.pipeline.submit_config "${CONFIG}" -- "$@"); then
        echo "Could not read the submit config ${CONFIG}" >&2
        exit 1
    fi
    # Every value in it is shell-quoted by the helper.
    eval "${CONFIG_VALUES}"
    : "${TIME:=${CFG_TIME}}" "${MEM:=${CFG_MEM}}" "${CPUS:=${CFG_CPUS_PER_TASK}}"
    : "${PARTITION:=${CFG_PARTITION}}" "${MAX_RUNNING:=${CFG_MAX_RUNNING}}"
    set -- ${CFG_PIPELINE_ARGS[@]+"${CFG_PIPELINE_ARGS[@]}"}
fi

# Forwarded to `sbatch` ahead of the script name, where they take precedence
# over the matching #SBATCH directives.
SBATCH_OVERRIDES=()
if [[ -n "${TIME}" ]]; then SBATCH_OVERRIDES+=(--time "${TIME}"); fi
if [[ -n "${MEM}" ]]; then SBATCH_OVERRIDES+=(--mem "${MEM}"); fi
if [[ -n "${CPUS}" ]]; then SBATCH_OVERRIDES+=(--cpus-per-task "${CPUS}"); fi
# The partition also applies to the report-merging job; its other resources
# are its own (see the end of this script).
PARTITION_OVERRIDE=()
if [[ -n "${PARTITION}" ]]; then
    SBATCH_OVERRIDES+=(--partition "${PARTITION}")
    PARTITION_OVERRIDE=(--partition "${PARTITION}")
fi

if (( $# == 0 )); then
    echo "No pipeline arguments given; for example:" >&2
    echo "  slurm/submit.sh --subject 66 --all-sessions --model my-model" >&2
    echo "  slurm/submit.sh --config slurm/submit.yaml" >&2
    exit 2
fi

# Logs, frozen task lists and per-task reports go to <derivatives>/slurm/,
# beside the outputs they describe -- the derivatives root these pipeline
# arguments resolve to (--derivatives-root, else the data profile).
if ! DERIVATIVES_ROOT=$(uv run --no-sync hypnose-eeg-pipeline "$@" --print-derivatives-root); then
    echo "Could not resolve the derivatives root for: $*" >&2
    exit 1
fi
JOB_DIR="${DERIVATIVES_ROOT}/slurm"
LOG_DIR="${JOB_DIR}/logs"
mkdir -p "${LOG_DIR}" "${JOB_DIR}/tasks" "${JOB_DIR}/reports"
TASK_LIST="${JOB_DIR}/tasks/tasks_$(date +%Y%m%d-%H%M%S)_$$.txt"
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
echo "  logs: ${LOG_DIR}"
if [[ -n "${CONFIG}" ]]; then
    echo "  config: ${CONFIG}"
fi
echo "  pipeline arguments: $*"
sed 's/^/  /' "${TASK_LIST}"
if (( ${#SBATCH_OVERRIDES[@]} > 0 )); then
    echo "  sbatch overrides: ${SBATCH_OVERRIDES[*]}"
fi
# `${a[@]+...}` expands an empty array to nothing under `set -u`, even on
# bash < 4.4.  --parsable prints the job ID (`ID` or `ID;cluster`).
JOB_ID=$(sbatch --parsable \
    ${SBATCH_OVERRIDES[@]+"${SBATCH_OVERRIDES[@]}"} \
    --output="${LOG_DIR}/hypnose_eeg_%A_%a.out" \
    --error="${LOG_DIR}/hypnose_eeg_%A_%a.err" \
    --export=ALL,REPO_DIR="${REPO_DIR}",JOB_DIR="${JOB_DIR}",TASK_LIST="${TASK_LIST}" \
    --array="${ARRAY}" \
    "${REPO_DIR}/slurm/run_pipeline_array.sbatch" "$@")
JOB_ID="${JOB_ID%%;*}"
echo "Submitted job array ${JOB_ID}"

# One report for the whole array: once every task has ended, whatever its
# state (afterany), merge the per-task reports in <derivatives>/slurm/reports/<job>/.
# Small resources of its own; only the partition follows the overrides.
if MERGE_ID=$(sbatch --parsable \
    ${PARTITION_OVERRIDE[@]+"${PARTITION_OVERRIDE[@]}"} \
    --dependency="afterany:${JOB_ID}" \
    --job-name=hypnose_eeg_report \
    --array=0 --cpus-per-task=1 --mem=4G --time=00:30:00 \
    --output="${LOG_DIR}/hypnose_eeg_${JOB_ID}_report.out" \
    --error="${LOG_DIR}/hypnose_eeg_${JOB_ID}_report.err" \
    --export=ALL,REPO_DIR="${REPO_DIR}",JOB_DIR="${JOB_DIR}",TASK_LIST="${TASK_LIST}",MERGE_REPORTS_FOR="${JOB_ID}" \
    "${REPO_DIR}/slurm/run_pipeline_array.sbatch" "$@"); then
    echo "Submitted report job ${MERGE_ID%%;*} (runs after ${JOB_ID} ends)"
else
    echo "WARNING: could not submit the report job; the per-task reports stay in" >&2
    echo "  ${JOB_DIR}/reports/${JOB_ID}/ -- merge them once the array ends with:" >&2
    echo "  uv run hypnose-eeg-pipeline $* --merge-reports ${JOB_DIR}/reports/${JOB_ID} --task-file ${TASK_LIST}" >&2
fi
