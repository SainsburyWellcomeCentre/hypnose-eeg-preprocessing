# SLURM workflow — SWC HPC (CPU)

Run `hypnose-eeg-pipeline` as a SLURM job array on CPU nodes. By default each
array task processes **one subject**, running its selected sessions in order;
`--task-unit session` makes each task **one session** instead. The environment
is managed with [uv](https://docs.astral.sh/uv/), and code lives in your NFS
home (`/nfs/nhome/live/$USER`). Data and outputs live on
`/ceph/harris/hypnose/hypnose_eeg/...`, reached through the `swc-hpc`
data-location profile.

```
SLURM job 12345  (default: --task-unit subject)
 ├── array task 0 → sub-066 → ses-001, ses-002, ses-003 in order → 4 CPUs
 ├── array task 1 → sub-067 → ses-001, ses-002            in order → 4 CPUs
 └── array task 2 → sub-068 → ses-001                              → 4 CPUs
```

| File | Purpose |
|---|---|
| `slurm/run_pipeline_array.sbatch` | The array job: finds its task with `--list-tasks`, runs it with `--task-index` |
| `slurm/submit.sh` | Wrapper: counts the tasks, freezes the task list, sizes `--array`, submits |
| `slurm/logs/` | SLURM stdout/stderr per array task (git-ignored) |
| `slurm/tasks/` | Task lists frozen by `submit.sh` (git-ignored) |

---

## 1. One-time setup

On a login node. The pipeline installs `hypnose-helpers` and
`hypnose-somnotate` from **sibling checkouts**, so clone all three side by side:

```bash
# uv, installed to ~/.local/bin (the scripts look there too)
curl -LsSf https://astral.sh/uv/install.sh | sh

mkdir -p /nfs/nhome/live/$USER/Repos
cd /nfs/nhome/live/$USER/Repos
git clone https://github.com/SainsburyWellcomeCentre/hypnose-helpers.git
git clone https://github.com/SainsburyWellcomeCentre/hypnose-somnotate.git
git clone https://github.com/SainsburyWellcomeCentre/hypnose-eeg-preprocessing.git
cd hypnose-eeg-preprocessing

# Optional: build the environment on /ceph instead of in the repo's .venv.
# Put it in ~/.bashrc so every shell, and every job submitted from one, sees it.
echo 'export UV_PROJECT_ENVIRONMENT=/ceph/python_envs/$USER/hypnose-eeg-env' >> ~/.bashrc
echo 'export UV_LINK_MODE=copy' >> ~/.bashrc   # cache and env on different filesystems
source ~/.bashrc

# Exact versions from uv.lock; the notebook group (JupyterLab, Qt) is not
# needed on the cluster.
uv sync --frozen --no-group notebook

# Point this checkout at /ceph
uv run hypnose-eeg-locations swc-hpc
uv run hypnose-eeg-locations --show      # both roots should say OK
```

uv downloads Python 3.12 itself, so no conda or system module is needed.
Somnotate's `pomegranate` dependency builds from source; if `uv sync` fails for
want of a C/C++ compiler, load one (`module avail gcc`, then `module load …`)
and rerun it.

The environment lives at `$UV_PROJECT_ENVIRONMENT` when that is set, otherwise
in `.venv` in the repo. `uv sync`, `uv run`, `submit.sh` and the job all follow
the variable, so it must be set the same way when you sync and when you
submit; `sbatch` passes it from your shell to the jobs. The task header in
each `.out` log prints the environment the job used.

After `git pull`, rerun `uv sync --frozen --no-group notebook` on the login
node. Jobs never sync (`uv run --no-sync`): every array task shares the one
environment, and a sync racing another task would rewrite it under them.

## 2. Sanity-check

Preview the tasks a selection splits into. This touches nothing:

```bash
uv run hypnose-eeg-pipeline --subject 66 67 --all-sessions --model my-model --list-tasks
# sub-066 ses-001_date-20260717 ses-002_date-20260718
# sub-067 ses-001_date-20260720
```

Each line is one array task: its subject, then the session directories it
runs. Sessions a selection asks for but the subject does not have are reported
on stderr and belong to no task.

Before queuing a large array, run one session on a compute node to check the
model path, `/ceph` mount, and memory:

```bash
srun --partition=cpu --cpus-per-task=4 --mem=32G --time=02:00:00 --pty bash
uv run --no-sync hypnose-eeg-pipeline --subject 66 --session 1 --model my-model
exit
```

## 3. Submit the array

Every argument is passed through to `hypnose-eeg-pipeline`, so selections
work exactly as in the main readme (`--subject all`, `66:1,3`, `--session 2-5`,
`--date …`, `--stage`, `--overwrite`, …).

### A. Wrapper (recommended)

Run from the repo root:

```bash
./slurm/submit.sh --subject 66 67 --all-sessions --model my-model
./slurm/submit.sh --subject all --all-sessions --model my-model
./slurm/submit.sh --subject 66:1,3 67:2-4 --stage qc
```

The wrapper writes the task list to `slurm/tasks/` and submits that many array
tasks. A task whose list no longer matches refuses to run rather than run the
wrong session — for example, after a new session appears in rawdata while the
array is still pending. Resubmit in that case.

SLURM overrides go **first**, and `--` separates them from the pipeline
arguments:

```bash
./slurm/submit.sh --time 48:00:00 --mem 64G -- --subject 66 --all-sessions --model my-model
./slurm/submit.sh --max-running 10 -- --subject all --all-sessions --model my-model
```

| Override | Effect |
|---|---|
| `--time`, `-t` | Wall-clock limit per task |
| `--mem` | RAM per task |
| `--cpus-per-task`, `-c` | CPU cores per task (thread pools follow it) |
| `--partition`, `-p` | SLURM partition |
| `--max-running N` | At most N tasks at once (`--array=0-M%N`), to go easy on `/ceph` |

### B. Raw `sbatch`

```bash
cd /nfs/nhome/live/$USER/Repos/hypnose-eeg-preprocessing
ARGS=(--subject 66 67 --all-sessions --model my-model)
N=$(uv run --no-sync hypnose-eeg-pipeline "${ARGS[@]}" --list-tasks | wc -l)
sbatch --array=0-$((N - 1)) slurm/run_pipeline_array.sbatch "${ARGS[@]}"
```

Submit **from the repo root**: the job finds the code through
`$SLURM_SUBMIT_DIR`, and the log paths are relative to it. Any `sbatch` flag
placed before the script name, such as `--qos` or `--exclude`, overrides the
matching `#SBATCH` directive. Raw `sbatch` skips the frozen-task-list check.

## 4. One task per subject, or per session?

| | `--task-unit subject` (default) | `--task-unit session` |
|---|---|---|
| One task runs | Every selected session of one subject, in order | One session |
| Parallelism | Across subjects | Across all sessions |
| Wall time | Sum of the subject's sessions | One session |
| Short-recording baselines | Always computed from finished sessions | May race (see below) |

A recording with less than 6 h of signal is normalized against a nearby
session of the **same animal** (main readme, "Short recordings: borrowing a
baseline"). If that session has not been processed yet, its statistics are
computed from its EDF and cached in *its* derivatives folder. With one task
per subject this happens in order, as in an interactive batch run. With one
task per session, two tasks can compute that file at once, and the borrowed
baseline may be computed before the reference session's artifact prescan
exists. Use `--task-unit session` when the selection has no short recordings,
when each short recording's reference is already scored, or when you rerun
those sessions afterwards with `--overwrite`.

Each per-session task writes its own report,
`batch_report_<timestamp>_<session dir>.csv`, so tasks of one subject that
start together do not overwrite each other's.

## 5. Resources

The defaults in `run_pipeline_array.sbatch` are `--partition=cpu`,
`--cpus-per-task=4`, `--mem=32G` and `--time=24:00:00`. The 24 h covers a
subject task with many sessions; a session-unit task needs far less, and a
shorter `--time` usually starts sooner. Check the partition names with
`sinfo -s`. `OMP_NUM_THREADS` and the BLAS thread pools are set to
`--cpus-per-task`, so raising it does give numpy and MNE more cores. Look at
`sacct -j <job> --format=JobID,State,Elapsed,MaxRSS` after a first run to size
the rest.

| Goal | How |
|---|---|
| Longer wall time (one-off) | `./slurm/submit.sh --time 48:00:00 -- …` |
| More RAM (one-off) | `./slurm/submit.sh --mem 64G -- …` |
| Change the baseline for everyone | Edit the `#SBATCH` directive in `run_pipeline_array.sbatch` |
| Recompute finished outputs | Add `--overwrite` to the pipeline arguments |
| Run one task again | `sbatch --array=3 slurm/run_pipeline_array.sbatch <same arguments>` |

## 6. Monitor

```bash
squeue -u $USER                                   # pending / running
sacct -j 12345 -X --format=JobID,State,ExitCode,Elapsed,MaxRSS,NodeList
tail -f slurm/logs/hypnose_eeg_12345_0.out        # array task 0
find slurm/logs -name 'hypnose_eeg_12345_*.err' ! -empty   # tasks with errors
scancel 12345                                     # the whole array (12345_0 for one task)
```

## 7. Logs and results

| File | Contents |
|---|---|
| `slurm/logs/hypnose_eeg_<job>_<task>.out` | Task header (node, CPUs, commit, environment, arguments, task, data profile), then the pipeline's output and batch summary |
| `slurm/logs/hypnose_eeg_<job>_<task>.err` | Warnings, tracebacks, `FAILED:` lines |
| `<derivatives>/sub-XXX/batch_report_<timestamp>[_<session>].csv` | One row per session: status, QC status, failed step, error, what was erased |
| `<derivatives>/sub-XXX/ses-…/eeg/` | The outputs themselves (main readme, "Output folders") |

A task exits non-zero, and `sacct` shows `FAILED`, when any of its sessions
failed or came out QC FAIL. As in any batch run, a failed session's outputs are
erased so a rerun recomputes it. Resubmitting the same arguments skips every
step whose output already exists, so it costs little for the sessions that
completed. `hypnose-eeg-qc-review --subject …` lists the sessions whose QC
needs review.

| Symptom | Likely cause |
|---|---|
| `uv not found` | uv not installed in `~/.local/bin` (§1) |
| `No uv environment at …` | `uv sync` not run, or run with a different `UV_PROJECT_ENVIRONMENT` (§1) |
| `REPO_DIR does not look like the repo root` | Raw `sbatch` submitted from outside the repo |
| `The tasks for this selection changed since submission` | Rawdata changed while the array was pending; resubmit |
| `rawdata … MISSING` in the task header | `/ceph` not mounted on that node, or the wrong profile (§1) |
| `OUT_OF_MEMORY` in `sacct` | Raise `--mem` |
| `TIMEOUT` in `sacct` | Raise `--time`, or use `--task-unit session` |
