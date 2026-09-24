# Remote visualization from macOS and VS Code

The scored-recording viewer runs Python, reads EEG data, and renders the figure
on the remote Linux machine. SSH/X11 forwards only the graphical window to the
local Mac:

```text
remote Python + Qt -> encrypted SSH/X11 tunnel -> local XQuartz window
```

The remote and local requirements are separate. The Conda environment supplies
Python, PyQt, and Qt's Linux XCB libraries. The Mac supplies XQuartz and the SSH
client configuration. Creating the remote environment cannot install or
configure software on the local Mac.

## 1. Remote environment

Create or update the environment from the repository root on the remote host:

```bash
conda activate hypnose-eeg-env
python -m pip uninstall -y PyQt6 PyQt6-Qt6 PyQt6-sip
conda env update --name hypnose-eeg-env --file environment.yml --prune
conda deactivate
conda activate hypnose-eeg-env
```

The uninstall step is needed only when updating an environment created from an
older revision, which installed PyQt as pip wheels. It runs before Conda installs
its replacement and avoids two package managers owning the same Qt files. A new
environment does not need that migration step.

For a clean installation, use:

```bash
conda env create --file environment.yml
```

The environment installs PyQt through Conda together with `xcb-util-cursor`,
`xcb-util-image`, `xcb-util-keysyms`, `xcb-util-renderutil`, and `xcb-util-wm`.
These satisfy the native `libqxcb.so` runtime that pip-only PyQt installations
can leave unresolved on headless Linux machines.

The remote SSH service must allow X11 forwarding and have `xauth` installed. A
user can verify `xauth` with:

```bash
command -v xauth
```

If forwarding is disabled in the SSH service, an administrator must set
`X11Forwarding yes` in `sshd_config` and reload the service.

## 2. Local Mac

Install XQuartz from <https://www.xquartz.org/>, start it, and then open a new
local terminal. Verify that the local display exists:

```bash
echo "$DISPLAY"
```

An XQuartz value resembles:

```text
/var/run/com.apple.launchd.../org.xquartz:0
```

Add a host alias to the **local Mac's** `~/.ssh/config`. Put the specific host
block before a broad `Host *` block:

```sshconfig
Host hypnose-server
    HostName actual.remote.hostname
    User your-username
    ForwardX11 yes
    ForwardX11Trusted yes
    XAuthLocation /opt/X11/bin/xauth
```

Restrict the configuration file permissions:

```bash
chmod 600 ~/.ssh/config
```

Confirm that the alias enables forwarding without manually passing `-Y`:

```bash
ssh -G hypnose-server | grep -Ei 'forwardx11|xauthlocation'
ssh hypnose-server 'echo "$DISPLAY"'
```

The effective settings should report `forwardx11 yes` and
`forwardx11trusted yes`; the remote display should resemble `localhost:10.0`.
Do not assign `DISPLAY` manually on the remote machine. SSH must create the
authenticated tunnel.

## 3. VS Code Remote SSH

VS Code must use the same local host alias that passed the terminal test.

1. Start XQuartz before VS Code.
2. In local VS Code User settings, point Remote SSH at the Mac configuration:

   ```json
   {
       "remote.SSH.configFile": "/Users/your-local-user/.ssh/config"
   }
   ```

3. Run `Remote-SSH: Kill VS Code Server on Host...` from the Command Palette.
4. Close all VS Code windows connected to the host and quit VS Code completely.
5. Start VS Code after XQuartz, preferably from the local terminal with `code`.
6. Run `Remote-SSH: Connect to Host...` and choose `hypnose-server`.
7. In a new remote integrated terminal, verify `echo "$DISPLAY"` is non-empty.

If the terminal test works but VS Code still loses `DISPLAY`, try these local
User settings, kill the remote VS Code server, and reconnect:

```json
{
    "remote.SSH.useLocalServer": false,
    "remote.SSH.useExecServer": false
}
```

Use `View -> Output -> Remote - SSH` to confirm that VS Code selected the
expected host alias and configuration file.

## 4. Launch the viewer

From the remote terminal, including a VS Code terminal once `DISPLAY` is set:

```bash
cd /home/volkan/repos/hypnose-eeg-analysis
conda activate hypnose-eeg-env
python -m hypnose_eeg.review.viewer \
  --subject 66 --date 20260717
```

The computation and data access stay remote; the interactive Qt window appears
on the Mac through XQuartz.

## Troubleshooting

- Empty remote `DISPLAY`: the SSH connection was not created with X11
  forwarding. Fix the local SSH configuration and establish a new connection.
- `xcb` plugin found but not loadable: update or recreate the Conda environment
  so its declared Qt/XCB packages are installed.
- `could not connect to display`: the Qt runtime loaded, but the X11 tunnel or
  local XQuartz process is unavailable.
- An old `tmux` or `screen` session can retain a missing/stale `DISPLAY`; test in
  a fresh SSH shell first.
- `QT_QPA_PLATFORM=offscreen` suppresses the interactive window and is therefore
  not a substitute for X11 forwarding.
