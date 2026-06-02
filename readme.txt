Mounting an SMB share and creating a repo symlink
================================================

Example remote SMB location:
  smb://ceph-gw02.hpc.swc.ucl.ac.uk/harris

Example Linux mount target:
  /mnt/harris

Example repo symlink:
  data/harris

Replace these example paths with the server, share, mount location, and
symlink name for your own setup.


1. Prerequisites
----------------

You need:

  - Network access to the SMB server, for example through the university,
    institute, or VPN network.
  - Permission to access the share.
  - Your username, password, and domain if the server uses one.
  - A local folder where the share will be mounted.

For the Harris share, the domain may be:

  AD


2. Mount on Linux
-----------------

Linux command-line mounts usually use CIFS syntax:

  //server/share

not browser-style SMB syntax:

  smb://server/share

For the Harris share:

  sudo mkdir -p /mnt/harris

  sudo mount -t cifs //ceph-gw02.hpc.swc.ucl.ac.uk/harris /mnt/harris \
    -o username=YOUR_USERNAME,domain=AD,vers=3.0,sec=ntlmssp,uid=$(id -u),gid=$(id -g)

Replace YOUR_USERNAME with your own username. The command should prompt for
your password.

3. Mount on macOS
-----------------

In Finder:

  1. Open Finder.
  2. Choose Go > Connect to Server.
  3. Enter the SMB address, for example:

       smb://ceph-gw02.hpc.swc.ucl.ac.uk/harris

  4. Enter your username, password, and domain if prompted.

From Terminal, macOS mounts usually appear under /Volumes after connecting
through Finder.

Check the mount:

  ls /Volumes


4. Mount on Windows
-------------------

In File Explorer:

  1. Open File Explorer.
  2. Choose Map network drive.
  3. Enter the UNC path, for example:

       \\ceph-gw02.hpc.swc.ucl.ac.uk\harris

  4. Choose a drive letter.
  5. Enter your username, password, and domain if prompted.

The mounted share should then appear as a drive such as H: or Z:.


5. Create a symlink from this repo to the mounted folder
-------------------------------------------------------

After the share is mounted, create a link from this repository to the mounted
location. This lets project code use a stable repo-local path while the large
or remote data stays outside git.

Linux example:

  mkdir -p data
  ln -s /mnt/harris data/harris

macOS example:

  mkdir -p data
  ln -s /Volumes/harris data/harris

Windows PowerShell example:

  New-Item -ItemType SymbolicLink -Path data\harris -Target Z:\

Replace Z:\ with the drive letter or mounted location for your system.

Check the symlink:

  ls -l data/harris
  ls /mnt/harris

On Windows PowerShell:

  Get-Item data\harris

The order matters on Linux and macOS:

  ln -s TARGET LINK_NAME

So this:

  ln -s /mnt/harris data/harris

means data/harris points to /mnt/harris.


6. If the symlink already exists
--------------------------------

Remove only the symlink, then recreate it.

Linux or macOS:

  rm data/harris
  ln -s /mnt/harris data/harris

Be careful not to add a trailing slash when removing the symlink.

Windows PowerShell:

  Remove-Item data\harris
  New-Item -ItemType SymbolicLink -Path data\harris -Target Z:\


7. After restarting the computer
--------------------------------

Manual mounts usually do not survive a restart. After rebooting, mount the
remote share again before using the symlink.

If the symlink exists but the share is not mounted, the repo-local link will
point to an empty or unavailable location until the remote share is mounted
again.
