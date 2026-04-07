# Ubuntu Installation Guide

This guide explains how to install the packaged Ubuntu build of `Robot Vision Data Studio`.

## Package Layout

The Ubuntu release folder contains:

- compiled application bundle in `app/`
- `install_ubuntu.sh`
- `uninstall_ubuntu.sh`
- `rvdstudio.desktop`
- usage and install guides in `docs/`

## Install

```bash
chmod +x install_ubuntu.sh
./install_ubuntu.sh
```

The installer will:

- copy the app into `/opt/RVDStudio`
- create `/usr/local/bin/rvdstudio`
- install a desktop launcher in `/usr/share/applications/rvdstudio.desktop`

## Launch

```bash
rvdstudio
```

You can also launch it from the Applications menu.

## Remove

```bash
chmod +x uninstall_ubuntu.sh
./uninstall_ubuntu.sh
```
