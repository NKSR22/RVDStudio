from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _print_kv(key: str, value: object) -> None:
    print(f"{key}: {value}")


def _run_ldd(path: Path) -> list[str]:
    if not shutil.which("ldd"):
        return []
    proc = subprocess.run(["ldd", str(path)], check=False, capture_output=True, text=True)
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    missing: list[str] = []
    for line in text.splitlines():
        if "not found" in line:
            missing.append(line.strip())
    return missing


def _suggest_packages(missing_lines: list[str]) -> list[str]:
    missing_libs: set[str] = set()
    for line in missing_lines:
        lib = line.split("=>", 1)[0].strip()
        if lib:
            missing_libs.add(lib)

    mapping = {
        "libxcb-cursor.so.0": "libxcb-cursor0",
        "libxcb-xinerama.so.0": "libxcb-xinerama0",
        "libxkbcommon-x11.so.0": "libxkbcommon-x11-0",
        "libGL.so.1": "libgl1",
        "libEGL.so.1": "libegl1",
        "libglib-2.0.so.0": "libglib2.0-0",
    }
    suggested = sorted({pkg for lib, pkg in mapping.items() if lib in missing_libs})
    return suggested


def main() -> int:
    print("== Robot Vision Data Studio: GUI Diagnose ==")
    _print_kv("python", sys.version.replace("\n", " "))
    _print_kv("platform", sys.platform)
    _print_kv("executable", sys.executable)
    _print_kv("cwd", os.getcwd())
    _print_kv("DISPLAY", os.environ.get("DISPLAY"))
    _print_kv("WAYLAND_DISPLAY", os.environ.get("WAYLAND_DISPLAY"))
    _print_kv("QT_QPA_PLATFORM", os.environ.get("QT_QPA_PLATFORM"))
    _print_kv("XDG_SESSION_TYPE", os.environ.get("XDG_SESSION_TYPE"))

    try:
        import PySide6  # noqa: PLC0415
        from PySide6.QtCore import QLibraryInfo, qVersion  # noqa: PLC0415
    except Exception as exc:
        print("\nPySide6 import failed.")
        _print_kv("error", repr(exc))
        print("\nOn Ubuntu, ensure you installed OS deps and then reinstall Python deps:")
        print("  sudo apt-get update")
        print("  sudo apt-get install -y python3-venv libgl1 libglib2.0-0")
        return 2

    _print_kv("PySide6", getattr(PySide6, "__version__", "unknown"))
    _print_kv("Qt", qVersion())

    plugins_dir = None
    try:
        plugins_dir = Path(QLibraryInfo.path(QLibraryInfo.PluginsPath))
    except Exception:
        plugins_dir = None

    print("\n== Qt Plugin Paths ==")
    _print_kv("plugins_dir", plugins_dir)
    if not plugins_dir or not plugins_dir.exists():
        print("Qt plugins directory not found. This usually indicates a broken PySide6 install.")
        return 3

    platforms_dir = plugins_dir / "platforms"
    _print_kv("platforms_dir", platforms_dir)
    if not platforms_dir.exists():
        print("Missing Qt 'platforms' plugins directory. Reinstall PySide6 in the venv.")
        return 3

    candidates = [
        platforms_dir / "libqxcb.so",
        platforms_dir / "libqwayland-egl.so",
        platforms_dir / "libqwayland-generic.so",
        platforms_dir / "libqminimal.so",
        platforms_dir / "libqoffscreen.so",
    ]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        print("No platform plugins found in Qt plugins path. Reinstall PySide6.")
        return 3

    print("\n== Platform Plugins Found ==")
    for p in existing:
        print(f"- {p.name}")

    if not sys.platform.startswith("linux"):
        print("\nNon-Linux platform: skipping ldd checks.")
        return 0

    xcb = platforms_dir / "libqxcb.so"
    if not xcb.exists():
        print("\nlibqxcb.so not found; if your desktop uses Wayland, try forcing wayland:")
        print("  QT_QPA_PLATFORM=wayland python -m app.gui")
        return 0

    print("\n== ldd (xcb plugin) ==")
    missing = _run_ldd(xcb)
    if not missing:
        print("No missing shared libraries detected for libqxcb.so.")
        print("If the GUI still does not appear, run with debug env:")
        print("  QT_DEBUG_PLUGINS=1 python -m app.gui")
        return 0

    print("Missing libraries:")
    for line in missing:
        print(f"- {line}")

    suggested = _suggest_packages(missing)
    if suggested:
        print("\nSuggested Ubuntu packages (based on missing libs):")
        print(f"  sudo apt-get install -y {' '.join(suggested)}")
    else:
        print("\nInstall the OS packages that provide the missing libraries above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

