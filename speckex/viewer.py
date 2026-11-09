"""Open a generated PNG in the operating system's default image viewer."""

import os
import platform
import shutil
import subprocess


# Choose the desktop opener without using a shell or assuming one display service.
def viewer_command():
    system = platform.system()
    if system == "Darwin":
        opener = shutil.which("open")
        return ([opener] if opener else None), "macOS open command unavailable"
    if system == "Linux":
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            return None, "no graphical display is available"
        opener = shutil.which("xdg-open")
        if opener:
            return [opener], ""
        opener = shutil.which("gio")
        return ([opener, "open"] if opener else None), "Linux desktop opener unavailable"
    return None, f"no default-image-viewer opener configured for {system}"


# Launch the viewer asynchronously; extraction products remain valid if opening fails.
def open_image(path):
    command, unavailable = viewer_command()
    if command is None:
        return False, unavailable
    try:
        process = subprocess.Popen([*command, str(path)], stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   start_new_session=True)
    except OSError as exc:
        return False, f"viewer launch failed: {exc}"
    if process.poll() not in (None, 0):
        return False, f"viewer exited with status {process.returncode}"
    return True, f"sent {path} to the default image viewer"
