"""Removal of the feature folders that older versions left in the temp directory.

Versions before the lazy per-plane features wrote the image features to memory mapped files in a
``<tmp prefix><random>_<pid>_nd_annotator`` folder (versions before 0.4: without the pid) and removed it on exit,
which did not happen after a crash. Nothing is written there any more; this only cleans up what is left.
"""
import glob
import os
import re
import shutil
import sys
import tempfile
import time

TEMP_SUFFIX = "_nd_annotator"
# folders are named <tmpprefix><random>_<pid>_nd_annotator, so the owning process can be told apart from others
TEMP_FOLDER_PATTERN = re.compile(r"_(\d+)%s$" % re.escape(TEMP_SUFFIX))
# folders without a pid in their name come from older versions: only removed once they are this old
LEGACY_TEMP_SECONDS = 24 * 60 * 60


def pid_alive(pid: int) -> bool:
    """Whether a process with this id is running. Errs on the side of "alive" when it cannot tell."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        # os.kill must not be used here: on Windows any signal other than CTRL_*_EVENT terminates the process
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        process_query_limited_information = 0x1000
        error_access_denied = 5
        still_active = 259
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return ctypes.get_last_error() == error_access_denied
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)  # signal 0 only checks that the process exists
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, but belongs to another user
    except OSError:
        return True
    return True


def clean_legacy_tmp():
    """Remove the feature folders of dead processes and the old ones without a pid. Never raises."""
    try:
        # only remove leftovers of processes that no longer run: other napari/pytest processes keep their folders
        temp_dir = tempfile.gettempdir()
        temp_folders = glob.glob(os.path.join(temp_dir, "%s*%s" % (tempfile.gettempprefix(), TEMP_SUFFIX)))
        for fold in temp_folders:
            try:
                match = TEMP_FOLDER_PATTERN.search(os.path.basename(fold))
                if match is not None:
                    if pid_alive(int(match.group(1))):
                        continue
                elif time.time() - os.path.getmtime(fold) < LEGACY_TEMP_SECONDS:
                    continue
                shutil.rmtree(fold)
            except OSError:
                pass  # e.g. a file is still open in a process we could not tell was alive
    except Exception:
        pass
