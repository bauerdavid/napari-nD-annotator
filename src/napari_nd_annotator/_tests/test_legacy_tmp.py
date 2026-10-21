"""Feature folders that older versions left in the temp directory are removed; live processes keep theirs."""
import os
import subprocess
import sys
import time

import pytest

from napari_nd_annotator.minimal_contour import _legacy_tmp
from napari_nd_annotator.minimal_contour._legacy_tmp import TEMP_SUFFIX, clean_legacy_tmp, pid_alive


@pytest.fixture
def tmp_root(tmp_path, monkeypatch):
    monkeypatch.setattr(_legacy_tmp.tempfile, "tempdir", str(tmp_path))
    return tmp_path


def _dead_pid():
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def _folder(root, name):
    path = root / name
    path.mkdir()
    (path / "tmp_ftrs_x_0_1_v.dat").write_bytes(b"0" * 100)
    return path


def test_pid_alive():
    assert pid_alive(os.getpid())
    assert not pid_alive(0)
    assert not pid_alive(_dead_pid())


def test_dead_process_and_old_folders_are_removed_live_ones_kept(tmp_root):
    dead = _folder(tmp_root, "tmpab12_%d%s" % (_dead_pid(), TEMP_SUFFIX))
    live = _folder(tmp_root, "tmpcd34_%d%s" % (os.getpid(), TEMP_SUFFIX))
    old = _folder(tmp_root, "tmpef56%s" % TEMP_SUFFIX)
    recent = _folder(tmp_root, "tmpgh78%s" % TEMP_SUFFIX)
    other = _folder(tmp_root, "tmpij90_other")
    two_days_ago = time.time() - 2 * 24 * 60 * 60
    os.utime(old, (two_days_ago, two_days_ago))
    clean_legacy_tmp()
    assert not dead.exists()
    assert not old.exists()
    assert live.exists() and recent.exists() and other.exists()


def test_the_sweep_never_raises(tmp_root, monkeypatch):
    dead = _folder(tmp_root, "tmpab12_%d%s" % (_dead_pid(), TEMP_SUFFIX))

    def locked(path, *args, **kwargs):
        raise PermissionError(path)
    monkeypatch.setattr(_legacy_tmp.shutil, "rmtree", locked)
    clean_legacy_tmp()
    assert dead.exists()

    def broken(*args, **kwargs):
        raise RuntimeError("no temp dir")
    monkeypatch.setattr(_legacy_tmp.tempfile, "gettempdir", broken)
    clean_legacy_tmp()
