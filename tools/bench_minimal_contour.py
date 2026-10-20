"""Benchmark of the Minimal Contour (MC) image/feature handling.

Not part of the test-suite and not run in CI. It starts a headless napari viewer with a synthetic
image + labels layer, creates the annotator widget, activates the labels layer, switches MC mode on
and measures what that costs (time the GUI is blocked, memory, temp files), then how slice changes
and dims.order changes behave. See tools/README.md.

    python tools/bench_minimal_contour.py Z Y X [--cores N] [--rgb] [--orders 1,0,2 0,1,2 2,1,0]
                                         [--start-with-labels] [--json out.json] [--label TEXT]
    python tools/bench_minimal_contour.py --compare before.json after.json

Every probe that is specific to the memmap based FeatureManager (temp folder, done mask, ...) is
optional: it is reported as null when the plugin under test does not have it any more.
"""
import argparse
import json
import os
import sys
import threading
import time
import warnings

SRC_ENV = "ND_ANNOTATOR_SRC"
SETTLE_TIMEOUT = 600.0
DEBOUNCE_WAIT = 0.6  # the plugin delays set_image by 0.5 s after the last dims change


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("shape", nargs="*", type=int, metavar="Z Y X", help="shape of the synthetic 3D volume")
    p.add_argument("--cores", type=int, default=None, help="restrict the process to the first N cores")
    p.add_argument("--rgb", action="store_true", help="use an RGB image (shape + (3,))")
    p.add_argument("--orders", nargs="*", default=None, metavar="A,B,C",
                   help="dims.order permutations to switch through (default: 1,0,2 0,1,2 2,1,0)")
    p.add_argument("--start-with-labels", action="store_true",
                   help="have the labels layer selected when the plugin is created, instead of the image layer")
    p.add_argument("--json", metavar="PATH", help="write the results as JSON")
    p.add_argument("--label", default="", help="free text stored in the JSON (e.g. 'main' or 'PR 3')")
    p.add_argument("--compare", nargs=2, metavar=("BEFORE.json", "AFTER.json"),
                   help="print two result files side by side and exit")
    args = p.parse_args(argv)
    if args.compare:
        return args
    if len(args.shape) != 3:
        p.error("give the volume shape as three integers: Z Y X")
    if args.orders is None:
        args.orders = ["1,0,2", "0,1,2", "2,1,0"]
    try:
        args.orders = [tuple(int(i) for i in o.split(",")) for o in args.orders]
    except ValueError:
        p.error("--orders entries must look like 1,0,2")
    for o in args.orders:
        if sorted(o) != [0, 1, 2]:
            p.error("--orders entry %s is not a permutation of 0,1,2" % (o,))
    return args


# ---------------------------------------------------------------------------------------------
# compare mode (needs no napari)
# ---------------------------------------------------------------------------------------------
def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = prefix + k
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def compare(path_a, path_b):
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)
    for name, d in (("before", a), ("after", b)):
        env = d.get("env", {})
        print("%-7s %s | %s | napari %s | cores %s | shape %s | %s" % (
            name, d.get("label") or "-", env.get("plugin"), env.get("napari"), env.get("cores"), env.get("shape"),
            d.get("args", {}).get("rgb") and "rgb" or "gray"))
    fa, fb = flatten(a.get("metrics", {})), flatten(b.get("metrics", {}))
    print()
    print("%-44s %14s %14s %9s" % ("metric", "before", "after", "after/before"))
    print("-" * 85)
    for key in list(fa) + [k for k in fb if k not in fa]:
        va, vb = fa.get(key), fb.get(key)
        ratio = ""
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (va, vb)) and va:
            ratio = "%.2fx" % (vb / va)
        print("%-44s %14s %14s %9s" % (key, fmt(va), fmt(vb), ratio))


def signed(v):
    return "n/a" if v is None else "%+.0f" % v


def fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return "%.3f" % v if abs(v) < 100 else "%.0f" % v
    if isinstance(v, list):
        return "[" + ",".join(fmt(x) for x in v) + "]"
    return str(v)


# ---------------------------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------------------------
def run(args):
    warnings.simplefilter("ignore")
    if os.environ.get(SRC_ENV):
        # the embedded python of napari_portable ignores PYTHONPATH
        sys.path.insert(0, os.environ[SRC_ENV])
    try:
        import psutil
    except ImportError:
        sys.exit("psutil is required for the benchmark: pip install psutil")
    import numpy as np

    proc = psutil.Process(os.getpid())
    if args.cores:
        proc.cpu_affinity(list(range(args.cores)))

    import napari
    import napari_nd_annotator
    from qtpy.QtWidgets import QApplication

    Z, Y, X = args.shape
    env = {
        "python": sys.version.split()[0],
        "napari": napari.__version__,
        "plugin": os.path.abspath(napari_nd_annotator.__file__),
        "cores": len(proc.cpu_affinity()) if hasattr(proc, "cpu_affinity") else os.cpu_count(),
        "platform": sys.platform,
        "shape": [Z, Y, X] + ([3] if args.rgb else []),
    }
    print("plugin: %s | napari %s | python %s | cores %s" % (env["plugin"], env["napari"], env["python"], env["cores"]))

    def rss():
        return proc.memory_info().rss / 1e6

    peak = [0.0]
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            peak[0] = max(peak[0], rss())
            time.sleep(0.02)

    def pump(n=30):
        for _ in range(n):
            QApplication.processEvents()

    def cpu_seconds():
        t = proc.cpu_times()
        return t.user + t.system

    def disk_alloc_mb(path):
        """Space allocated on disk (differs from the apparent size for sparse files on Windows)."""
        if sys.platform != "win32":
            return os.path.getsize(path) / 1e6
        import ctypes
        hi = ctypes.c_ulong()
        ctypes.windll.kernel32.GetCompressedFileSizeW.restype = ctypes.c_ulong
        lo = ctypes.windll.kernel32.GetCompressedFileSizeW(ctypes.c_wchar_p(path), ctypes.byref(hi))
        return ((hi.value << 32) + lo) / 1e6

    m = {}  # metric name -> value (nested dicts allowed)

    viewer = napari.Viewer(show=False)
    shape = (Z, Y, X, 3) if args.rgb else (Z, Y, X)
    data = (np.random.default_rng(0).random(shape) * 4000).astype(np.uint16)
    img = viewer.add_image(data, name="img", rgb=args.rgb)
    lab = viewer.add_labels(np.zeros((Z, Y, X), np.uint16), name="lab", metadata={"source_image_layer": img})
    viewer.layers.selection.select_only(lab if args.start_with_labels else img)
    pump()
    m["data_mb"] = data.nbytes / 1e6
    base_rss = rss()
    peak[0] = base_rss
    threading.Thread(target=sampler, daemon=True).start()

    from napari_nd_annotator import AnnotatorWidget

    # --- plugin start -----------------------------------------------------------------------
    t = time.perf_counter()
    widget = AnnotatorWidget(viewer, persist=False)
    viewer.window.add_dock_widget(widget, name="annot")
    pump()
    m["start_s"] = time.perf_counter() - t
    m["start_rss_delta_mb"] = rss() - base_rss
    m["data_is_original_after_start"] = img.data is data
    mc = widget.ToolsWidget.minimal_contour_widget

    # optional, FeatureManager specific probes: absent after the memmaps are removed
    def feature_manager():
        return getattr(mc, "feature_manager", None)

    def done_mask():
        extractor = getattr(feature_manager(), "feature_extractor", None)
        return getattr(extractor, "done_mask", None)

    def temp_folder():
        folder = getattr(feature_manager(), "temp_folder", None)
        return folder if folder and os.path.isdir(folder) else None

    def temp_stats():
        folder = temp_folder()
        if folder is None:
            return None
        files = [os.path.join(folder, f) for f in os.listdir(folder)]
        apparent = sum(os.path.getsize(f) for f in files)
        return {
            "files": len(files),
            "apparent_mb": apparent / 1e6,
            "allocated_mb": sum(disk_alloc_mb(f) for f in files),
            "x_image": apparent / data.nbytes,
        }

    # count (and time) _set_image calls; the dims callbacks look the method up on the instance
    set_image_times = []
    if callable(getattr(mc, "_set_image", None)):
        original_set_image = mc._set_image

        def timed_set_image(*a, **k):
            t0 = time.perf_counter()
            try:
                return original_set_image(*a, **k)
            finally:
                set_image_times.append(time.perf_counter() - t0)

        mc._set_image = timed_set_image

    # --- activation: select the labels layer, then switch MC mode on --------------------------
    # later versions only do the heavy work when MC mode is enabled, so both steps count
    cpu0, wall0 = cpu_seconds(), time.perf_counter()
    t = time.perf_counter()
    viewer.layers.selection.select_only(lab)
    pump()
    m["select_s"] = time.perf_counter() - t
    t = time.perf_counter()
    enable = getattr(mc, "_enable_mc_mode", None)
    if callable(enable):
        enable()
        pump()
    m["enable_mc_s"] = time.perf_counter() - t if callable(enable) else None
    m["activation_s"] = m["select_s"] + (m["enable_mc_s"] or 0.0)
    m["activation_rss_delta_mb"] = rss() - base_rss
    m["activation_peak_delta_mb"] = peak[0] - base_rss
    m["data_is_original_after_activation"] = img.data is data
    m["set_image_calls_at_activation"] = len(set_image_times)

    # --- first usable contour ---------------------------------------------------------------
    first_run = None
    try:
        from napari_nd_annotator._widgets.minimal_contour_overlay.vispy_minimal_contour_overlay import \
            VispyMinimalContourOverlay
        calc = VispyMinimalContourOverlay.mc_calculator
        s = min(Y, X)
        lo, hi, p0, p1a, p1b = (int(f * s) for f in (0.39, 0.59, 0.44, 0.49, 0.51))
        t = time.perf_counter()
        calc.set_boundaries(lo, lo, hi, hi)
        calc.run(np.array([[p0, p0], [p1a, p1b], [p0, p0]], dtype=float), True, True, True)
        first_run = time.perf_counter() - t
    except Exception as e:  # noqa: BLE001 - report instead of aborting the whole benchmark
        print("first contour run failed: %r" % e)
    m["first_run_s"] = first_run
    m["first_contour_s"] = m["activation_s"] + first_run if first_run is not None else None

    # --- immediate jump to the farthest slice, right after activation (cold) ------------------
    t = time.perf_counter()
    viewer.dims.set_point(0, Z - 1)
    pump()
    m["cold_jump_s"] = time.perf_counter() - t
    mask = done_mask()
    m["cold_jump_done_fraction"] = float(mask.mean()) if mask is not None and mask.size else None

    # --- background work (only exists with the feature extractor) ------------------------------
    settle = None
    if mask is not None:
        while True:
            pump(5)
            mask = done_mask()
            if mask is None or mask.all():
                break
            if time.perf_counter() - wall0 > SETTLE_TIMEOUT:
                print("TIMEOUT waiting for the features")
                break
            time.sleep(0.05)
        wall = time.perf_counter() - wall0
        cpu = cpu_seconds() - cpu0
        settle = {"wall_s": wall, "cpu_s": cpu, "avg_cores": cpu / max(wall, 1e-6)}
    m["background_features"] = settle
    m["temp"] = temp_stats()
    m["rss_after_settle_delta_mb"] = rss() - base_rss

    # --- warm slice step: the debounce is bypassed by waiting longer than it ---------------------
    lat = []
    for i in range(5):
        time.sleep(DEBOUNCE_WAIT)
        t = time.perf_counter()
        viewer.dims.set_point(0, (Z // 2 + 1 + i) % Z)
        pump()
        lat.append(time.perf_counter() - t)
    m["warm_step_s"] = lat
    m["warm_step_median_s"] = sorted(lat)[len(lat) // 2]
    if set_image_times:
        t = time.perf_counter()
        mc._set_image()
        m["set_image_alone_s"] = time.perf_counter() - t
    else:
        m["set_image_alone_s"] = None

    # --- 20 fast steps: wall time includes the debounce ------------------------------------------
    n0 = len(set_image_times)
    t = time.perf_counter()
    for i in range(20):
        viewer.dims.set_point(0, (Z // 2 + 10 + i) % Z)
        pump(2)
    time.sleep(DEBOUNCE_WAIT + 0.1)
    pump(30)
    m["fast_scroll_20_s"] = time.perf_counter() - t
    m["fast_scroll_set_image_calls"] = len(set_image_times) - n0 if set_image_times else None

    # --- dims.order switching -----------------------------------------------------------------
    orders = {}
    for order in args.orders:
        time.sleep(DEBOUNCE_WAIT)
        before = temp_stats()
        rss_before = rss()
        t = time.perf_counter()
        viewer.dims.order = order
        pump()
        entry = {"gui_blocked_s": time.perf_counter() - t}
        after = temp_stats()
        entry["temp_growth_mb"] = after["apparent_mb"] - before["apparent_mb"] if before and after else None
        entry["temp_total_mb"] = after["apparent_mb"] if after else None
        entry["rss_growth_mb"] = rss() - rss_before
        if done_mask() is not None:
            t = time.perf_counter()
            while done_mask() is not None and not done_mask().all() and time.perf_counter() - t < 120:
                pump(2)
                time.sleep(0.05)
            entry["features_complete_after_s"] = time.perf_counter() - t
        else:
            entry["features_complete_after_s"] = None
        orders[",".join(map(str, order))] = entry
    m["order_switch"] = orders

    m["final_rss_delta_mb"] = rss() - base_rss
    m["peak_rss_delta_mb"] = peak[0] - base_rss
    m["data_is_original_final"] = img.data is data

    stop.set()
    result = {"label": args.label, "args": {k: v for k, v in vars(args).items() if k != "compare"},
              "env": env, "metrics": m}
    print_table(m)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print("\nresults written to %s" % args.json)
    sys.stdout.flush()
    try:
        viewer.close()
    except Exception as e:  # noqa: BLE001 - e.g. PermissionError deleting still-open temp files
        print("closing the viewer failed: %r" % e)
    return result


ROWS = [
    ("start_s", "plugin start", "s"),
    ("start_rss_delta_mb", "  RSS delta", "MB"),
    ("select_s", "select labels layer (GUI blocked)", "s"),
    ("enable_mc_s", "enable MC mode (GUI blocked)", "s"),
    ("activation_s", "activation total", "s"),
    ("activation_rss_delta_mb", "  RSS delta after activation", "MB"),
    ("activation_peak_delta_mb", "  peak RSS delta so far", "MB"),
    ("first_run_s", "first calculator.run", "s"),
    ("first_contour_s", "time to first usable contour", "s"),
    ("cold_jump_s", "jump to far slice after activation", "s"),
    ("background_features.wall_s", "background features: wall", "s"),
    ("background_features.cpu_s", "background features: CPU", "s"),
    ("background_features.avg_cores", "background features: avg cores", ""),
    ("temp.apparent_mb", "temp files (apparent)", "MB"),
    ("temp.allocated_mb", "temp files (allocated)", "MB"),
    ("temp.x_image", "temp files / image size", "x"),
    ("rss_after_settle_delta_mb", "RSS delta after features done", "MB"),
    ("warm_step_median_s", "warm slice step (median of 5)", "s"),
    ("set_image_alone_s", "  _set_image alone", "s"),
    ("fast_scroll_20_s", "20 fast steps (incl. debounce)", "s"),
    ("fast_scroll_set_image_calls", "  _set_image calls", ""),
    ("peak_rss_delta_mb", "peak RSS delta (whole run)", "MB"),
    ("final_rss_delta_mb", "final RSS delta", "MB"),
    ("data_is_original_after_start", "img.data is original: after start", ""),
    ("data_is_original_after_activation", "img.data is original: after activation", ""),
    ("data_is_original_final", "img.data is original: end", ""),
]


def print_table(m):
    flat = flatten(m)
    print()
    print("%-44s %12s %s" % ("metric", "value", "unit"))
    print("-" * 62)
    for key, label, unit in ROWS:
        print("%-44s %12s %s" % (label, fmt(flat.get(key)), unit))
    for order, e in m.get("order_switch", {}).items():
        print("%-44s %12s s   temp %s MB (total %s), RSS %s MB, features complete after %s s" % (
            "dims.order=%s: GUI blocked" % order, fmt(e["gui_blocked_s"]), signed(e["temp_growth_mb"]),
            fmt(e["temp_total_mb"]), signed(e["rss_growth_mb"]), fmt(e["features_complete_after_s"])))


def main(argv=None):
    args = parse_args(argv)
    if args.compare:
        compare(*args.compare)
        return
    run(args)


if __name__ == "__main__":
    main()
