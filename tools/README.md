# Developer tools

Scripts here are for developers only. They are not installed with the package and not run in CI.

## `bench_minimal_contour.py`

Measures what the Minimal Contour (MC) widget costs on large data: time the GUI is blocked, memory,
temp files, slice-step latency and `dims.order` switching. It starts a headless napari viewer with a
synthetic uint16 volume (an image layer plus a labels layer that carries
`metadata["source_image_layer"]`), creates `AnnotatorWidget(viewer, persist=False)`, activates the
labels layer, enables MC mode and measures.

Requirements: a Python environment with napari, this plugin and `psutil` (`pip install psutil`).
Run it on a machine that is otherwise idle; the numbers depend on the machine.

### Usage

```
python tools/bench_minimal_contour.py Z Y X [--cores N] [--rgb] [--orders 1,0,2 0,1,2 2,1,0]
                                      [--start-with-labels] [--json out.json] [--label TEXT]
python tools/bench_minimal_contour.py --compare before.json after.json
```

| option | meaning |
|---|---|
| `Z Y X` | shape of the random uint16 volume, e.g. `60 512 512`, `150 1024 1024`, `300 1024 1024` |
| `--cores N` | restrict the process to the first N cores (psutil `cpu_affinity`), to imitate a laptop |
| `--rgb` | use an RGB image of shape `(Z, Y, X, 3)` |
| `--orders` | `dims.order` permutations to switch through at the end (default `1,0,2 0,1,2 2,1,0`) |
| `--object` | select an object in the object list before MC is activated (a box of a quarter of the plane side in the middle of the plane, 11 slices; needs `napari-bbox`): MC then works on the ROI of the object. Adds the metric `select object in the object list` |
| `--start-with-labels` | have the labels layer selected when the plugin is created (instead of the image layer); the measured steps are the same |
| `--json PATH` | also write all numbers as JSON (`null` = not applicable) |
| `--label TEXT` | tag stored in the JSON, e.g. `main` or `PR3` |

Make sure the plugin under test is the one you mean to measure: the first output line prints
`napari_nd_annotator.__file__`. To measure a checkout (for example another worktree) set the
`ND_ANNOTATOR_SRC` environment variable to its `src` folder; it is put first on `sys.path`. This works
for interpreters that ignore `PYTHONPATH` (such as the embedded Python of napari_portable). The compiled
`_eikonal_wrapper*.pyd` files are not tracked by git and must exist in that checkout's
`src/napari_nd_annotator/minimal_contour/`.

```
# Windows, bash
ND_ANNOTATOR_SRC=/path/to/worktree/src python tools/bench_minimal_contour.py 150 1024 1024 --json after.json
# PowerShell
$env:ND_ANNOTATOR_SRC = "C:\path\to\worktree\src"; python tools\bench_minimal_contour.py 150 1024 1024 --json after.json
```

### What is measured (in this order)

| metric | meaning |
|---|---|
| plugin start | creating and docking the widget; RSS delta |
| activation | selecting the labels layer + enabling MC mode, both with the GUI blocked. Versions that do their heavy work only when MC mode is enabled therefore stay comparable with current `main` |
| time to first usable contour | activation + one contour estimate (`VispyMinimalContourOverlay._estimate`, so it also works when the calculator only has a ROI window of the plane) |
| jump to far slice after activation | `dims.set_point(0, Z-1)` directly after activation: how long the GUI is blocked while features of the target slice are not ready |
| background features | wall time, process CPU seconds and average cores until the memmap feature extractor is done (`n/a` while a `FeatureManager` exists and its feature extractor is not available) |
| temp files | apparent/allocated size of the feature temp folder (`*_<pid>_nd_annotator`) and the ratio to the image size; 0 for versions that write no feature files |
| warm slice step | `dims.set_point` + event processing, with the 0.5 s debounce bypassed by waiting 0.6 s first; median of 5 |
| 20 fast steps | 20 consecutive steps without waiting, wall time including the debounce, and how many `_set_image` calls they caused |
| dims.order switches | per order: GUI blocked time, temp growth, RSS growth, time until features are complete |
| peak / final RSS | peak sampled every 20 ms from before the plugin started, relative to the baseline RSS |
| `img.data is original` | whether the plugin replaced the user's array in the image layer (after start / activation / end) |

RSS deltas are relative to the process after the viewer, image and labels were created. Everything
that is specific to the memmap implementation (`feature_manager`, done mask) is probed with
`getattr`, so the script keeps working after those are removed; the corresponding rows then show `n/a`
(`temp files` shows 0).

### Comparing before and after a change

Run the same command on the same machine, with the same `--cores`, for both versions:

```
git checkout main            # or point ND_ANNOTATOR_SRC at a main checkout
python tools/bench_minimal_contour.py 150 1024 1024 --label main --json before.json
git checkout my-branch
python tools/bench_minimal_contour.py 150 1024 1024 --label my-branch --json after.json
python tools/bench_minimal_contour.py --compare before.json after.json
```

`--compare` prints every metric side by side with the after/before ratio. Suggested sizes:
`60 512 512` (quick), `150 1024 1024` and `300 1024 1024` (the reported freezing case; the latter needs
about 8 GB of RAM and 5 GB of temp disk on current `main`), each also with `--cores 2`.

Reference values of current `main` (16 cores, napari 0.9.2) are in the pull request that added this
script. Targets for the planned rework: activation below 0.1 s, RSS delta below 150 MB, no temp files,
warm step below 80 ms, cold jump as fast as a warm step, free `dims.order` switches.
