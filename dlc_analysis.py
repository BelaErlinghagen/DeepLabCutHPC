#!/usr/bin/env python
"""DeepLabCut analysis pipeline for a *single* video, made to run on the HPC.

Usage
-----
    python dlc_analysis.py <config_path> <video_path> [--results-root DIR]

    config_path   Path to the DeepLabCut project's config.yaml (lives at the
                  root of the project folder, which sits inside your workspace).
    video_path    Path to the single video to analyse.

Behaviour
---------
    * If the project has *no* trained model yet, the script trains one
      (``deeplabcut.train_network``) and then exits. Analysis is skipped
      because there is nothing to analyse the video with yet.
    * If a trained model already exists, the script analyses the given video:
        1. deeplabcut.analyze_videos    (save_as_csv=True)
        2. deeplabcut.filterpredictions
        3. deeplabcut.plot_trajectories (filtered)
        4. deeplabcut.create_labeled_video (filtered)
      Every output (h5, csv, plots, labeled video) is written into
      ``<results-root>/Results/<videoname>/`` via each function's
      ``destfolder`` argument, so nothing lands next to the raw videos.

The script processes exactly one video, so a SLURM job array can launch it
once per video and analyse the whole dataset in parallel (see
``run_dlc_array.sh``).
"""

import argparse
import sys
from pathlib import Path

import deeplabcut


def ensure_project_path(config_path: Path) -> None:
    """Make the config's ``project_path`` match where config.yaml actually is.

    A DeepLabCut project carries the absolute ``project_path`` it was created
    with. After copying the project to the cluster that path is wrong (it still
    points at the local machine), so DeepLabCut tries to read/write training and
    model folders at the old location and fails (e.g. PermissionError on
    ``/home/belae/...``). We rewrite it in place to the folder that holds
    config.yaml. ``edit_config`` round-trips with ruamel, so comments and the
    ``video_sets`` mapping are preserved. The write only happens when the value
    differs, so it is idempotent and safe across parallel array tasks.
    """
    project_path = str(config_path.parent)
    cfg = deeplabcut.auxiliaryfunctions.read_plainconfig(str(config_path))
    if cfg.get("project_path") != project_path:
        print(
            f"Updating project_path in config: {cfg.get('project_path')!r} -> "
            f"{project_path!r}",
            flush=True,
        )
        deeplabcut.auxiliaryfunctions.edit_config(
            str(config_path), {"project_path": project_path}
        )


def sanitize_pytorch_configs(project_path: Path) -> None:
    """Make PyTorch model configs readable by DeepLabCut 3.0.0.

    When a training dataset is created with a newer DeepLabCut than the one on
    the cluster, the shuffle's ``pytorch_config.yaml`` carries keys in a format
    the older reader does not tolerate. Two kinds of fix are applied, both of
    which make the file match what native DeepLabCut 3.0.0 would have written:

    1. Present-but-``null`` keys that 3.0.0 reads as dicts. 3.0.0 uses the
       ``some_dict.get("key", {}).get(...)`` idiom, whose ``{}`` default only
       kicks in when the key is *absent*. A present-but-null value returns
       ``None``, so the following ``None.get(...)`` raises AttributeError.
       Newer DeepLabCut writes these keys explicitly as ``null``. Cases seen:
         * ``data.train.top_down_crop`` / ``data.inference.top_down_crop``
           (crashes ``analyze_videos`` for bottom-up models)
         * ``train_settings.weight_init`` and ``detector``
           (crash ``create_labeled_video`` for bottom-up models)
       A native 3.0.0 config simply omits these, so replacing ``null`` with
       ``{}`` reproduces its behaviour. Note: only keys that 3.0.0 dereferences
       as dicts are coerced — keys that are legitimately ``None`` (e.g.
       ``logger``, ``resume_training_from``, disabled augmentations) are left
       untouched.

    2. The top-level ``inference`` block gains extra keys (``snapshot``,
       ``eval``, ``output_dir``). At analysis time 3.0.0 validates that whole
       block against ``InferenceConfig``, which only allows ``multithreading``,
       ``compile``, ``autocast`` and ``conditions`` and raises
       ``KeyError: Invalid key path: ...`` on anything else. 3.0.0 reads those
       extras from function arguments / the project config instead, so dropping
       them from ``pytorch_config.yaml`` is safe.

    The edits are idempotent and only written when something actually changed,
    so this is safe to run on every job and across parallel array tasks.
    """
    from ruamel.yaml import YAML

    # Nested paths where a present-but-null value must become an empty dict.
    null_to_empty_dict_paths = (
        ("data", "train", "top_down_crop"),
        ("data", "inference", "top_down_crop"),
        ("train_settings", "weight_init"),
        ("detector",),
    )
    # Keys DeepLabCut 3.0.0's InferenceConfig accepts in the `inference` block.
    allowed_inference_keys = {"multithreading", "compile", "autocast", "conditions"}

    model_root = project_path / "dlc-models-pytorch"
    if not model_root.is_dir():
        return

    yaml = YAML()  # round-trip loader: preserves comments and layout
    for cfg_file in model_root.rglob("pytorch_config.yaml"):
        with open(cfg_file) as fh:
            cfg = yaml.load(fh)
        if not isinstance(cfg, dict):
            continue

        changed = False

        # (1) null -> {} for keys 3.0.0 dereferences as dicts
        for *parents, leaf in null_to_empty_dict_paths:
            block = cfg
            for step in parents:
                block = block.get(step) if isinstance(block, dict) else None
            if isinstance(block, dict) and leaf in block and block[leaf] is None:
                block[leaf] = {}
                changed = True

        # (2) prune the top-level inference block to 3.0.0's allowed keys
        inference = cfg.get("inference")
        if isinstance(inference, dict):
            for key in [k for k in inference if k not in allowed_inference_keys]:
                del inference[key]
                changed = True

        if changed:
            with open(cfg_file, "w") as fh:
                yaml.dump(cfg, fh)
            print(f"Sanitized model config for DLC 3.0.0: {cfg_file}", flush=True)


def has_trained_model(project_path: Path) -> bool:
    """Return True if the project already contains a trained pose model.

    DeepLabCut stores training snapshots as ``snapshot-*.pt`` for the PyTorch
    engine (inside ``dlc-models-pytorch``) or ``snapshot-*.index`` for the
    TensorFlow engine (inside ``dlc-models``). We just look for any such file
    anywhere under the project, which works regardless of iteration, shuffle
    or training fraction.
    """
    pytorch_snapshots = (project_path / "dlc-models-pytorch").rglob("snapshot-*.pt")
    tensorflow_snapshots = (project_path / "dlc-models").rglob("snapshot-*.index")
    return any(True for _ in pytorch_snapshots) or any(True for _ in tensorflow_snapshots)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepLabCut single-video analysis pipeline for HPC.",
    )
    parser.add_argument(
        "config_path",
        help="Path to the DeepLabCut project's config.yaml",
    )
    parser.add_argument(
        "video_path",
        help="Path to the video to analyse",
    )
    parser.add_argument(
        "--results-root",
        default=None,
        help=(
            "Base directory for results. Defaults to the parent folder of the "
            "DLC project (i.e. your workspace). Outputs go to "
            "<results-root>/Results/<videoname>/."
        ),
    )
    args = parser.parse_args()

    config_path = Path(args.config_path).resolve()
    if not config_path.is_file():
        sys.exit(f"config.yaml not found: {config_path}")

    # config.yaml sits at the root of the DeepLabCut project folder.
    project_path = config_path.parent

    # Fix the stored project_path if the project was moved (e.g. local -> cluster).
    ensure_project_path(config_path)

    # Repair model configs written by a newer DeepLabCut than the cluster's.
    sanitize_pytorch_configs(project_path)

    # ------------------------------------------------------------------
    # No trained model yet -> train and stop. Analysis runs happen later,
    # once a snapshot exists. See run_dlc_array.sh for the two-phase flow.
    # ------------------------------------------------------------------
    if not has_trained_model(project_path):
        print(f"No trained model found under {project_path}", flush=True)
        print("Training the network (deeplabcut.train_network) ...", flush=True)
        deeplabcut.train_network(str(config_path))
        print("Training finished. Re-run to analyse videos.", flush=True)
        return

    # ------------------------------------------------------------------
    # A model exists -> analyse the video and collect every output.
    # ------------------------------------------------------------------
    video_path = Path(args.video_path).resolve()
    if not video_path.is_file():
        sys.exit(f"Video not found: {video_path}")

    results_root = (
        Path(args.results_root).resolve() if args.results_root else project_path.parent
    )
    dest = results_root / "Results" / video_path.stem
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Analysing {video_path.name} -> {dest}", flush=True)

    videos = [str(video_path)]
    config = str(config_path)
    destfolder = str(dest)

    # 1) Pose estimation: writes <video>.h5 and <video>.csv into dest.
    deeplabcut.analyze_videos(
        config,
        videos,
        save_as_csv=True,
        destfolder=destfolder,
    )

    # 2) Filter the raw predictions (median filter by default).
    deeplabcut.filterpredictions(
        config,
        videos,
        destfolder=destfolder,
    )

    # 3) Plot trajectories from the filtered predictions (saves .png plots).
    deeplabcut.plot_trajectories(
        config,
        videos,
        filtered=True,
        destfolder=destfolder,
    )

    # 4) Render the labeled video from the filtered predictions.
    deeplabcut.create_labeled_video(
        config,
        videos,
        filtered=True,
        destfolder=destfolder,
    )

    print(f"Done. All results are in {dest}", flush=True)


if __name__ == "__main__":
    main()
