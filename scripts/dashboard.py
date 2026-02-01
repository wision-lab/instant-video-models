import atexit
import logging
import shutil
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp

import streamlit as st
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torchvision.transforms import v2 as tf
from torchvision.transforms.v2.functional import convert_image_dtype
from torchvision.utils import flow_to_image

from instant_video_models.config import (
    configure_loggers,
    instantiate,
    list_config_names,
    load_config,
)
from instant_video_models.hooks import add_hook_modules
from instant_video_models.utils import (
    best_pytorch_device,
    invoke_on_values,
    load_clip,
    save_video,
    set_random_seeds,
    stack_videos,
    visualize_with_cmap,
)

logger = logging.getLogger(__name__)

# Use next(widget_keys) to get a unique widget key if Streamlit gives a
# DuplicateWidgetID exception
widget_keys = (i for i in range(100_000))


@st.cache_data(max_entries=10, show_spinner=False)
def cached_initialize_transforms(transform_configs):
    if len(transform_configs) != 0:
        return tf.Compose(instantiate(transform_configs))
    else:
        return nn.Identity()


@st.cache_data(max_entries=10, show_spinner=False)
def cached_instantiate(object_config):
    return instantiate(object_config)


@st.cache_data(max_entries=5, show_spinner=False)
def cached_load_clip(clip_config):
    return load_clip(clip_config)


def check_for_comparison(filename, comparison_list):
    if st.checkbox("Compare", key=next(widget_keys)):
        comparison_list.append(filename)


def common_prefix_size(keys):
    if len(keys) <= 1:
        return 0
    prefix = keys[0].split(".")
    broken = False
    for key in keys:
        for i, (s_1, s_2) in enumerate(zip(prefix, key.split("."))):
            if s_1 != s_2:
                prefix = prefix[:i]
                broken = True
                break
    if len(prefix) > 0:
        return len(".".join(prefix)) + int(broken)
    else:
        return 0


def fill_skipped_steps(results_partial, skipped_steps, total_steps, fill_value):
    filled = {}
    for key, value in results_partial.items():
        fill_tensor = torch.full_like(value[0], fill_value)
        results = []
        i = 0
        for t in range(total_steps):
            if t in skipped_steps[key]:
                results.append(fill_tensor)
            else:
                results.append(value[i])
                i += 1
        filled[key] = torch.stack(results)
    return filled


# Contains logic that should be re-run if run_config, result_video_config, or
# comparison_video_config changes
@st.cache_data(max_entries=1, show_spinner=False)
def generate_comparison_video(
    run_config, result_video_config, comparison_video_config, tmp_dir, comparison_list
):
    filename = tmp_dir / "comparison.mp4"
    if filename.exists():
        filename.unlink()
    if len(comparison_list) >= 2:
        stack_videos(
            comparison_list,
            str(tmp_dir / "comparison.mp4"),
            comparison_video_config["size"],
            crf=result_video_config["crf"],
            mode=comparison_video_config["stack_mode"],
            preset=result_video_config["preset"],
        )


# Contains logic that should be re-run if run_config changes
@st.cache_data(max_entries=1, show_spinner=False)
def generate_results(run_config, tmp_dir):
    device = best_pytorch_device()

    logger.info("Loading video clip...")
    clip, fps = cached_load_clip(run_config["clip"])

    logger.info("Initializing transforms...")
    transforms = cached_initialize_transforms(run_config["transforms"])
    n_transforms = len(run_config["transforms"])
    if n_transforms > 0:
        logger.info(f"Applying {n_transforms} transforms to the video...")
    clip = transforms(clip)

    logger.info("Initializing model...")
    model = cached_instantiate(run_config["model"]).to(device)
    model.eval()

    logger.info("Initializing visualizer...")
    visualizer = cached_instantiate(run_config["visualizer"])

    if "controllers" in run_config:
        logger.info("Initializing controllers...")
        controller_dict = add_hook_modules(
            model, run_config["controllers"], device=device
        )
    else:
        controller_dict = {}

    logger.info("Initializing stabilizers...")
    stabilizer_dict = add_hook_modules(model, run_config["stabilizers"], device=device)

    logger.info("Initializing monitors...")
    monitor_dict = add_hook_modules(model, run_config["monitors"], device=device)

    if "override_weights_filepath" in run_config:
        override_weights_filepath = run_config["override_weights_filepath"]
        logger.info(f"Loading override weights from {override_weights_filepath}...")

        # Use assign=True in case any components have uninitialized parameters
        model.load_state_dict(torch.load(override_weights_filepath), assign=True)

    results = {"clip": clip, "fps": fps}

    invoke_on_values(controller_dict, "disable")
    invoke_on_values(stabilizer_dict, "disable")
    logger.info("Running unstabilized inference...")
    results["unstabilized"] = process_clip(
        model,
        visualizer,
        clip,
        device,
        stabilizer_dict,
        monitor_dict,
        "Unstabilized inference",
    )
    logger.info("Unstabilized inference complete.")

    if len(stabilizer_dict) > 0:
        invoke_on_values(controller_dict, "enable")
        invoke_on_values(controller_dict, "reset")
        invoke_on_values(stabilizer_dict, "enable")
        invoke_on_values(stabilizer_dict, "reset")
        logger.info("Running stabilized inference...")
        results["stabilized"] = process_clip(
            model,
            visualizer,
            clip,
            device,
            stabilizer_dict,
            monitor_dict,
            "Stabilized inference",
        )
        logger.info("Stabilized inference complete.")
    else:
        logger.info("No stabilizers present, skipping stabilized inference.")

    results["legend"] = visualizer.markdown_legend()
    return results


# Contains logic that should be re-run if run_config or video_config changes
# The leading underscore in _results indicates to Streamlit that it should not hash that
# argument during the cache lookup. By passing run_config, we ensure that this function
# gets re-run whenever the results change.
@st.cache_data(max_entries=1, show_spinner=False)
def generate_result_videos(run_config, result_video_config, tmp_dir, _results):
    # Remove old results
    for item in tmp_dir.iterdir():
        if item.name not in ("session.log", "config.yml"):
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    n_videos = 2 + len(_results["unstabilized"]["history"])
    i = 0
    progress_text = "Generating result videos"
    progress_bar = st.progress(0.0, text=progress_text)

    # Save the input video
    output_fps = _results["fps"] * result_video_config["speed"]
    save_video(
        tmp_dir / "input.mp4",
        _results["clip"],
        output_fps,
        crf=result_video_config["crf"],
        preset=result_video_config["preset"],
    )
    i += 1
    progress_bar.progress(i / n_videos, text=progress_text)

    # Save the unstabilized/stabilized outputs
    full = _results["unstabilized"]["annotated"]
    if "stabilized" in _results:
        full = torch.cat([full, _results["stabilized"]["annotated"]], dim=-2)
    save_video(
        tmp_dir / "output.mp4",
        full,
        output_fps,
        crf=result_video_config["crf"],
        preset=result_video_config["preset"],
    )
    i += 1
    progress_bar.progress(i / n_videos, text=progress_text)

    # Create a legend for output annotations
    if _results["legend"] is not None:
        with open(tmp_dir / "legend.html", "w") as legend_file:
            legend_file.write(_results["legend"])

    # Save the unstabilized/stabilized monitors
    monitors_dir = tmp_dir / "monitors"
    monitors_dir.mkdir()
    for key, unstabilized in _results["unstabilized"]["history"].items():
        full = unstabilized
        if "stabilized" in _results:
            stabilized = _results["stabilized"]["history"][key]
            full = torch.cat([full, stabilized], dim=-2)
        if (full.ndim == 4) and (full.shape[1] == 2):
            visualized = flow_to_image(full)
        else:
            visualized = visualize_with_cmap(full.abs())
        save_video(
            monitors_dir / f"{key}.mp4",
            visualized,
            output_fps,
            crf=result_video_config["crf"],
            preset=result_video_config["preset"],
        )
        i += 1
        progress_bar.progress(i / n_videos, text=progress_text)

    progress_bar.empty()


def get_comparison_video_config():
    config = {}
    config["size"] = st.number_input(
        "Size", min_value=135, max_value=4320, value=1080, step=135
    )
    config["stack_mode"] = st.selectbox("Stack mode", ["hstack", "vstack"])
    return config


def get_output_config():
    config = {}
    config["output_dir"] = st.text_input("Location", value=Path("outputs", "dashboard"))
    config["save_outputs"] = st.button("Save", use_container_width=True)
    return config


def get_result_video_config():
    config = {}
    config["speed"] = st.selectbox(
        "Playback speed", [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0], index=3
    )
    config["rendering"] = st.selectbox("Browser rendering", ["pixelated", "smooth"])
    config["crf"] = st.number_input(
        "CRF (1-51)", min_value=1, max_value=51, value=23, step=1
    )
    config["preset"] = st.selectbox(
        "Encoding preset",
        [
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
        ],
        index=1,
    )
    return config


def get_run_config():
    config = DictConfig({"_original_dir": str(Path(__file__).parents[1])})

    clips_dir = Path("configs", "clips")
    clip_options = list_config_names(clips_dir)
    clip_id = st.selectbox(
        "Clip", clip_options, index=clip_options.index("mountain_driving_1s")
    )
    config = OmegaConf.merge(config, load_config(clips_dir / f"{clip_id}.yml"))

    transforms_dir = Path("configs", "transforms")
    transform_ids = st.multiselect("Transforms", list_config_names(transforms_dir))
    config["transforms"] = []
    for transform_id in transform_ids:
        config["transforms"].extend(
            load_config(transforms_dir / f"{transform_id}.yml")["transforms"]
        )

    model_dir = Path("configs", "models")
    model_options = list_config_names(model_dir)
    model_id = st.selectbox(
        "Model", model_options, index=model_options.index("adain-reservoir")
    )
    model_base = model_id.split("-")[0]
    config = OmegaConf.merge(config, load_config(model_dir / f"{model_id}.yml"))

    stabilizers_dir = Path("configs", "stabilizers", model_base)
    stabilizer_id = st.selectbox(
        "Stabilizers", list_config_names(stabilizers_dir), index=None
    )
    if stabilizer_id is None:
        config["stabilizers"] = []
    else:
        config = OmegaConf.merge(
            config, load_config(stabilizers_dir / f"{stabilizer_id}.yml")
        )

    monitors_dir = Path("configs", "monitors", model_base)
    monitor_id = st.selectbox("Monitors", list_config_names(monitors_dir), index=None)
    if monitor_id is None:
        config["monitors"] = []
    else:
        config = OmegaConf.merge(
            config, load_config(monitors_dir / f"{monitor_id}.yml")
        )

    override_weights_filepath = st.text_input("Override weights", value=None)
    if (override_weights_filepath is not None) and override_weights_filepath != "":
        config["override_weights_filepath"] = override_weights_filepath

    return OmegaConf.to_container(config, resolve=True)


@torch.no_grad
def process_clip(
    model, visualizer, clip, device, stabilizer_dict, monitor_dict, progress_text
):
    annotated = []
    progress_bar = st.progress(0.0, text=progress_text)
    history_partial = {k: [] for k in monitor_dict}
    history_skipped = {k: set() for k in monitor_dict}
    invoke_on_values(monitor_dict, "reset")
    for t, frame in enumerate(clip):
        frame_preprocessed = convert_image_dtype(
            frame.to(device).unsqueeze(dim=0), torch.float32
        )
        pred = model(frame_preprocessed)
        progress_bar.progress((t + 1) / len(clip), text=progress_text)
        annotated.append(visualizer(frame, pred).squeeze(dim=0).cpu())
        for key, monitor in monitor_dict.items():
            if monitor.available():
                history_partial[key].append(monitor.retrieve().cpu())
            else:
                history_skipped[key].add(t)

    # Add zero values to monitors for skipped time steps
    history = fill_skipped_steps(history_partial, history_skipped, clip.shape[0], 0)

    progress_bar.empty()
    return {"annotated": torch.stack(annotated), "history": history}


def main():
    st.set_page_config(layout="wide")
    set_random_seeds(42)
    with st.sidebar:
        with st.expander("Output", expanded=True):
            output_config = get_output_config()
        with st.expander("Run", expanded=True):
            run_config = get_run_config()
        with st.expander("Result videos", expanded=True):
            result_video_config = get_result_video_config()
        with st.expander("Comparison video", expanded=True):
            comparison_video_config = get_comparison_video_config()

    # Once per session, create a unique temporary directory for saving results
    base_path = Path("outputs", "dashboard_tmp")
    base_path.mkdir(parents=True, exist_ok=True)
    if "tmp_dir" not in st.session_state:
        st.session_state["tmp_dir"] = mkdtemp(dir=base_path)
    tmp_dir = Path(st.session_state["tmp_dir"])

    def cleanup_tmp():
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
            logger.info(f"Removed temporary directory {tmp_dir}.")

    # We want the temporary directory to persist between dashboard passes but be cleaned
    # up when Streamlit closes. This atexit handler gets called when the Streamlit
    # server is terminated.
    atexit.register(cleanup_tmp)

    configure_loggers(log_filepath=tmp_dir / "session.log")
    logger.info("Starting new dashboard pass...")

    config_filename = tmp_dir / "config.yml"
    OmegaConf.save(
        {
            "output": output_config,
            "run": run_config,
            "result_videos": result_video_config,
            "comparison_video": comparison_video_config,
        },
        config_filename,
    )
    logger.info(f"Saved configuration to {config_filename}.")

    results = generate_results(run_config, tmp_dir)
    generate_result_videos(run_config, result_video_config, tmp_dir, results)

    # Used to remove common prefixes when displaying keys (better readability)
    prefix_size = common_prefix_size(list(results["unstabilized"]["history"].keys()))

    # Controls in-browser video rendering
    st.markdown(
        f"<style> .stVideo {{ image-rendering: {result_video_config['rendering']}; }} </style>",
        unsafe_allow_html=True,
    )

    # Video filenames that should be stacked together for comparison
    comparison_list = []

    tab_1, tab_2 = st.tabs(["Results", "Comparison"])

    with tab_1:
        col_1, col_2 = st.columns(2)

        # Show the input video and unstabilized/stabilized outputs
        with col_1:
            st.header("Input and Outputs")
            with st.expander("Input"):
                filename = tmp_dir / "input.mp4"
                st.video(str(filename))
                check_for_comparison(filename, comparison_list)
            with st.expander("Outputs", expanded=True):
                filename = tmp_dir / "output.mp4"
                st.video(str(filename))
                check_for_comparison(filename, comparison_list)
                if "stabilized" in results:
                    st.markdown("Top unstabilized, bottom stabilized")
                legend_filename = tmp_dir / "legend.html"
                if legend_filename.exists():
                    with open(legend_filename, "r") as legend_file:
                        legend = legend_file.read()
                    st.markdown(legend, unsafe_allow_html=True)

        # Show the unstabilized/stabilized monitors
        with col_2:
            st.header("Monitors")
            monitors_dir = tmp_dir / "monitors"
            for key in results["unstabilized"]["history"]:
                with st.expander(key[prefix_size:]):
                    filename = monitors_dir / f"{key}.mp4"
                    st.video(str(filename))
                    check_for_comparison(filename, comparison_list)
                    if "stabilized" in results:
                        st.markdown("Top unstabilized, bottom stabilized")

    generate_comparison_video(
        run_config,
        result_video_config,
        comparison_video_config,
        tmp_dir,
        comparison_list,
    )

    # Show the comparison video
    with tab_2:
        filename = tmp_dir / "comparison.mp4"
        if filename.exists():
            st.video(str(filename))
        else:
            st.markdown("Select two or more result videos to compare.")

    # Copy results to a permanent location
    if output_config["save_outputs"]:
        output_dir = Path(
            output_config["output_dir"], datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        )
        logger.info(f"Copying outputs to {output_dir}...")
        shutil.copytree(tmp_dir, output_dir)

    logger.info("Dashboard pass done!")


if __name__ == "__main__":
    main()
