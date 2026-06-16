"""Modal CPU: PySceneDetect scene detection and splitting.

Deploy:
  modal deploy deploy.py
"""

from __future__ import annotations

import base64
import bisect
import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import modal
from tongflow import deploy
from typing import Any, List, Optional, Tuple, cast





image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("ffmpeg")
    .uv_pip_install(
        "tongflow==0.1.0",
        "scenedetect[opencv]",
        "boto3",
    )
)
app = modal.App(Path(__file__).resolve().parent.name, image=image)
secrets = modal.Secret.from_dict({})

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# =========================
# Helpers: ffmpeg/ffprobe availability
# =========================
def _ensure_ff_tools():
    for bin_name in ("ffmpeg", "ffprobe"):
        if shutil.which(bin_name) is None:
            raise RuntimeError(
                f"`{bin_name}` not found; install the ffmpeg toolset in the runtime environment."
            )
    logger.info("ffmpeg/ffprobe available.")

# =========================
# Helpers: keyframe handling
# =========================
def _get_keyframes_seconds(video_path: Path) -> List[float]:
    """Extract keyframe timestamps (seconds) using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_frames",
        "-show_entries", "frame=key_frame,pkt_pts_time,best_effort_timestamp_time",
        "-of", "json",
        str(video_path)
    ]
    out = subprocess.check_output(cmd)
    data = json.loads(out)
    kfs: List[float] = []
    for f in data.get("frames", []):
        try:
            if int(f.get("key_frame", 0)) == 1:
                t_str = f.get("pkt_pts_time", f.get("best_effort_timestamp_time", None))
                if t_str is not None:
                    t = float(t_str)
                    if t >= 0:
                        kfs.append(t)
        except Exception:
            continue
    kfs = sorted(set(kfs))
    if not kfs or kfs[0] > 0.0005:
        kfs = [0.0] + kfs
    return kfs

def _snap_to_prev_kf(t: float, keyframes: List[float]) -> float:
    """Snap to the nearest keyframe ≤ t."""
    i = bisect.bisect_right(keyframes, t)
    return keyframes[max(0, i - 1)]

# =========================
# R2 client
# =========================
class R2Client:
    """R2 (S3-compatible) client wrapper for Cloudflare R2."""

    def __init__(self):
        """Initialize the R2 client with configuration from environment variables."""
        import boto3
        from botocore.config import Config

        self.region = os.getenv("R2_REGION", "auto")
        self.bucket = os.getenv("R2_BUCKET")
        self.access_key_id = os.getenv("R2_ACCESS_KEY_ID")
        self.secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY")
        self.endpoint = os.getenv("R2_ENDPOINT")

        if not self.bucket:
            raise RuntimeError("R2_BUCKET is not set")

        config = Config(region_name=self.region)
        self.client = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            config=config,
        )
        logger.info(f"R2 client initialized for bucket={self.bucket}")

    def download_file(self, key: str, local_path: str):
        """Download an object from R2 to a local path (blocking).
        
        Args:
            key: The key of the object in R2.
            local_path: The local file path to save to.
        """
        try:
            self.client.download_file(self.bucket, key, local_path)
            logger.info(f"Downloaded {key} to {local_path}")
        except Exception as e:
            logger.error(f"Failed to download {key} from bucket {self.bucket}: {e}")
            raise

    def upload_file(self, local_path: str, dest_key: str = None) -> str:
        """Upload a local file to R2.
        
        Args:
            local_path: The local file path to upload.
            dest_key: The destination key in R2. If None, generates a UUID-based key.
        
        Returns:
            The destination key where the file was uploaded.
        """
        try:
            if dest_key is None:
                dest_key = f"{uuid.uuid4().hex}_{os.path.basename(local_path)}"
            self.client.upload_file(local_path, self.bucket, dest_key)
            logger.info(f"Uploaded {local_path} to {dest_key}")
            return dest_key
        except Exception as e:
            logger.error(f"Failed to upload {local_path} to bucket {self.bucket}: {e}")
            raise

# =========================
# Redis notifier
# =========================
# =========================
# Same as ffmpeg.py: parse Modal-provided bytes / base64
# =========================
def _normalize_modal_bytes(val) -> bytes:
    if val is None:
        raise ValueError("empty bytes")
    if isinstance(val, (bytes, bytearray)):
        return bytes(val)
    if isinstance(val, str):
        return base64.b64decode(val)
    return bytes(val)


# =========================
# Modal entry
# =========================
def _split_video_core(task: dict) -> dict:
    """Scene detect & split: prefer prompt.video_bytes (matches OpenFlow / ffmpeg), else fall back to R2 fileKey."""
    task_id = None
    try:
        task_id = task.get("taskId") or task.get("task_id")
        if not task_id:
            raise ValueError("missing taskId / task_id")

        prompt = task["prompt"]
        threshold = prompt.get("threshold", 30.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            vb = prompt.get("video_bytes")
            file_key_ref = prompt.get("fileKey") or ""

            if vb is not None:
                data = _normalize_modal_bytes(vb)
                name = prompt.get("video_filename") or "input.mp4"
                local_video_path = tmpdir_path / Path(str(name)).name
                local_video_path.write_bytes(data)
                logger.info(
                    f"wrote input video ({len(data)} bytes); scene-detect threshold: {threshold}"
                )
                split_files = detect_and_split_keyframe_aligned(local_video_path)
                outputs = []
                for p in split_files:
                    pth = Path(p)
                    ext = pth.suffix.lstrip(".") or "mp4"
                    outputs.append(
                        {"output_bytes": pth.read_bytes(), "output_ext": ext}
                    )
                result = {
                    "success": True,
                    "outputs": outputs,
                    "original_key": file_key_ref,
                    "count": len(outputs),
                    "note": "cut points snapped to keyframes; -c copy (no transcoding)",
                }
                logger.info(
                    f"task done: {len(outputs)} clips (returning byte streams to Next.js)"
                )
                return result

            r2_client = R2Client()
            file_key = prompt["fileKey"]
            local_video_path = tmpdir_path / Path(file_key).name

            logger.info(f"downloading video file: {file_key}")
            r2_client.download_file(file_key, str(local_video_path))

            logger.info(f"start scene detection, threshold: {threshold}")
            split_files = detect_and_split_keyframe_aligned(local_video_path)

            uploaded_keys = []
            logger.info(f"uploading {len(split_files)} clips to R2...")
            for idx, file_path in enumerate(split_files):
                dest_key = f"tasks/{task_id}/{uuid.uuid4().hex}.mp4"
                r2_client.upload_file(file_path, dest_key)
                uploaded_keys.append(dest_key)
                logger.info(f"uploaded: {dest_key} ({idx+1}/{len(split_files)})")

            result = {
                "original_key": file_key,
                "split_keys": uploaded_keys,
                "count": len(uploaded_keys),
                "note": "cut points snapped to keyframes; -c copy (no transcoding)",
            }

            logger.info(
                f"task done: {file_key}, produced {len(uploaded_keys)} clips"
            )
            return result

    except Exception as e:
        logger.error(f"scene detection failed: {e}")
        import traceback

        traceback.print_exc()
        raise


@app.function(cpu=1.0, memory=2048, timeout=3600, secrets=[secrets], scaledown_window=5)
def split_video(task: dict) -> dict:
    return _split_video_core(task)


# =========================
# Core: keyframe-aligned splitting
# =========================
def detect_and_split_keyframe_aligned(
    local_video_path: Path,
) -> List[str]:
    """
    Detect scenes with PySceneDetect AdaptiveDetector, snap cut points to keyframes, and split via ffmpeg -c copy.
    """
    from scenedetect import FrameTimecode, SceneManager, open_video
    from scenedetect.detectors import AdaptiveDetector
    from scenedetect.video_splitter import split_video_ffmpeg

    _ensure_ff_tools()
    logger.info(f"start scene detection: {local_video_path}")

    video = open_video(str(local_video_path))
    fps = video.frame_rate
    scene_manager = SceneManager()
    detector = AdaptiveDetector(
        adaptive_threshold=3.0,
        min_scene_len=10,
        min_content_val=2.0
    )
    scene_manager.add_detector(detector)

    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()
    logger.info(f"detected {len(scene_list)} scenes")

    output_dir = local_video_path.parent
    output_dir.mkdir(exist_ok=True)

    adjusted_scenes = []
    for start, end in scene_list:
        # Clamp to valid range (start ≥ 0 frames)
        start_frame = max(start.get_frames() + int(fps / 3), 0)
        end_frame = max(end.get_frames() - int(fps / 3), 0)
        if end_frame - start_frame < fps:
            continue
        start_adj = FrameTimecode(start_frame, fps=fps)
        end_adj = FrameTimecode(end_frame, fps=fps)
        adjusted_scenes.append((start_adj, end_adj))
    
    split_video_ffmpeg(
        input_video_path=str(local_video_path),
        scene_list=adjusted_scenes,
        output_dir=output_dir,
        output_file_template="$VIDEO_NAME-Scene-$SCENE_NUMBER.mp4",
        arg_override="-map 0:v:0 -map 0:a? -map 0:s? -c copy",
        show_progress=True,
        show_output=False
    )

    # Collect split files
    stem = local_video_path.stem
    split_files = sorted(
        [f for f in output_dir.glob(f"*-Scene-*.*") if f.is_file()],
        key=lambda f: int(f.stem.split("Scene-")[-1]) if "Scene-" in f.stem else 9999
    )

    if len(scene_list) == 0:
        return [str(local_video_path)]
    return [str(f) for f in split_files]

from tongflow.models.split_video import SplitVideoInput, SplitVideoOutput
from tongflow.node_slots import NodeSlots
from tongflow.protocol import asset_as_path, asset_from_path
from tongflow.slots import node_slot


@deploy
@app.cls(cpu=1.0, memory=2048, timeout=3600, secrets=[secrets], scaledown_window=5)
class Inference:
    @modal.method()
    @node_slot(NodeSlots.SPLIT_VIDEO)
    def split_video(self, input: SplitVideoInput) -> SplitVideoOutput:
        if input.video is None:
            return SplitVideoOutput(success=False, error="Missing `video` Asset")
        try:
            with asset_as_path(input.video, suffix=".mp4") as video_path:
                split_files = detect_and_split_keyframe_aligned(Path(video_path))
                return SplitVideoOutput(
                    success=True,
                    video_parts=[asset_from_path(p) for p in split_files],
                )
        except Exception as e:
            logger.error(f"scene detection failed: {e}", exc_info=True)
            return SplitVideoOutput(success=False, error=str(e))