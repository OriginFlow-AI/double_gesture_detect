"""Reliable OpenCV camera access and Linux video-device diagnostics."""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)
CameraSource = int | str


@dataclass(frozen=True)
class VideoDevice:
    path: str
    name: str


@dataclass(frozen=True)
class CameraSettings:
    source: CameraSource = "/dev/video0"
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    fourcc: str = "MJPG"
    open_retries: int = 5
    retry_delay_sec: float = 0.5
    warmup_reads: int = 5
    read_failure_limit: int = 5

    def __post_init__(self) -> None:
        if isinstance(self.source, str) and not self.source.strip():
            raise ValueError("camera source must not be empty")
        if self.width < 1 or self.height < 1:
            raise ValueError("camera width and height must be positive")
        if not math.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError("camera fps must be finite and positive")
        if len(self.fourcc) != 4:
            raise ValueError("camera fourcc must contain exactly four characters")
        if self.open_retries < 1:
            raise ValueError("camera open_retries must be at least 1")
        if not math.isfinite(self.retry_delay_sec) or self.retry_delay_sec < 0.0:
            raise ValueError("camera retry_delay_sec must be finite and non-negative")
        if self.warmup_reads < 1:
            raise ValueError("camera warmup_reads must be at least 1")
        if self.read_failure_limit < 1:
            raise ValueError("camera read_failure_limit must be at least 1")


@dataclass(frozen=True)
class CameraInfo:
    source: CameraSource
    backend: str
    width: int
    height: int
    fps: float
    fourcc: str


@dataclass
class CameraStream:
    """Opened camera with transient read-failure handling."""

    capture: Any
    settings: CameraSettings
    info: CameraInfo
    _pending_frame: Any | None = field(default=None, repr=False)
    _consecutive_failures: int = field(default=0, init=False, repr=False)

    def read(self) -> Any | None:
        if self._pending_frame is not None:
            frame = self._pending_frame
            self._pending_frame = None
            return frame

        ok, frame = self.capture.read()
        if ok and frame is not None:
            self._consecutive_failures = 0
            return frame

        self._consecutive_failures += 1
        if self._consecutive_failures >= self.settings.read_failure_limit:
            raise RuntimeError(
                f"Camera {format_camera_source(self.settings.source)} failed to read "
                f"{self._consecutive_failures} consecutive frames"
            )
        LOGGER.warning(
            "Transient camera read failure (%d/%d) from %s",
            self._consecutive_failures,
            self.settings.read_failure_limit,
            format_camera_source(self.settings.source),
        )
        return None

    def close(self) -> None:
        self.capture.release()

    def __enter__(self) -> "CameraStream":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()


def normalize_camera_source(source: CameraSource) -> CameraSource:
    if isinstance(source, int):
        return source
    value = source.strip()
    if value.isdecimal():
        return int(value)
    return value


def format_camera_source(source: CameraSource) -> str:
    normalized = normalize_camera_source(source)
    return str(normalized) if isinstance(normalized, str) else f"index {normalized}"


def list_video_devices(sys_class_path: str | Path = "/sys/class/video4linux") -> list[VideoDevice]:
    root = Path(sys_class_path)
    if not root.is_dir():
        return []

    devices = []
    for device_path in sorted(root.glob("video*"), key=_video_device_sort_key):
        try:
            name = (device_path / "name").read_text(encoding="utf-8").strip()
        except OSError:
            name = "unknown"
        devices.append(VideoDevice(path=f"/dev/{device_path.name}", name=name))
    return devices


def format_video_devices(devices: list[VideoDevice] | None = None) -> str:
    available = list_video_devices() if devices is None else devices
    if not available:
        return "No /dev/video* devices found"
    return "\n".join(f"{device.path}: {device.name}" for device in available)


def open_camera(
    settings: CameraSettings,
    *,
    cv2_module: Any | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> CameraStream:
    """Open and warm up a camera, retrying devices that are still initializing."""

    if cv2_module is None:
        import cv2 as cv2_module

    source = normalize_camera_source(settings.source)
    last_error = "camera did not open"
    for attempt in range(1, settings.open_retries + 1):
        capture = _create_capture(cv2_module, source)
        keep_open = False
        try:
            if not capture.isOpened():
                last_error = "backend could not open the device"
            else:
                _configure_capture(cv2_module, capture, settings)
                frame = _read_warmup_frame(capture, settings.warmup_reads, sleep)
                if frame is None:
                    last_error = "device opened but did not return a frame"
                else:
                    info = _camera_info(cv2_module, capture, settings, source)
                    LOGGER.info(
                        "Opened camera %s via %s at %dx%d %.1f FPS (%s)",
                        format_camera_source(source),
                        info.backend,
                        info.width,
                        info.height,
                        info.fps,
                        info.fourcc,
                    )
                    keep_open = True
                    return CameraStream(
                        capture=capture,
                        settings=settings,
                        info=info,
                        _pending_frame=frame,
                    )
        finally:
            if not keep_open:
                capture.release()

        if attempt < settings.open_retries:
            LOGGER.warning(
                "Camera %s open attempt %d/%d failed: %s",
                format_camera_source(source),
                attempt,
                settings.open_retries,
                last_error,
            )
            sleep(settings.retry_delay_sec)

    devices = format_video_devices()
    raise RuntimeError(
        f"Cannot open camera {format_camera_source(source)} after "
        f"{settings.open_retries} attempts: {last_error}\nAvailable devices:\n{devices}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="List and probe OpenCV camera devices.")
    parser.add_argument("--camera", default="/dev/video0", help="Camera index or device path.")
    parser.add_argument("--list", action="store_true", help="List Linux video devices.")
    parser.add_argument("--probe", action="store_true", help="Open the selected camera and read one frame.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--fourcc", default="MJPG")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.list or not args.probe:
        print(format_video_devices())
    if args.probe:
        settings = CameraSettings(
            source=args.camera,
            width=args.width,
            height=args.height,
            fps=args.fps,
            fourcc=args.fourcc,
        )
        with open_camera(settings) as stream:
            frame = stream.read()
            if frame is None:
                raise RuntimeError("Camera probe did not receive a frame")
            print(
                f"OK {format_camera_source(stream.info.source)} "
                f"{stream.info.width}x{stream.info.height} "
                f"{stream.info.fps:.1f}FPS {stream.info.fourcc}"
            )


def _create_capture(cv2_module: Any, source: CameraSource) -> Any:
    if sys.platform.startswith("linux") and hasattr(cv2_module, "CAP_V4L2"):
        return cv2_module.VideoCapture(source, cv2_module.CAP_V4L2)
    return cv2_module.VideoCapture(source)


def _configure_capture(cv2_module: Any, capture: Any, settings: CameraSettings) -> None:
    capture.set(
        cv2_module.CAP_PROP_FOURCC,
        cv2_module.VideoWriter_fourcc(*settings.fourcc),
    )
    capture.set(cv2_module.CAP_PROP_FRAME_WIDTH, settings.width)
    capture.set(cv2_module.CAP_PROP_FRAME_HEIGHT, settings.height)
    capture.set(cv2_module.CAP_PROP_FPS, settings.fps)
    if hasattr(cv2_module, "CAP_PROP_BUFFERSIZE"):
        capture.set(cv2_module.CAP_PROP_BUFFERSIZE, 1)


def _read_warmup_frame(
    capture: Any,
    warmup_reads: int,
    sleep: Callable[[float], None],
) -> Any | None:
    for _ in range(warmup_reads):
        ok, frame = capture.read()
        if ok and frame is not None:
            return frame
        sleep(0.05)
    return None


def _camera_info(
    cv2_module: Any,
    capture: Any,
    settings: CameraSettings,
    source: CameraSource,
) -> CameraInfo:
    try:
        backend = capture.getBackendName()
    except Exception:
        backend = "unknown"
    width = int(round(capture.get(cv2_module.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2_module.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2_module.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0.0:
        fps = settings.fps
    fourcc_value = int(capture.get(cv2_module.CAP_PROP_FOURCC))
    fourcc = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4))
    fourcc = fourcc.strip("\x00 ") or settings.fourcc
    return CameraInfo(
        source=source,
        backend=backend,
        width=width,
        height=height,
        fps=fps,
        fourcc=fourcc,
    )


def _video_device_sort_key(path: Path) -> tuple[int, str]:
    suffix = path.name.removeprefix("video")
    return (int(suffix), path.name) if suffix.isdecimal() else (sys.maxsize, path.name)


if __name__ == "__main__":
    main()
