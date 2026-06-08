import numpy as np
import pytest

from double_ok_gesture.camera import (
    CameraSettings,
    format_video_devices,
    list_video_devices,
    normalize_camera_source,
    open_camera,
)


class FakeCapture:
    def __init__(self, opened, frames):
        self.opened = opened
        self.frames = list(frames)
        self.released = False
        self.values = {}

    def isOpened(self):
        return self.opened and not self.released

    def set(self, key, value):
        self.values[key] = value
        return True

    def get(self, key):
        return self.values.get(key, 0.0)

    def getBackendName(self):
        return "V4L2"

    def read(self):
        if not self.frames:
            return False, None
        return self.frames.pop(0)

    def release(self):
        self.released = True


class FakeCV2:
    CAP_V4L2 = 200
    CAP_PROP_FOURCC = 1
    CAP_PROP_FRAME_WIDTH = 2
    CAP_PROP_FRAME_HEIGHT = 3
    CAP_PROP_FPS = 4
    CAP_PROP_BUFFERSIZE = 5

    def __init__(self, captures):
        self.captures = list(captures)
        self.calls = []

    def VideoCapture(self, source, backend):
        self.calls.append((source, backend))
        return self.captures.pop(0)

    @staticmethod
    def VideoWriter_fourcc(*characters):
        return sum(ord(character) << (8 * index) for index, character in enumerate(characters))


def test_open_camera_retries_and_returns_warmup_frame():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    first = FakeCapture(False, [])
    second = FakeCapture(True, [(True, frame)])
    fake_cv2 = FakeCV2([first, second])

    stream = open_camera(
        CameraSettings(source="0", open_retries=2, retry_delay_sec=0.0, warmup_reads=1),
        cv2_module=fake_cv2,
        sleep=lambda _seconds: None,
    )

    assert fake_cv2.calls == [(0, fake_cv2.CAP_V4L2), (0, fake_cv2.CAP_V4L2)]
    assert first.released
    assert stream.read() is frame
    assert stream.info.width == 1280
    assert stream.info.height == 720
    stream.close()
    assert second.released


def test_camera_stream_raises_after_consecutive_read_failures():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    capture = FakeCapture(True, [(True, frame), (False, None), (False, None)])
    stream = open_camera(
        CameraSettings(open_retries=1, warmup_reads=1, read_failure_limit=2),
        cv2_module=FakeCV2([capture]),
        sleep=lambda _seconds: None,
    )

    assert stream.read() is frame
    assert stream.read() is None
    with pytest.raises(RuntimeError, match="consecutive frames"):
        stream.read()


def test_video_device_listing_reads_linux_sysfs_names(tmp_path):
    for index, name in ((8, "Integrated Camera"), (0, "Orbbec Gemini 335")):
        path = tmp_path / f"video{index}"
        path.mkdir()
        (path / "name").write_text(name, encoding="utf-8")

    devices = list_video_devices(tmp_path)

    assert [device.path for device in devices] == ["/dev/video0", "/dev/video8"]
    assert "Orbbec Gemini 335" in format_video_devices(devices)


def test_camera_settings_and_source_validation():
    assert normalize_camera_source("8") == 8
    assert normalize_camera_source("/dev/video0") == "/dev/video0"
    with pytest.raises(ValueError, match="fourcc"):
        CameraSettings(fourcc="RGB")
