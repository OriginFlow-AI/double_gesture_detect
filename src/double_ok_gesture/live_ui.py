"""Polished OpenCV dashboard for realtime double OK monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .capture_gate import CaptureGateDecision, GateReason
from .recognizer import DoubleOKResult
from .runtime import RuntimeSnapshot

HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)

REASON_TITLES = {
    GateReason.READY: "READY TO CAPTURE",
    GateReason.GLASSES_POSE_MISSING: "WAITING FOR POSE",
    GateReason.GLASSES_POSE_BAD: "ADJUST GLASSES",
    GateReason.NEED_TWO_HANDS: "SHOW BOTH HANDS",
    GateReason.HANDS_OUT_OF_FRAME: "HANDS OUT OF FRAME",
    GateReason.HANDS_NOT_CENTERED: "MOVE TO CENTER",
    GateReason.HANDS_TOO_CLOSE: "SEPARATE HANDS",
    GateReason.NEED_DOUBLE_OK: "MAKE DOUBLE OK",
    GateReason.AVOID_DOUBLE_OK: "AVOID DOUBLE OK",
}

CHECK_LABELS = (
    ("POSE", "glasses_pose_ok"),
    ("VISIBLE", "hands_visible"),
    ("CENTERED", "hands_centered"),
    ("SEPARATED", "hands_separated"),
    ("GESTURE", "gesture_ok"),
)


@dataclass(frozen=True)
class DashboardTheme:
    background: tuple[int, int, int] = (25, 20, 14)
    panel: tuple[int, int, int] = (38, 31, 23)
    panel_alt: tuple[int, int, int] = (47, 39, 29)
    border: tuple[int, int, int] = (74, 63, 50)
    text: tuple[int, int, int] = (242, 239, 233)
    muted: tuple[int, int, int] = (175, 164, 149)
    accent: tuple[int, int, int] = (220, 176, 43)
    success: tuple[int, int, int] = (139, 211, 67)
    warning: tuple[int, int, int] = (91, 181, 244)
    danger: tuple[int, int, int] = (95, 91, 241)


class CachedUnicodeText:
    """Render cached Unicode masks without converting the full frame to Pillow."""

    def __init__(self, font_path: str | Path | None = None) -> None:
        self.font_path = str(font_path or _find_cjk_font() or "")
        self._fonts: dict[int, Any] = {}
        self._masks: dict[tuple[str, int], np.ndarray] = {}

    @property
    def available(self) -> bool:
        return bool(self.font_path)

    def draw(
        self,
        image: np.ndarray,
        text: str,
        origin: tuple[int, int],
        font_size: int,
        color: tuple[int, int, int],
    ) -> None:
        if not self.available or not text:
            return
        mask = self._mask(text, font_size)
        x, y = origin
        h, w = mask.shape
        if x >= image.shape[1] or y >= image.shape[0]:
            return
        x2 = min(x + w, image.shape[1])
        y2 = min(y + h, image.shape[0])
        if x2 <= x or y2 <= y:
            return
        alpha = mask[: y2 - y, : x2 - x].astype(np.float32)[..., None] / 255.0
        target = image[y:y2, x:x2].astype(np.float32)
        fill = np.asarray(color, dtype=np.float32)
        image[y:y2, x:x2] = (target * (1.0 - alpha) + fill * alpha).astype(np.uint8)

    def _mask(self, text: str, font_size: int) -> np.ndarray:
        key = (text, font_size)
        cached = self._masks.get(key)
        if cached is not None:
            return cached

        from PIL import Image, ImageDraw, ImageFont

        font = self._fonts.get(font_size)
        if font is None:
            font = ImageFont.truetype(self.font_path, font_size)
            self._fonts[font_size] = font
        left, top, right, bottom = font.getbbox(text)
        width = max(1, right - left + 4)
        height = max(1, bottom - top + 4)
        image = Image.new("L", (width, height), 0)
        ImageDraw.Draw(image).text((2 - left, 2 - top), text, fill=255, font=font)
        mask = np.asarray(image)
        self._masks[key] = mask
        return mask


class LiveDashboardRenderer:
    def __init__(
        self,
        width: int = 1440,
        height: int = 810,
        *,
        theme: DashboardTheme | None = None,
        font_path: str | Path | None = None,
    ) -> None:
        if width < 960 or height < 600:
            raise ValueError("Dashboard must be at least 960x600")
        self.width = width
        self.height = height
        self.theme = theme or DashboardTheme()
        self.unicode = CachedUnicodeText(font_path)

    def render(
        self,
        camera_frame: np.ndarray,
        result: DoubleOKResult,
        decision: CaptureGateDecision | None,
        snapshot: RuntimeSnapshot,
        *,
        camera_label: str,
        target_fps: float,
        gate_ready: bool | None = None,
        gate_reason: GateReason | None = None,
        gate_prompt: str | None = None,
        model_label: str = "OK hand classifier",
    ) -> np.ndarray:
        import cv2

        if camera_frame.ndim != 3 or camera_frame.shape[2] != 3:
            raise ValueError("camera_frame must be a BGR image")

        canvas = np.full((self.height, self.width, 3), self.theme.background, dtype=np.uint8)
        margin = max(16, self.width // 90)
        header_h = max(72, self.height // 11)
        sidebar_w = max(330, int(self.width * 0.245))
        content_y = header_h + margin
        content_h = self.height - content_y - margin
        camera_x = margin
        camera_w = self.width - sidebar_w - margin * 3
        sidebar_x = camera_x + camera_w + margin

        effective_ready = decision.ready if gate_ready is None and decision else bool(gate_ready)
        effective_reason = gate_reason or (decision.reason if decision else None)
        effective_prompt = gate_prompt or (decision.prompt if decision else "Capture gate is disabled")

        self._draw_header(
            canvas,
            snapshot,
            camera_label,
            target_fps,
            effective_ready,
            effective_reason,
            margin,
            header_h,
        )
        camera_rect = (camera_x, content_y, camera_w, content_h)
        self._draw_camera_panel(
            canvas,
            camera_frame,
            camera_rect,
            effective_ready,
            effective_reason,
            effective_prompt,
        )
        self._draw_sidebar(
            canvas,
            (sidebar_x, content_y, sidebar_w, content_h),
            result,
            decision,
            snapshot,
            camera_label,
            model_label,
            effective_ready,
        )

        cv2.rectangle(
            canvas,
            (0, self.height - 2),
            (self.width, self.height),
            self.theme.accent,
            -1,
        )
        return canvas

    def _draw_header(
        self,
        canvas: np.ndarray,
        snapshot: RuntimeSnapshot,
        camera_label: str,
        target_fps: float,
        ready: bool,
        reason: GateReason | None,
        margin: int,
        header_h: int,
    ) -> None:
        import cv2

        title_y = 35
        self.unicode.draw(canvas, "双手 OK 采集门控", (margin, 12), 29, self.theme.text)
        if not self.unicode.available:
            _text(canvas, "DOUBLE OK CAPTURE GATE", (margin, title_y), 0.82, self.theme.text, 2)
        _text(
            canvas,
            "REALTIME MONITOR  /  ORBBEC VISION",
            (margin, header_h - 13),
            0.42,
            self.theme.muted,
            1,
        )

        status_w = 210
        status_h = 44
        status_x = self.width - margin - status_w
        status_y = max(10, (header_h - status_h) // 2)
        status_color = self.theme.success if ready else self.theme.warning
        _rounded_rect(
            canvas,
            (status_x, status_y, status_w, status_h),
            status_color,
            radius=16,
        )
        title = "READY" if ready else REASON_TITLES.get(reason, "MONITORING")
        _centered_text(canvas, title, (status_x, status_y, status_w, status_h), 0.52, (18, 24, 22), 2)

        chip_w = 132
        chip_gap = 10
        chips = [
            ("FPS", f"{snapshot.fps:4.1f}", snapshot.fps >= target_fps or snapshot.frame_count < 10),
            ("LATENCY", f"{snapshot.processing_ms:4.1f} ms", snapshot.processing_ms <= 100.0),
            ("CAMERA", camera_label.split()[0], True),
        ]
        chip_x = status_x - chip_gap
        for label, value, healthy in reversed(chips):
            chip_x -= chip_w
            _rounded_rect(
                canvas,
                (chip_x, status_y, chip_w, status_h),
                self.theme.panel,
                radius=12,
                border=self.theme.border,
            )
            _text(canvas, label, (chip_x + 12, status_y + 15), 0.32, self.theme.muted, 1)
            value_color = self.theme.text if healthy else self.theme.warning
            _text(canvas, value, (chip_x + 12, status_y + 34), 0.45, value_color, 1)
            chip_x -= chip_gap

        cv2.line(
            canvas,
            (margin, header_h),
            (self.width - margin, header_h),
            self.theme.border,
            1,
        )

    def _draw_camera_panel(
        self,
        canvas: np.ndarray,
        camera_frame: np.ndarray,
        rect: tuple[int, int, int, int],
        ready: bool,
        reason: GateReason | None,
        prompt: str,
    ) -> None:
        import cv2

        x, y, w, h = rect
        _rounded_rect(canvas, rect, self.theme.panel, radius=18, border=self.theme.border)
        inner = (x + 8, y + 8, w - 16, h - 16)
        fitted, image_rect = _cover_image(camera_frame, inner)
        ix, iy, iw, ih = image_rect
        canvas[iy : iy + ih, ix : ix + iw] = fitted

        shade = canvas[iy : iy + ih, ix : ix + iw].copy()
        cv2.rectangle(shade, (0, 0), (iw, 70), (12, 16, 20), -1)
        cv2.addWeighted(shade, 0.40, canvas[iy : iy + ih, ix : ix + iw], 0.60, 0, canvas[iy : iy + ih, ix : ix + iw])
        _text(canvas, "LIVE CAMERA", (ix + 20, iy + 30), 0.55, self.theme.text, 2)
        _text(canvas, "TRACKING  /  TWO HANDS", (ix + 20, iy + 53), 0.36, self.theme.muted, 1)
        cv2.circle(canvas, (ix + iw - 28, iy + 28), 6, self.theme.danger, -1, cv2.LINE_AA)
        _text(canvas, "LIVE", (ix + iw - 78, iy + 34), 0.38, self.theme.text, 1)

        banner_h = 72
        banner_y = iy + ih - banner_h
        banner_color = self.theme.success if ready else self.theme.warning
        overlay = canvas[banner_y : iy + ih, ix : ix + iw].copy()
        cv2.rectangle(overlay, (0, 0), (iw, banner_h), (17, 22, 27), -1)
        cv2.addWeighted(
            overlay,
            0.86,
            canvas[banner_y : iy + ih, ix : ix + iw],
            0.14,
            0,
            canvas[banner_y : iy + ih, ix : ix + iw],
        )
        cv2.rectangle(canvas, (ix, banner_y), (ix + 6, iy + ih), banner_color, -1)
        _text(
            canvas,
            REASON_TITLES.get(reason, "CAPTURE GATE OFF"),
            (ix + 24, banner_y + 29),
            0.58,
            banner_color,
            2,
        )
        if self.unicode.available:
            self.unicode.draw(canvas, prompt, (ix + 24, banner_y + 39), 18, self.theme.text)
        else:
            _text(canvas, prompt, (ix + 24, banner_y + 55), 0.38, self.theme.text, 1)

    def _draw_sidebar(
        self,
        canvas: np.ndarray,
        rect: tuple[int, int, int, int],
        result: DoubleOKResult,
        decision: CaptureGateDecision | None,
        snapshot: RuntimeSnapshot,
        camera_label: str,
        model_label: str,
        ready: bool,
    ) -> None:
        import cv2

        x, y, w, h = rect
        _rounded_rect(canvas, rect, self.theme.panel, radius=18, border=self.theme.border)
        pad = 20
        sx = x + pad
        sw = w - pad * 2
        cursor = y + 28

        _section_title(canvas, "CAPTURE READINESS", (sx, cursor), self.theme)
        cursor += 24
        passed, total = gate_progress(decision)
        percent = int(round(passed / total * 100.0)) if total else 0
        _text(canvas, f"{percent}%", (sx, cursor + 28), 0.78, self.theme.text, 2)
        _progress_bar(
            canvas,
            (sx + 74, cursor + 11, sw - 74, 14),
            percent / 100.0,
            self.theme.success if ready else self.theme.warning,
            self.theme.panel_alt,
        )
        cursor += 54

        for label, attribute in CHECK_LABELS:
            value = bool(getattr(decision, attribute, False)) if decision else False
            self._draw_check_row(canvas, (sx, cursor, sw, 40), label, value)
            cursor += 46

        cursor += 12
        _section_title(canvas, "HAND CONFIDENCE", (sx, cursor), self.theme)
        cursor += 20
        hands = list(result.hands[:2])
        for index in range(2):
            hand = hands[index] if index < len(hands) else None
            self._draw_hand_card(canvas, (sx, cursor, sw, 78), hand, index)
            cursor += 88

        remaining = y + h - cursor - 18
        if remaining >= 115:
            _section_title(canvas, "SYSTEM", (sx, cursor), self.theme)
            cursor += 22
            system_rows = [
                ("Camera", camera_label),
                ("Model", model_label),
                ("Frame", f"#{snapshot.frame_count:,}"),
                ("Controls", "Q / ESC  Exit    S  Snapshot"),
            ]
            for label, value in system_rows:
                _text(canvas, label.upper(), (sx, cursor + 15), 0.31, self.theme.muted, 1)
                _right_text(canvas, value, (sx + sw, cursor + 15), 0.34, self.theme.text, 1)
                cursor += 25

        cv2.line(canvas, (sx, y + h - 42), (sx + sw, y + h - 42), self.theme.border, 1)
        _text(canvas, "DOUBLE OK  /  V0.1", (sx, y + h - 18), 0.32, self.theme.muted, 1)
        _right_text(
            canvas,
            "ONLINE",
            (sx + sw, y + h - 18),
            0.34,
            self.theme.success,
            1,
        )

    def _draw_check_row(
        self,
        canvas: np.ndarray,
        rect: tuple[int, int, int, int],
        label: str,
        passed: bool,
    ) -> None:
        import cv2

        x, y, w, h = rect
        _rounded_rect(canvas, rect, self.theme.panel_alt, radius=10)
        color = self.theme.success if passed else self.theme.warning
        cv2.circle(canvas, (x + 19, y + h // 2), 9, color, -1, cv2.LINE_AA)
        marker = "OK" if passed else "--"
        _centered_text(canvas, marker, (x + 8, y + 9, 22, 22), 0.27, (20, 25, 22), 1)
        _text(canvas, label, (x + 40, y + 25), 0.41, self.theme.text, 1)
        _right_text(canvas, "PASS" if passed else "WAIT", (x + w - 12, y + 25), 0.32, color, 1)

    def _draw_hand_card(
        self,
        canvas: np.ndarray,
        rect: tuple[int, int, int, int],
        hand: Any | None,
        index: int,
    ) -> None:
        x, y, w, h = rect
        _rounded_rect(canvas, rect, self.theme.panel_alt, radius=12)
        if hand is None:
            label = f"HAND {index + 1}"
            score = 0.0
            status = "NOT DETECTED"
            color = self.theme.muted
        else:
            label = hand.handedness.upper()
            score = float(hand.ok_score)
            status = "OK GESTURE" if hand.is_ok else "NOT OK"
            color = self.theme.success if hand.is_ok else self.theme.accent

        _text(canvas, label, (x + 14, y + 24), 0.43, self.theme.text, 1)
        _right_text(canvas, f"{score * 100:5.1f}%", (x + w - 14, y + 24), 0.45, color, 1)
        _progress_bar(
            canvas,
            (x + 14, y + 36, w - 28, 10),
            score,
            color,
            self.theme.background,
        )
        _text(canvas, status, (x + 14, y + 65), 0.32, color, 1)


def draw_hand_tracking(frame_bgr: np.ndarray, result: DoubleOKResult) -> np.ndarray:
    """Draw clean hand skeletons, boxes, and confidence labels."""

    import cv2

    h, w = frame_bgr.shape[:2]
    for hand in result.hands:
        pts = hand.landmarks
        pixels = np.column_stack((pts[:, 0] * w, pts[:, 1] * h)).astype(np.int32)
        color = (139, 211, 67) if hand.is_ok else (220, 176, 43)
        for start, end in HAND_CONNECTIONS:
            cv2.line(
                frame_bgr,
                tuple(pixels[start]),
                tuple(pixels[end]),
                color,
                2,
                cv2.LINE_AA,
            )
        for x, y in pixels:
            cv2.circle(frame_bgr, (int(x), int(y)), 4, (245, 245, 245), -1, cv2.LINE_AA)
            cv2.circle(frame_bgr, (int(x), int(y)), 5, color, 1, cv2.LINE_AA)

        x1 = max(0, int(pixels[:, 0].min()) - 14)
        y1 = max(0, int(pixels[:, 1].min()) - 14)
        x2 = min(w - 1, int(pixels[:, 0].max()) + 14)
        y2 = min(h - 1, int(pixels[:, 1].max()) + 14)
        _corner_box(frame_bgr, (x1, y1, x2, y2), color)

        label = f"{hand.handedness.upper()}  {hand.ok_score * 100:4.1f}%"
        label_w = max(150, cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0] + 24)
        label_y = max(8, y1 - 34)
        _rounded_rect(frame_bgr, (x1, label_y, label_w, 28), (24, 29, 34), radius=8)
        _text(frame_bgr, label, (x1 + 10, label_y + 19), 0.48, color, 1)
    return frame_bgr


def draw_capture_guides(
    frame_bgr: np.ndarray,
    decision: CaptureGateDecision,
    center_bounds: tuple[float, float, float, float],
) -> np.ndarray:
    """Draw restrained center-area guides without covering the camera image."""

    import cv2

    h, w = frame_bgr.shape[:2]
    x_min, x_max, y_min, y_max = center_bounds
    x1, x2 = int(x_min * w), int(x_max * w)
    y1, y2 = int(y_min * h), int(y_max * h)
    color = (139, 211, 67) if decision.ready else (91, 181, 244)
    length = max(20, min(w, h) // 18)
    for start, end in (
        ((x1, y1 + length), (x1, y1)),
        ((x1, y1), (x1 + length, y1)),
        ((x2 - length, y1), (x2, y1)),
        ((x2, y1), (x2, y1 + length)),
        ((x1, y2 - length), (x1, y2)),
        ((x1, y2), (x1 + length, y2)),
        ((x2 - length, y2), (x2, y2)),
        ((x2, y2 - length), (x2, y2)),
    ):
        cv2.line(frame_bgr, start, end, color, 3, cv2.LINE_AA)
    return frame_bgr


def gate_progress(decision: CaptureGateDecision | None) -> tuple[int, int]:
    if decision is None:
        return 0, len(CHECK_LABELS)
    values = [bool(getattr(decision, attribute)) for _label, attribute in CHECK_LABELS]
    return sum(values), len(values)


def _find_cjk_font() -> Path | None:
    candidates = (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    )
    return next((path for path in candidates if path.exists()), None)


def _cover_image(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    import cv2

    x, y, w, h = rect
    source_h, source_w = image.shape[:2]
    scale = max(w / source_w, h / source_h)
    scaled_w = max(1, int(round(source_w * scale)))
    scaled_h = max(1, int(round(source_h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (scaled_w, scaled_h), interpolation=interpolation)
    crop_x = max(0, (scaled_w - w) // 2)
    crop_y = max(0, (scaled_h - h) // 2)
    cropped = resized[crop_y : crop_y + h, crop_x : crop_x + w]
    return cropped, (x, y, w, h)


def _rounded_rect(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int],
    *,
    radius: int,
    border: tuple[int, int, int] | None = None,
) -> None:
    import cv2

    x, y, w, h = rect
    radius = max(1, min(radius, w // 2, h // 2))
    cv2.rectangle(image, (x + radius, y), (x + w - radius, y + h), color, -1)
    cv2.rectangle(image, (x, y + radius), (x + w, y + h - radius), color, -1)
    for center in (
        (x + radius, y + radius),
        (x + w - radius, y + radius),
        (x + radius, y + h - radius),
        (x + w - radius, y + h - radius),
    ):
        cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)
    if border:
        cv2.line(image, (x + radius, y), (x + w - radius, y), border, 1, cv2.LINE_AA)
        cv2.line(image, (x + radius, y + h), (x + w - radius, y + h), border, 1, cv2.LINE_AA)
        cv2.line(image, (x, y + radius), (x, y + h - radius), border, 1, cv2.LINE_AA)
        cv2.line(image, (x + w, y + radius), (x + w, y + h - radius), border, 1, cv2.LINE_AA)


def _corner_box(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int],
) -> None:
    import cv2

    x1, y1, x2, y2 = rect
    length = max(12, min(x2 - x1, y2 - y1) // 4)
    for start, end in (
        ((x1, y1 + length), (x1, y1)),
        ((x1, y1), (x1 + length, y1)),
        ((x2 - length, y1), (x2, y1)),
        ((x2, y1), (x2, y1 + length)),
        ((x1, y2 - length), (x1, y2)),
        ((x1, y2), (x1 + length, y2)),
        ((x2 - length, y2), (x2, y2)),
        ((x2, y2 - length), (x2, y2)),
    ):
        cv2.line(image, start, end, color, 3, cv2.LINE_AA)


def _progress_bar(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    progress: float,
    color: tuple[int, int, int],
    track: tuple[int, int, int],
) -> None:
    x, y, w, h = rect
    progress = float(np.clip(progress, 0.0, 1.0))
    _rounded_rect(image, rect, track, radius=max(2, h // 2))
    fill_w = int(round(w * progress))
    if fill_w > 0:
        _rounded_rect(image, (x, y, max(h, fill_w), h), color, radius=max(2, h // 2))


def _section_title(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    theme: DashboardTheme,
) -> None:
    import cv2

    x, y = origin
    cv2.rectangle(image, (x, y - 13), (x + 4, y + 2), theme.accent, -1)
    _text(image, text, (x + 13, y), 0.43, theme.muted, 1)


def _text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    import cv2

    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _right_text(
    image: np.ndarray,
    text: str,
    anchor: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    import cv2

    width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
    _text(image, text, (anchor[0] - width, anchor[1]), scale, color, thickness)


def _centered_text(
    image: np.ndarray,
    text: str,
    rect: tuple[int, int, int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    import cv2

    x, y, w, h = rect
    (text_w, text_h), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness,
    )
    origin = (x + (w - text_w) // 2, y + (h + text_h - baseline) // 2)
    _text(image, text, origin, scale, color, thickness)
