"""Generate a local HTML dashboard for the double OK capture gate."""

from __future__ import annotations

import argparse
import csv
import html
import json
import webbrowser
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from string import Template

DEFAULT_CONFIG_PATH = Path("configs/default.json")
DEFAULT_CSV_PATH = Path("data/processed/hagrid_ok_features.csv")
DEFAULT_MODEL_PATH = Path("models/ok_hand_numpy_logreg.pkl")
DEFAULT_OUTPUT_PATH = Path("reports/gui/index.html")


@dataclass(frozen=True)
class FileSummary:
    path: str
    exists: bool
    size_bytes: int
    size_label: str


@dataclass(frozen=True)
class CsvSummary:
    file: FileSummary
    row_count: int
    positive_count: int
    negative_count: int
    feature_count: int
    column_count: int
    split_counts: dict[str, int]
    gesture_counts: dict[str, int]


@dataclass(frozen=True)
class DashboardSummary:
    config_path: str
    config: dict
    csv: CsvSummary
    model: FileSummary


def human_size(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{size_bytes} B"


def file_summary(path: str | Path) -> FileSummary:
    file_path = Path(path)
    if not file_path.exists():
        return FileSummary(str(file_path), False, 0, "missing")
    size = file_path.stat().st_size
    return FileSummary(str(file_path), True, size, human_size(size))


def scan_feature_csv(path: str | Path) -> CsvSummary:
    summary = file_summary(path)
    if not summary.exists:
        return CsvSummary(summary, 0, 0, 0, 0, 0, {}, {})

    row_count = 0
    positive_count = 0
    negative_count = 0
    split_counts: Counter[str] = Counter()
    gesture_counts: Counter[str] = Counter()
    feature_count = 0
    column_count = 0

    with Path(path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return CsvSummary(summary, 0, 0, 0, 0, 0, {}, {})

        column_count = len(header)
        feature_count = sum(1 for name in header if name.startswith(("lm_", "geom_")))
        column_index = {name: index for index, name in enumerate(header)}
        target_index = column_index.get("target")
        split_index = column_index.get("split")
        gesture_index = column_index.get("gesture_label")

        for row in reader:
            if not row:
                continue
            row_count += 1
            if target_index is not None and target_index < len(row):
                if row[target_index] == "1":
                    positive_count += 1
                elif row[target_index] == "0":
                    negative_count += 1
            if split_index is not None and split_index < len(row):
                split_counts[row[split_index] or "unknown"] += 1
            if gesture_index is not None and gesture_index < len(row):
                gesture_counts[row[gesture_index] or "unknown"] += 1

    return CsvSummary(
        file=summary,
        row_count=row_count,
        positive_count=positive_count,
        negative_count=negative_count,
        feature_count=feature_count,
        column_count=column_count,
        split_counts=dict(split_counts.most_common()),
        gesture_counts=dict(gesture_counts.most_common(12)),
    )


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_dashboard_summary(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    csv_path: str | Path = DEFAULT_CSV_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> DashboardSummary:
    return DashboardSummary(
        config_path=str(config_path),
        config=load_config(config_path),
        csv=scan_feature_csv(csv_path),
        model=file_summary(model_path),
    )


def render_dashboard(summary: DashboardSummary) -> str:
    payload = _safe_json_for_script(asdict(summary))
    config = summary.config
    gate_config = config.get("capture_gate", {})
    total = max(summary.csv.row_count, 1)
    positive_rate = summary.csv.positive_count / total * 100.0
    negative_rate = summary.csv.negative_count / total * 100.0

    cards = [
        ("训练 CSV", "已找到" if summary.csv.file.exists else "缺失", summary.csv.file.size_label),
        ("样本行数", f"{summary.csv.row_count:,}", "每行是一只手"),
        ("OK 正样本", f"{summary.csv.positive_count:,}", f"{positive_rate:.1f}%"),
        ("not OK 负样本", f"{summary.csv.negative_count:,}", f"{negative_rate:.1f}%"),
        ("特征列", f"{summary.csv.feature_count}", f"总列数 {summary.csv.column_count}"),
        ("模型文件", "已找到" if summary.model.exists else "缺失", summary.model.size_label),
    ]
    card_html = "\n".join(
        f"""
        <article class="metric">
          <span>{html.escape(title)}</span>
          <strong>{html.escape(value)}</strong>
          <small>{html.escape(note)}</small>
        </article>
        """
        for title, value, note in cards
    )

    config_rows = [
        ("OK 阈值", config.get("ok_threshold", "未配置")),
        ("稳定窗口", config.get("stable_window", "未配置")),
        ("稳定命中", config.get("stable_min_positive", "未配置")),
        ("最多手数", config.get("max_num_hands", "未配置")),
        ("中心 X", f"{gate_config.get('center_x_min', '?')} - {gate_config.get('center_x_max', '?')}"),
        ("中心 Y", f"{gate_config.get('center_y_min', '?')} - {gate_config.get('center_y_max', '?')}"),
        ("双手距离", gate_config.get("min_hand_separation", "未配置")),
        ("目标手势", "稳定双 OK" if gate_config.get("require_double_ok", True) else "仅几何门控"),
        ("眼镜姿态", "需要" if gate_config.get("require_glasses_pose") else "默认不需要"),
    ]
    config_html = "\n".join(
        f"<tr><th>{html.escape(str(name))}</th><td>{html.escape(str(value))}</td></tr>" for name, value in config_rows
    )

    split_html = _bar_rows(summary.csv.split_counts, summary.csv.row_count)
    gesture_html = _bar_rows(summary.csv.gesture_counts, summary.csv.row_count)

    return Template(_HTML_TEMPLATE).substitute(
        payload=payload,
        cards=card_html,
        config_rows=config_html,
        split_rows=split_html,
        gesture_rows=gesture_html,
        csv_path=html.escape(summary.csv.file.path),
        model_path=html.escape(summary.model.path),
        config_path=html.escape(summary.config_path),
    )


def write_dashboard(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    csv_path: str | Path = DEFAULT_CSV_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> Path:
    summary = collect_dashboard_summary(config_path=config_path, csv_path=csv_path, model_path=model_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_dashboard(summary), encoding="utf-8")
    return output


def _bar_rows(counts: dict[str, int], total: int) -> str:
    if not counts:
        return '<p class="muted">没有可显示的数据</p>'
    safe_total = max(total, 1)
    rows = []
    for name, count in counts.items():
        percent = count / safe_total * 100.0
        rows.append(
            f"""
            <div class="bar-row">
              <div class="bar-label"><span>{html.escape(str(name))}</span><b>{count:,}</b></div>
              <div class="bar-track"><div class="bar-fill" style="width: {percent:.2f}%"></div></div>
            </div>
            """
        )
    return "\n".join(rows)


def _safe_json_for_script(value: object) -> str:
    return (
        json.dumps(value, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a GUI-style HTML dashboard for double OK results.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--csv", default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--open", action="store_true", help="Open the generated HTML in the default browser.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = write_dashboard(
        output_path=args.output,
        config_path=args.config,
        csv_path=args.csv,
        model_path=args.model,
    )
    print(output)
    if args.open:
        webbrowser.open(output.resolve().as_uri())


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Double OK Gesture GUI</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --panel-2: #eef2f6;
      --ink: #1a1f2b;
      --muted: #667085;
      --line: #d7dde6;
      --accent: #177a68;
      --accent-2: #2f6fed;
      --warn: #b76313;
      --bad: #bb2b35;
      --ok: #17845f;
      --shadow: 0 8px 20px rgba(23, 32, 45, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
      letter-spacing: 0;
    }
    .app {
      display: grid;
      grid-template-columns: 260px minmax(0, 1fr);
      min-height: 100vh;
    }
    aside {
      background: #18212d;
      color: #eef4f7;
      padding: 24px 18px;
      border-right: 1px solid #111922;
    }
    aside h1 {
      margin: 0 0 8px;
      font-size: 22px;
      line-height: 1.2;
    }
    aside p {
      margin: 0 0 22px;
      color: #b9c3cf;
      font-size: 13px;
      line-height: 1.6;
    }
    nav {
      display: grid;
      gap: 8px;
    }
    nav a {
      color: #ecf4f7;
      text-decoration: none;
      padding: 10px 12px;
      border-radius: 6px;
      background: rgba(255,255,255,0.06);
      font-size: 14px;
    }
    nav a:hover { background: rgba(255,255,255,0.12); }
    main {
      padding: 22px;
      display: grid;
      gap: 18px;
      align-content: start;
    }
    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    .topbar h2 {
      margin: 0;
      font-size: 24px;
      line-height: 1.2;
    }
    .status-pill {
      min-width: 170px;
      text-align: center;
      padding: 9px 12px;
      border-radius: 999px;
      background: #eaf7f2;
      color: var(--ok);
      font-weight: 700;
      border: 1px solid #b8e1d3;
    }
    .grid {
      display: grid;
      gap: 14px;
    }
    .metrics {
      grid-template-columns: repeat(6, minmax(130px, 1fr));
    }
    .metric, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .metric {
      padding: 14px;
      min-height: 106px;
      display: grid;
      align-content: space-between;
      gap: 8px;
    }
    .metric span, .muted {
      color: var(--muted);
      font-size: 13px;
    }
    .metric strong {
      font-size: 24px;
      line-height: 1.1;
    }
    .metric small {
      color: var(--muted);
      font-size: 12px;
    }
    .two-col {
      grid-template-columns: minmax(0, 1.1fr) minmax(340px, 0.9fr);
      align-items: start;
    }
    .panel {
      padding: 18px;
      overflow: hidden;
    }
    .panel h3 {
      margin: 0 0 14px;
      font-size: 17px;
    }
    .flow {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 10px;
      align-items: stretch;
    }
    .step {
      position: relative;
      background: var(--panel-2);
      border: 1px solid #d4dbe4;
      border-radius: 8px;
      padding: 12px;
      min-height: 95px;
    }
    .step b {
      display: block;
      font-size: 14px;
      margin-bottom: 8px;
    }
    .step span {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    th, td {
      padding: 9px 8px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }
    th {
      width: 110px;
      color: var(--muted);
      font-weight: 600;
    }
    .paths {
      display: grid;
      gap: 8px;
      font-size: 13px;
      color: var(--muted);
      word-break: break-all;
    }
    .sim {
      display: grid;
      grid-template-columns: minmax(260px, 0.75fr) minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }
    .controls {
      display: grid;
      gap: 12px;
    }
    label.control {
      display: grid;
      gap: 6px;
      font-size: 14px;
    }
    input[type="range"] {
      width: 100%;
      accent-color: var(--accent-2);
    }
    input[type="checkbox"] {
      width: 18px;
      height: 18px;
      accent-color: var(--accent);
    }
    .check {
      display: grid;
      grid-template-columns: 28px minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
      padding: 11px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-bottom: 8px;
      background: #fff;
    }
    .dot {
      display: grid;
      place-items: center;
      width: 24px;
      height: 24px;
      border-radius: 999px;
      background: #edf2f7;
      color: var(--muted);
      font-weight: 800;
    }
    .check.pass .dot {
      background: #e8f7ef;
      color: var(--ok);
    }
    .check.fail .dot {
      background: #fff1e2;
      color: var(--warn);
    }
    .check strong {
      font-size: 14px;
    }
    .check small {
      color: var(--muted);
    }
    .result-banner {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 16px;
      border-radius: 8px;
      background: #fff3e0;
      border: 1px solid #ffd09a;
      color: var(--warn);
      margin-bottom: 14px;
      font-weight: 800;
    }
    .result-banner.ready {
      background: #e8f7ef;
      border-color: #a8ddc4;
      color: var(--ok);
    }
    .bar-row {
      display: grid;
      gap: 6px;
      margin-bottom: 11px;
    }
    .bar-label {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-size: 13px;
    }
    .bar-label b { color: var(--muted); }
    .bar-track {
      height: 9px;
      background: #e8edf3;
      border-radius: 999px;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
    }
    .compact-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    @media (max-width: 1180px) {
      .metrics { grid-template-columns: repeat(3, minmax(150px, 1fr)); }
      .flow { grid-template-columns: repeat(3, minmax(120px, 1fr)); }
      .step::after { content: ""; }
      .two-col, .sim { grid-template-columns: 1fr; }
    }
    @media (max-width: 760px) {
      .app { grid-template-columns: 1fr; }
      aside { position: static; }
      main { padding: 14px; }
      .metrics, .compact-grid, .flow { grid-template-columns: 1fr; }
      .topbar { align-items: flex-start; flex-direction: column; }
      .status-pill { width: 100%; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <h1>Double OK GUI</h1>
      <p>训练数据、模型产物、实时门控逻辑集中可视化。</p>
      <nav>
        <a href="#overview">结果概览</a>
        <a href="#flow">识别流程</a>
        <a href="#gate">门控模拟</a>
        <a href="#data">数据分布</a>
      </nav>
    </aside>
    <main>
      <section class="topbar" id="overview">
        <div>
          <h2>当前工程结果</h2>
          <p class="muted">训练用 landmark CSV；运行时用摄像头视频帧转 landmark。</p>
        </div>
        <div class="status-pill" id="projectStatus">LOADED</div>
      </section>

      <section class="grid metrics">
        $cards
      </section>

      <section class="grid two-col">
        <div class="panel" id="flow">
          <h3>运行时识别流程</h3>
          <div class="flow">
            <div class="step"><b>视频帧</b><span>摄像头逐帧输入 BGR 图像。</span></div>
            <div class="step"><b>MediaPipe</b><span>检测最多两只手，输出 21 个点。</span></div>
            <div class="step"><b>特征</b><span>归一化坐标、距离、角度、伸展分数。</span></div>
            <div class="step"><b>单手 OK</b><span>模型或规则分数判断每只手。</span></div>
            <div class="step"><b>双手稳定</b><span>最近窗口内多帧命中才通过。</span></div>
            <div class="step"><b>采集门控</b><span>姿态、入框、居中、分离全部通过。</span></div>
          </div>
        </div>
        <div class="panel">
          <h3>关键配置</h3>
          <table>$config_rows</table>
        </div>
      </section>

      <section class="panel" id="gate">
        <h3>门控模拟器</h3>
        <div class="sim">
          <div class="controls">
            <label class="control">检测到的手数 <span id="handCountText"></span>
              <input id="handCount" type="range" min="0" max="2" step="1" value="2">
            </label>
            <label class="control">双手距离 <span id="separationText"></span>
              <input id="separation" type="range" min="0" max="0.5" step="0.01" value="0.22">
            </label>
            <label class="control">pitch <span id="pitchText"></span>
              <input id="pitch" type="range" min="-45" max="45" step="1" value="0">
            </label>
            <label class="control">roll <span id="rollText"></span>
              <input id="roll" type="range" min="-35" max="35" step="1" value="0">
            </label>
            <label class="control">yaw <span id="yawText"></span>
              <input id="yaw" type="range" min="-60" max="60" step="1" value="0">
            </label>
            <label class="control"><span><input id="poseRequired" type="checkbox"> 要求眼镜姿态</span></label>
            <label class="control"><span><input id="poseAvailable" type="checkbox" checked> 姿态数据可用</span></label>
            <label class="control"><span><input id="visible" type="checkbox" checked> 双手完整入框</span></label>
            <label class="control"><span><input id="centered" type="checkbox" checked> 双手在中心区域</span></label>
            <label class="control"><span><input id="stableOk" type="checkbox" checked> 稳定双手 OK</span></label>
          </div>
          <div>
            <div class="result-banner" id="gateResult"><span>MAKE_DOUBLE_OK</span><small>等待计算</small></div>
            <div id="checkList"></div>
          </div>
        </div>
      </section>

      <section class="grid compact-grid" id="data">
        <div class="panel">
          <h3>Split 分布</h3>
          $split_rows
        </div>
        <div class="panel">
          <h3>手势类别 Top 12</h3>
          $gesture_rows
        </div>
      </section>

      <section class="panel">
        <h3>文件路径</h3>
        <div class="paths">
          <div>Config: $config_path</div>
          <div>CSV: $csv_path</div>
          <div>Model: $model_path</div>
        </div>
      </section>
    </main>
  </div>
  <script>
    const DASHBOARD = $payload;
    const cfg = DASHBOARD.config || {};
    const gateCfg = cfg.capture_gate || {};
    const promptMap = {
      ready: ["READY_TO_CAPTURE", "条件满足，开始采集"],
      glasses_pose_missing: ["WAITING_GLASSES_POSE", "等待眼镜姿态数据"],
      glasses_pose_bad: ["ADJUST_GLASSES_ANGLE", "请调整眼镜角度"],
      need_two_hands: ["SHOW_TWO_HANDS", "请把双手放入相机画面"],
      hands_out_of_frame: ["HANDS_OUT_OF_FRAME", "请把双手完整放入画面"],
      hands_not_centered: ["MOVE_HANDS_TO_CENTER", "请把双手移到画面中心"],
      hands_too_close: ["SEPARATE_HANDS", "请将双手分开一些"],
      need_double_ok: ["MAKE_DOUBLE_OK", "请双手分开并做出 OK 手势"]
    };
    const checks = [
      ["pose", "眼镜姿态", "pose"],
      ["twoHands", "检测两只手", "hand"],
      ["visible", "双手完整入框", "frame"],
      ["centered", "双手在中心区域", "center"],
      ["separated", "双手分开", "distance"],
      ["doubleOk", "稳定双手 OK", "gesture"]
    ];
    const ids = ["handCount", "separation", "pitch", "roll", "yaw", "poseRequired", "poseAvailable", "visible", "centered", "stableOk"];
    ids.forEach((id) => document.getElementById(id).addEventListener("input", update));
    document.getElementById("poseRequired").checked = Boolean(gateCfg.require_glasses_pose);

    function inRange(value, min, max) {
      return value >= min && value <= max;
    }
    function evaluate() {
      const handCount = Number(document.getElementById("handCount").value);
      const separation = Number(document.getElementById("separation").value);
      const pitch = Number(document.getElementById("pitch").value);
      const roll = Number(document.getElementById("roll").value);
      const yaw = Number(document.getElementById("yaw").value);
      const poseRequired = document.getElementById("poseRequired").checked;
      const poseAvailable = document.getElementById("poseAvailable").checked;
      const visible = document.getElementById("visible").checked;
      const centered = document.getElementById("centered").checked;
      const stableOk = document.getElementById("stableOk").checked;
      const poseOk = !poseRequired || (poseAvailable &&
        inRange(pitch, gateCfg.pitch_min ?? -20, gateCfg.pitch_max ?? 20) &&
        inRange(roll, gateCfg.roll_min ?? -12, gateCfg.roll_max ?? 12) &&
        inRange(yaw, gateCfg.yaw_min ?? -25, gateCfg.yaw_max ?? 25));
      const state = {
        pose: poseOk,
        twoHands: handCount >= 2,
        visible,
        centered,
        separated: separation >= (gateCfg.min_hand_separation ?? 0.16),
        doubleOk: stableOk
      };
      let reason = "ready";
      if (poseRequired && !poseAvailable) reason = "glasses_pose_missing";
      else if (!poseOk) reason = "glasses_pose_bad";
      else if (!state.twoHands) reason = "need_two_hands";
      else if (!state.visible) reason = "hands_out_of_frame";
      else if (!state.centered) reason = "hands_not_centered";
      else if (!state.separated) reason = "hands_too_close";
      else if (!state.doubleOk) reason = "need_double_ok";
      return { reason, state, values: { handCount, separation, pitch, roll, yaw } };
    }
    function update() {
      const result = evaluate();
      document.getElementById("handCountText").textContent = `$${result.values.handCount}`;
      document.getElementById("separationText").textContent = `$${result.values.separation.toFixed(2)}`;
      document.getElementById("pitchText").textContent = `$${result.values.pitch}°`;
      document.getElementById("rollText").textContent = `$${result.values.roll}°`;
      document.getElementById("yawText").textContent = `$${result.values.yaw}°`;

      const banner = document.getElementById("gateResult");
      const [label, prompt] = promptMap[result.reason];
      banner.classList.toggle("ready", result.reason === "ready");
      banner.innerHTML = `<span>$${label}</span><small>$${prompt}</small>`;

      const list = document.getElementById("checkList");
      list.innerHTML = checks.map(([key, label, tag]) => {
        const pass = result.state[key];
        return `<div class="check $${pass ? "pass" : "fail"}">
          <div class="dot">$${pass ? "✓" : "!"}</div>
          <strong>$${label}</strong>
          <small>$${tag}</small>
        </div>`;
      }).join("");
    }
    update();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
