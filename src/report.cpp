#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "double_ok_gesture/report.hpp"

namespace double_ok_gesture {

std::string human_size(std::uintmax_t bytes) {
    double value = static_cast<double>(bytes);
    for (const std::string unit : {"B", "KB", "MB", "GB", "TB"}) {
        if (value < 1024.0 || unit == "TB") {
            std::ostringstream out;
            if (unit == "B") {
                out << bytes << " B";
            } else {
                out << std::fixed << std::setprecision(1) << value << ' ' << unit;
            }
            return out.str();
        }
        value /= 1024.0;
    }
    return std::to_string(bytes) + " B";
}

FileSummary summarize_file(const std::filesystem::path& path) {
    FileSummary summary;
    summary.path = path;
    summary.exists = std::filesystem::exists(path);
    if (summary.exists) {
        summary.size_bytes = std::filesystem::file_size(path);
        summary.size_label = human_size(summary.size_bytes);
    }
    return summary;
}

std::vector<std::string> parse_csv_row(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (ch == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
                field.push_back('"');
                ++i;
            } else {
                quoted = !quoted;
            }
        } else if (ch == ',' && !quoted) {
            fields.push_back(field);
            field.clear();
        } else {
            field.push_back(ch);
        }
    }
    fields.push_back(field);
    return fields;
}

std::optional<std::size_t> column_index(const std::vector<std::string>& header, const std::string& name) {
    const auto it = std::find(header.begin(), header.end(), name);
    if (it == header.end()) {
        return std::nullopt;
    }
    return static_cast<std::size_t>(std::distance(header.begin(), it));
}

CsvSummary scan_feature_csv(const std::filesystem::path& path) {
    CsvSummary summary;
    summary.file = summarize_file(path);
    if (!summary.file.exists) {
        return summary;
    }

    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open feature CSV: " + path.string());
    }
    std::string line;
    if (!std::getline(in, line)) {
        return summary;
    }

    const auto header = parse_csv_row(line);
    summary.column_count = header.size();
    summary.feature_count = static_cast<std::size_t>(std::count_if(header.begin(), header.end(), [](const std::string& name) {
        return name.rfind("lm_", 0) == 0 || name.rfind("geom_", 0) == 0;
    }));
    const auto target_index = column_index(header, "target");
    const auto split_index = column_index(header, "split");
    const auto gesture_index = column_index(header, "gesture_label");

    while (std::getline(in, line)) {
        const auto row = parse_csv_row(line);
        if (row.empty() || std::all_of(row.begin(), row.end(), [](const std::string& value) { return value.empty(); })) {
            continue;
        }
        ++summary.row_count;
        if (target_index && *target_index < row.size()) {
            if (row[*target_index] == "1") {
                ++summary.positive_count;
            } else if (row[*target_index] == "0") {
                ++summary.negative_count;
            }
        }
        if (split_index && *split_index < row.size()) {
            ++summary.split_counts[row[*split_index].empty() ? "unknown" : row[*split_index]];
        }
        if (gesture_index && *gesture_index < row.size()) {
            ++summary.gesture_counts[row[*gesture_index].empty() ? "unknown" : row[*gesture_index]];
        }
    }
    return summary;
}

std::string html_escape(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (const char ch : value) {
        switch (ch) {
            case '&':
                out += "&amp;";
                break;
            case '<':
                out += "&lt;";
                break;
            case '>':
                out += "&gt;";
                break;
            case '"':
                out += "&quot;";
                break;
            case '\'':
                out += "&#39;";
                break;
            default:
                out.push_back(ch);
                break;
        }
    }
    return out;
}

std::string fixed(double value, int precision = 1) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

std::string format_count(std::size_t value) {
    std::string digits = std::to_string(value);
    for (int insert_pos = static_cast<int>(digits.size()) - 3; insert_pos > 0; insert_pos -= 3) {
        digits.insert(static_cast<std::size_t>(insert_pos), ",");
    }
    return digits;
}

std::string percent(std::size_t count, std::size_t total) {
    if (total == 0) {
        return "0.0%";
    }
    return fixed(static_cast<double>(count) / static_cast<double>(total) * 100.0, 1) + "%";
}

std::string card_html(const std::string& title, const std::string& value, const std::string& note) {
    std::ostringstream out;
    out << "<article class=\"metric\"><span>" << html_escape(title) << "</span><strong>" << html_escape(value)
        << "</strong><small>" << html_escape(note) << "</small></article>\n";
    return out.str();
}

std::vector<std::pair<std::string, std::size_t>> sorted_counts(const std::map<std::string, std::size_t>& counts) {
    std::vector<std::pair<std::string, std::size_t>> rows(counts.begin(), counts.end());
    std::sort(rows.begin(), rows.end(), [](const auto& left, const auto& right) {
        if (left.second != right.second) {
            return left.second > right.second;
        }
        return left.first < right.first;
    });
    return rows;
}

std::string bar_rows(const std::map<std::string, std::size_t>& counts, std::size_t total, std::size_t limit = 0) {
    if (counts.empty()) {
        return "<p class=\"muted\">没有可显示的数据</p>";
    }
    std::ostringstream out;
    auto rows = sorted_counts(counts);
    if (limit > 0 && rows.size() > limit) {
        rows.resize(limit);
    }
    const double safe_total = static_cast<double>(std::max<std::size_t>(total, 1));
    for (const auto& [name, count] : rows) {
        const double width = static_cast<double>(count) / safe_total * 100.0;
        out << "<div class=\"bar-row\"><div class=\"bar-label\"><span>" << html_escape(name) << "</span><b>" << format_count(count)
            << "</b></div><div class=\"bar-track\"><div class=\"bar-fill\" style=\"width: " << fixed(width, 2)
            << "%\"></div></div></div>\n";
    }
    return out.str();
}

std::string config_table(const double_ok_gesture::RuntimeConfig& config) {
    const auto& gate = config.capture_gate;
    const std::vector<std::pair<std::string, std::string>> rows = {
        {"OK 阈值", fixed(config.recognizer.ok_threshold, 2)},
        {"稳定窗口", std::to_string(config.recognizer.stable_window)},
        {"稳定命中", std::to_string(config.recognizer.stable_min_positive)},
        {"最多手数", std::to_string(config.recognizer.max_num_hands)},
        {"中心 X", fixed(gate.center_x_min, 2) + " - " + fixed(gate.center_x_max, 2)},
        {"中心 Y", fixed(gate.center_y_min, 2) + " - " + fixed(gate.center_y_max, 2)},
        {"双手距离", fixed(gate.min_hand_separation, 2)},
        {"目标手势", gate.require_double_ok ? "稳定双 OK" : "仅几何门控"},
        {"眼镜姿态", gate.require_glasses_pose ? "需要" : "默认不需要"},
    };

    std::ostringstream out;
    for (const auto& [name, value] : rows) {
        out << "<tr><th>" << html_escape(name) << "</th><td>" << html_escape(value) << "</td></tr>\n";
    }
    return out.str();
}

std::string js_bool(bool value) {
    return value ? "true" : "false";
}

std::string render_gui_report(
    const std::filesystem::path& config_path,
    const double_ok_gesture::RuntimeConfig& config,
    const CsvSummary& csv,
    const FileSummary& model) {
    const std::size_t total = std::max<std::size_t>(csv.row_count, 1);
    std::ostringstream cards;
    cards << card_html("训练 CSV", csv.file.exists ? "已找到" : "缺失", csv.file.size_label);
    cards << card_html("样本行数", format_count(csv.row_count), "每行是一只手");
    cards << card_html("OK 正样本", format_count(csv.positive_count), percent(csv.positive_count, total));
    cards << card_html("not OK 负样本", format_count(csv.negative_count), percent(csv.negative_count, total));
    cards << card_html("特征列", std::to_string(csv.feature_count), "总列数 " + std::to_string(csv.column_count));
    cards << card_html("模型文件", model.exists ? "已找到" : "缺失", model.size_label);

    const auto& gate = config.capture_gate;
    std::ostringstream out;
    out << R"(<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Double OK Gesture GUI</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #181818;
      --topbar: #1f1f1f;
      --panel: #252526;
      --panel-2: #2d2d30;
      --ink: #f2f2f2;
      --muted: #a9b0bb;
      --line: #414147;
      --accent: #4f8ff7;
      --accent-2: #7aabff;
      --warn: #f0a044;
      --ok: #4ac26b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
      letter-spacing: 0;
    }
    .app { display: grid; grid-template-columns: 260px minmax(0, 1fr); min-height: 100vh; }
    aside { background: var(--topbar); padding: 24px 18px; border-right: 1px solid #111; }
    aside h1 { margin: 0 0 8px; font-size: 22px; line-height: 1.2; }
    aside p { margin: 0 0 22px; color: var(--muted); font-size: 13px; line-height: 1.6; }
    nav { display: grid; gap: 8px; }
    nav a {
      color: var(--ink);
      text-decoration: none;
      padding: 10px 12px;
      border-radius: 6px;
      background: #2a2a2d;
      border: 1px solid #3a3a40;
      font-size: 14px;
    }
    nav a:hover { background: #333338; border-color: var(--accent); }
    main { padding: 22px; display: grid; gap: 18px; align-content: start; }
    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      min-height: 66px;
      padding: 12px 14px 12px 18px;
      background: var(--topbar);
      border: 1px solid var(--line);
      border-radius: 8px;
      position: relative;
      overflow: hidden;
    }
    .topbar::before { content: ""; position: absolute; left: 0; top: 0; width: 3px; height: 100%; background: var(--accent); }
    .title-line { display: flex; align-items: center; gap: 14px; min-width: 0; }
    .repo-meta { display: flex; align-items: center; gap: 8px; flex: 0 0 auto; font-weight: 700; }
    .branch-chip {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      height: 32px;
      padding: 0 10px;
      border-radius: 6px;
      background: var(--panel-2);
      border: 1px solid #5d6470;
    }
    .branch-dot { width: 8px; height: 8px; border-radius: 999px; background: var(--accent); box-shadow: 0 0 0 3px rgba(79,143,247,0.14); }
    .topbar h2 { margin: 0; font-size: 24px; line-height: 1.2; }
    .status-pill {
      min-width: 170px;
      text-align: center;
      padding: 9px 12px;
      border-radius: 6px;
      background: rgba(74,194,107,0.16);
      color: #8ee5a6;
      font-weight: 700;
      border: 1px solid rgba(74,194,107,0.42);
    }
    .grid { display: grid; gap: 14px; }
    .metrics { grid-template-columns: repeat(6, minmax(130px, 1fr)); }
    .metric, .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }
    .metric { padding: 14px; min-height: 106px; display: grid; align-content: space-between; gap: 8px; }
    .metric span, .muted { color: var(--muted); font-size: 13px; }
    .metric strong { font-size: 24px; line-height: 1.1; }
    .metric small { color: var(--muted); font-size: 12px; }
    .two-col { grid-template-columns: minmax(0, 1.1fr) minmax(340px, 0.9fr); align-items: start; }
    .panel { padding: 18px; overflow: hidden; }
    .panel h3 { margin: 0 0 14px; font-size: 17px; }
    .flow { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; align-items: stretch; }
    .step { background: var(--panel-2); border: 1px solid var(--line); border-radius: 8px; padding: 12px; min-height: 95px; }
    .step b { display: block; font-size: 14px; margin-bottom: 8px; }
    .step span { color: var(--muted); font-size: 12px; line-height: 1.45; }
    .landmark-panel {
      display: grid;
      grid-template-columns: minmax(260px, 0.86fr) minmax(0, 1fr);
      gap: 18px;
      align-items: center;
    }
    .hand-schematic {
      min-height: 328px;
      display: grid;
      place-items: center;
      padding: 14px;
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .hand-schematic svg { width: min(100%, 360px); height: auto; display: block; }
    .landmark-line { stroke: var(--accent-2); stroke-width: 4; stroke-linecap: round; opacity: 0.72; }
    .ok-ring { fill: rgba(74,194,107,0.12); stroke: var(--ok); stroke-width: 4; }
    .landmark-dot { fill: var(--accent); stroke: #181818; stroke-width: 3; }
    .landmark-dot.tip { fill: var(--ok); }
    .landmark-dot.base { fill: var(--warn); }
    .landmark-label { fill: var(--ink); font-size: 13px; font-weight: 800; }
    .landmark-copy { display: grid; gap: 12px; align-content: center; }
    .landmark-copy h3 { margin-bottom: 0; }
    .landmark-copy p { margin: 0; color: var(--muted); font-size: 14px; line-height: 1.6; }
    .landmark-facts { display: grid; grid-template-columns: repeat(3, minmax(110px, 1fr)); gap: 10px; }
    .landmark-fact {
      min-height: 82px;
      padding: 11px;
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .landmark-fact b { display: block; margin-bottom: 6px; font-size: 13px; }
    .landmark-fact span { color: var(--muted); font-size: 12px; line-height: 1.45; }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th, td { padding: 9px 8px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
    th { width: 110px; color: var(--muted); font-weight: 600; }
    .paths { display: grid; gap: 8px; font-size: 13px; color: var(--muted); word-break: break-all; }
    .sim { display: grid; grid-template-columns: minmax(260px, 0.75fr) minmax(0, 1fr); gap: 18px; align-items: start; }
    .controls { display: grid; gap: 12px; }
    label.control { display: grid; gap: 6px; font-size: 14px; }
    input[type="range"] { width: 100%; accent-color: var(--accent-2); }
    input[type="checkbox"] { width: 18px; height: 18px; accent-color: var(--accent); vertical-align: middle; }
    .check {
      display: grid;
      grid-template-columns: 28px minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
      padding: 11px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-bottom: 8px;
      background: var(--panel-2);
    }
    .dot {
      display: grid;
      place-items: center;
      width: 24px;
      height: 24px;
      border-radius: 999px;
      background: #36363b;
      color: var(--muted);
      font-weight: 800;
    }
    .check.pass .dot { background: rgba(74,194,107,0.18); color: #8ee5a6; }
    .check.fail .dot { background: rgba(79,143,247,0.18); color: #b8d2ff; }
    .check strong { font-size: 14px; }
    .check small { color: var(--muted); }
    .result-banner {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 16px;
      border-radius: 8px;
      background: rgba(79,143,247,0.14);
      border: 1px solid rgba(79,143,247,0.48);
      color: #b8d2ff;
      margin-bottom: 14px;
      font-weight: 800;
    }
    .result-banner.ready { background: rgba(74,194,107,0.16); border-color: rgba(74,194,107,0.46); color: #8ee5a6; }
    .bar-row { display: grid; gap: 6px; margin-bottom: 11px; }
    .bar-label { display: flex; justify-content: space-between; gap: 12px; font-size: 13px; }
    .bar-label b { color: var(--muted); }
    .bar-track { height: 9px; background: #35353a; border-radius: 999px; overflow: hidden; }
    .bar-fill { height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent-2)); }
    .compact-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    @media (max-width: 1180px) {
      .metrics { grid-template-columns: repeat(3, minmax(150px, 1fr)); }
      .two-col, .sim, .landmark-panel { grid-template-columns: 1fr; }
    }
    @media (max-width: 760px) {
      .app { grid-template-columns: 1fr; }
      main { padding: 14px; }
      .metrics, .compact-grid, .flow, .landmark-facts { grid-template-columns: 1fr; }
      .hand-schematic { min-height: 280px; }
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
        <a href="#landmarks">关键点示意</a>
        <a href="#gate">门控模拟</a>
        <a href="#data">数据分布</a>
      </nav>
    </aside>
    <main>
      <section class="topbar" id="overview">
        <div class="title-line">
          <div class="repo-meta"><span class="branch-chip"><span class="branch-dot"></span>main</span><strong>03</strong></div>
          <div>
            <h2>当前工程结果</h2>
            <p class="muted">训练用 landmark CSV；运行时目标是摄像头视频帧转 21 点 landmark。</p>
          </div>
        </div>
        <div class="status-pill" id="projectStatus">LOADED</div>
      </section>

      <section class="grid metrics">
)"
        << cards.str() << R"(      </section>

      <section class="grid two-col">
        <div class="panel" id="flow">
          <h3>运行时识别流程</h3>
          <div class="flow">
            <div class="step"><b>视频帧</b><span>摄像头逐帧输入 BGR 图像。</span></div>
            <div class="step"><b>MediaPipe 目标</b><span>检测最多两只手，输出 21 个点。</span></div>
            <div class="step"><b>特征</b><span>归一化坐标、距离、角度、伸展分数。</span></div>
            <div class="step"><b>单手 OK</b><span>模型或规则分数判断每只手。</span></div>
            <div class="step"><b>双手稳定</b><span>最近窗口内多帧命中才通过。</span></div>
            <div class="step"><b>采集门控</b><span>姿态、入框、居中、分离全部通过。</span></div>
          </div>
        </div>
        <div class="panel">
          <h3>关键配置</h3>
          <table>)"
        << config_table(config) << R"(</table>
        </div>
      </section>

      <section class="panel landmark-panel" id="landmarks">
        <div class="hand-schematic">
          <svg viewBox="0 0 300 340" role="img" aria-label="MediaPipe 21 点手部关键点示意">
            <circle class="ok-ring" cx="96" cy="132" r="27"></circle>
            <g>
              <line class="landmark-line" x1="150" y1="300" x2="122" y2="262"></line>
              <line class="landmark-line" x1="122" y1="262" x2="96" y2="224"></line>
              <line class="landmark-line" x1="96" y1="224" x2="82" y2="180"></line>
              <line class="landmark-line" x1="82" y1="180" x2="89" y2="140"></line>
              <line class="landmark-line" x1="150" y1="300" x2="142" y2="248"></line>
              <line class="landmark-line" x1="142" y1="248" x2="136" y2="202"></line>
              <line class="landmark-line" x1="136" y1="202" x2="126" y2="164"></line>
              <line class="landmark-line" x1="126" y1="164" x2="103" y2="126"></line>
              <line class="landmark-line" x1="142" y1="248" x2="164" y2="236"></line>
              <line class="landmark-line" x1="164" y1="236" x2="168" y2="184"></line>
              <line class="landmark-line" x1="168" y1="184" x2="168" y2="134"></line>
              <line class="landmark-line" x1="168" y1="134" x2="168" y2="84"></line>
              <line class="landmark-line" x1="164" y1="236" x2="192" y2="242"></line>
              <line class="landmark-line" x1="192" y1="242" x2="212" y2="196"></line>
              <line class="landmark-line" x1="212" y1="196" x2="226" y2="154"></line>
              <line class="landmark-line" x1="226" y1="154" x2="238" y2="114"></line>
              <line class="landmark-line" x1="150" y1="300" x2="216" y2="266"></line>
              <line class="landmark-line" x1="192" y1="242" x2="216" y2="266"></line>
              <line class="landmark-line" x1="216" y1="266" x2="244" y2="228"></line>
              <line class="landmark-line" x1="244" y1="228" x2="264" y2="194"></line>
              <line class="landmark-line" x1="264" y1="194" x2="280" y2="160"></line>
            </g>
            <g>
              <circle class="landmark-dot base" cx="150" cy="300" r="8"></circle>
              <circle class="landmark-dot" cx="122" cy="262" r="7"></circle>
              <circle class="landmark-dot" cx="96" cy="224" r="7"></circle>
              <circle class="landmark-dot" cx="82" cy="180" r="7"></circle>
              <circle class="landmark-dot tip" cx="89" cy="140" r="8"></circle>
              <circle class="landmark-dot" cx="142" cy="248" r="7"></circle>
              <circle class="landmark-dot" cx="136" cy="202" r="7"></circle>
              <circle class="landmark-dot" cx="126" cy="164" r="7"></circle>
              <circle class="landmark-dot tip" cx="103" cy="126" r="8"></circle>
              <circle class="landmark-dot" cx="164" cy="236" r="7"></circle>
              <circle class="landmark-dot" cx="168" cy="184" r="7"></circle>
              <circle class="landmark-dot" cx="168" cy="134" r="7"></circle>
              <circle class="landmark-dot tip" cx="168" cy="84" r="8"></circle>
              <circle class="landmark-dot" cx="192" cy="242" r="7"></circle>
              <circle class="landmark-dot" cx="212" cy="196" r="7"></circle>
              <circle class="landmark-dot" cx="226" cy="154" r="7"></circle>
              <circle class="landmark-dot tip" cx="238" cy="114" r="8"></circle>
              <circle class="landmark-dot" cx="216" cy="266" r="7"></circle>
              <circle class="landmark-dot" cx="244" cy="228" r="7"></circle>
              <circle class="landmark-dot" cx="264" cy="194" r="7"></circle>
              <circle class="landmark-dot tip" cx="280" cy="160" r="8"></circle>
            </g>
            <g>
              <text class="landmark-label" x="130" y="322">0</text>
              <text class="landmark-label" x="58" y="138">4</text>
              <text class="landmark-label" x="84" y="114">8</text>
              <text class="landmark-label" x="158" y="66">12</text>
              <text class="landmark-label" x="230" y="98">16</text>
              <text class="landmark-label" x="268" y="144">20</text>
            </g>
          </svg>
        </div>
        <div class="landmark-copy">
          <h3>21 点关键点示意</h3>
          <p>对齐原 dev_ GUI 的手部骨架语义：每只手输出 0-20 共 21 个 landmark，拇指端点 4 与食指端点 8 收拢形成 OK 圈。</p>
          <div class="landmark-facts">
            <div class="landmark-fact"><b>最多两只手</b><span>左右手各一套 21 点，输入后分别计算 OK 分数。</span></div>
            <div class="landmark-fact"><b>OK 圈</b><span>4/8 距离、角度和伸展特征共同描述手势形态。</span></div>
            <div class="landmark-fact"><b>双手门控</b><span>仍以稳定双 OK、完整入框、居中和分离作为采集条件。</span></div>
          </div>
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
)"
        << bar_rows(csv.split_counts, csv.row_count) << R"(        </div>
        <div class="panel">
          <h3>手势类别 Top 12</h3>
)"
        << bar_rows(csv.gesture_counts, csv.row_count, 12) << R"(        </div>
      </section>

      <section class="panel">
        <h3>文件路径</h3>
        <div class="paths">
          <div>Config: )"
        << html_escape(config_path.string()) << R"(</div>
          <div>CSV: )"
        << html_escape(csv.file.path.string()) << R"(</div>
          <div>Model: )"
        << html_escape(model.path.string()) << R"(</div>
        </div>
      </section>
    </main>
  </div>
  <script>
    const gateCfg = {
      require_glasses_pose: )"
        << js_bool(gate.require_glasses_pose) << R"(,
      pitch_min: )"
        << fixed(gate.pitch_min, 3) << R"(,
      pitch_max: )"
        << fixed(gate.pitch_max, 3) << R"(,
      roll_min: )"
        << fixed(gate.roll_min, 3) << R"(,
      roll_max: )"
        << fixed(gate.roll_max, 3) << R"(,
      yaw_min: )"
        << fixed(gate.yaw_min, 3) << R"(,
      yaw_max: )"
        << fixed(gate.yaw_max, 3) << R"(,
      min_hand_separation: )"
        << fixed(gate.min_hand_separation, 3) << R"(
    };
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
    function inRange(value, min, max) { return value >= min && value <= max; }
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
        inRange(pitch, gateCfg.pitch_min, gateCfg.pitch_max) &&
        inRange(roll, gateCfg.roll_min, gateCfg.roll_max) &&
        inRange(yaw, gateCfg.yaw_min, gateCfg.yaw_max));
      const state = {
        pose: poseOk,
        twoHands: handCount >= 2,
        visible,
        centered,
        separated: separation >= gateCfg.min_hand_separation,
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
      document.getElementById("handCountText").textContent = `${result.values.handCount}`;
      document.getElementById("separationText").textContent = `${result.values.separation.toFixed(2)}`;
      document.getElementById("pitchText").textContent = `${result.values.pitch}°`;
      document.getElementById("rollText").textContent = `${result.values.roll}°`;
      document.getElementById("yawText").textContent = `${result.values.yaw}°`;
      const banner = document.getElementById("gateResult");
      const [label, prompt] = promptMap[result.reason];
      banner.classList.toggle("ready", result.reason === "ready");
      banner.innerHTML = `<span>${label}</span><small>${prompt}</small>`;
      document.getElementById("checkList").innerHTML = checks.map(([key, label, tag]) => {
        const pass = result.state[key];
        return `<div class="check ${pass ? "pass" : "fail"}">
          <div class="dot">${pass ? "✓" : "!"}</div>
          <strong>${label}</strong>
          <small>${tag}</small>
        </div>`;
      }).join("");
    }
    update();
  </script>
</body>
</html>
)";
    return out.str();
}

void write_gui_report(const GuiReportRequest& request) {
    const auto config = std::filesystem::exists(request.config) ? load_runtime_config(request.config) : RuntimeConfig{};
    const auto csv = scan_feature_csv(request.csv);
    const auto model = summarize_file(request.model);
    if (!request.output.parent_path().empty()) {
        std::filesystem::create_directories(request.output.parent_path());
    }
    std::ofstream out(request.output);
    if (!out) {
        throw std::runtime_error("Cannot write report: " + request.output.string());
    }
    out << render_gui_report(request.config, config, csv, model);
}

}  // namespace double_ok_gesture
