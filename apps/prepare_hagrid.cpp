#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "double_ok_gesture/features.hpp"
#include "double_ok_gesture/json.hpp"

namespace {

const std::set<std::string> kDefaultNegativeLabels = {
    "call",          "dislike",       "fist",      "four",        "like",       "mute",
    "no_gesture",    "one",           "palm",      "peace",       "peace_inverted",
    "rock",          "stop",          "stop_inverted", "three",   "three2",     "two_up",
    "two_up_inverted", "grabbing",    "grip",      "little_finger", "point",    "take_picture",
    "timeout",
};

struct Args {
    std::filesystem::path annotations_dir;
    std::filesystem::path output = "data/processed/hagrid_ok_features.csv";
    std::string positive_label = "ok";
    std::set<std::string> negative_labels = kDefaultNegativeLabels;
    int max_negative_per_class = 12000;
};

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing value for " + key);
            }
            return argv[++i];
        };
        if (key == "--annotations-dir") {
            args.annotations_dir = next();
        } else if (key == "--output") {
            args.output = next();
        } else if (key == "--positive-label") {
            args.positive_label = lower(next());
        } else if (key == "--max-negative-per-class") {
            args.max_negative_per_class = std::stoi(next());
        } else if (key == "--negative-labels") {
            args.negative_labels.clear();
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                args.negative_labels.insert(lower(argv[++i]));
            }
        } else {
            throw std::invalid_argument("Unknown argument: " + key);
        }
    }
    if (args.annotations_dir.empty()) {
        throw std::invalid_argument("--annotations-dir is required");
    }
    if (args.max_negative_per_class < 0) {
        throw std::invalid_argument("max_negative_per_class must be non-negative");
    }
    return args;
}

std::string infer_split(const std::filesystem::path& path) {
    for (const auto& part : path) {
        const std::string value = part.string();
        if (value == "valid" || value == "validation") {
            return "val";
        }
        if (value == "train" || value == "val" || value == "test") {
            return value;
        }
    }
    return "unknown";
}

std::vector<std::string> item_labels(const double_ok_gesture::Json& item, const std::string& fallback_label) {
    if (const auto* labels = item.get("labels"); labels && labels->is_array() && !labels->as_array().empty()) {
        std::vector<std::string> result;
        for (const auto& label : labels->as_array()) {
            if (label.is_string()) {
                result.push_back(lower(label.as_string()));
            }
        }
        if (!result.empty()) {
            return result;
        }
    }
    if (const auto* label = item.get("label"); label && label->is_string()) {
        return {lower(label->as_string())};
    }
    return {fallback_label};
}

std::string item_handedness(const double_ok_gesture::Json& item, std::size_t index) {
    const auto* leading = item.get("leading_hand");
    if (!leading) {
        return "";
    }
    if (leading->is_string()) {
        return leading->as_string();
    }
    if (leading->is_array() && index < leading->as_array().size() && leading->as_array()[index].is_string()) {
        return leading->as_array()[index].as_string();
    }
    return "";
}

double point_number(const double_ok_gesture::Json& value, const std::string& key, std::size_t array_index) {
    if (value.is_object()) {
        const auto* number = value.get(key);
        return number && number->is_number() ? number->as_number() : 0.0;
    }
    if (value.is_array() && array_index < value.as_array().size() && value.as_array()[array_index].is_number()) {
        return value.as_array()[array_index].as_number();
    }
    return 0.0;
}

double_ok_gesture::Landmarks parse_landmarks(const double_ok_gesture::Json& value) {
    if (!value.is_array() || value.as_array().size() != 21) {
        throw std::runtime_error("landmark hand must contain 21 points");
    }
    double_ok_gesture::Landmarks landmarks{};
    const auto& points = value.as_array();
    for (std::size_t i = 0; i < points.size(); ++i) {
        landmarks[i] = {
            point_number(points[i], "x", 0),
            point_number(points[i], "y", 1),
            point_number(points[i], "z", 2),
        };
    }
    double_ok_gesture::validate_landmarks(landmarks);
    return landmarks;
}

std::string csv_escape(const std::string& value) {
    if (value.find_first_of(",\"\n\r") == std::string::npos) {
        return value;
    }
    std::string result = "\"";
    for (char ch : value) {
        if (ch == '"') {
            result += "\"\"";
        } else {
            result.push_back(ch);
        }
    }
    result.push_back('"');
    return result;
}

bool is_checkpoint_path(const std::filesystem::path& path) {
    for (const auto& part : path) {
        if (part == ".ipynb_checkpoints") {
            return true;
        }
    }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        if (!std::filesystem::is_directory(args.annotations_dir)) {
            throw std::runtime_error("Not a directory: " + args.annotations_dir.string());
        }
        std::filesystem::create_directories(args.output.parent_path());
        const auto temporary = args.output.parent_path() / ("." + args.output.filename().string() + ".tmp");
        std::ofstream out(temporary);
        if (!out) {
            throw std::runtime_error("Cannot write output CSV: " + temporary.string());
        }

        const auto names = double_ok_gesture::feature_names();
        out << "split,source_json,image_id,gesture_label,target,handedness";
        for (const auto& name : names) {
            out << ',' << name;
        }
        out << '\n';

        std::map<std::pair<std::string, std::string>, int> negative_counts;
        int rows_written = 0;
        int skipped_invalid = 0;

        for (const auto& entry : std::filesystem::recursive_directory_iterator(args.annotations_dir)) {
            if (!entry.is_regular_file() || entry.path().extension() != ".json" || is_checkpoint_path(entry.path())) {
                continue;
            }
            const std::string fallback_label = lower(entry.path().stem().string());
            const std::string split = infer_split(entry.path());
            const auto payload = double_ok_gesture::load_json(entry.path());
            if (!payload.is_object()) {
                continue;
            }

            for (const auto& [image_id, item] : payload.as_object()) {
                if (!item.is_object()) {
                    continue;
                }
                const auto* landmarks_value = item.get("hand_landmarks");
                if (!landmarks_value) {
                    landmarks_value = item.get("landmarks");
                }
                if (!landmarks_value || !landmarks_value->is_array()) {
                    continue;
                }
                std::vector<std::string> labels = item_labels(item, fallback_label);
                if (labels.size() == 1 && landmarks_value->as_array().size() > 1) {
                    labels.resize(landmarks_value->as_array().size(), labels.front());
                }

                for (std::size_t index = 0; index < landmarks_value->as_array().size(); ++index) {
                    const std::string label = index < labels.size() ? lower(labels[index]) : fallback_label;
                    int target = -1;
                    if (label == args.positive_label) {
                        target = 1;
                    } else if (args.negative_labels.contains(label)) {
                        auto key = std::make_pair(split, label);
                        int& count = negative_counts[key];
                        if (count >= args.max_negative_per_class) {
                            continue;
                        }
                        ++count;
                        target = 0;
                    } else {
                        continue;
                    }

                    try {
                        const std::string handedness = item_handedness(item, index);
                        const auto landmarks = parse_landmarks(landmarks_value->as_array()[index]);
                        const auto features = double_ok_gesture::feature_vector(landmarks, handedness);
                        out << csv_escape(split) << ',' << csv_escape(entry.path().string()) << ',' << csv_escape(image_id)
                            << ',' << csv_escape(label) << ',' << target << ',' << csv_escape(handedness);
                        for (double value : features) {
                            out << ',' << value;
                        }
                        out << '\n';
                        ++rows_written;
                    } catch (const std::exception&) {
                        ++skipped_invalid;
                    }
                }
            }
        }
        out.close();
        std::filesystem::rename(temporary, args.output);
        if (skipped_invalid > 0) {
            std::cerr << "Skipped " << skipped_invalid << " invalid landmark rows\n";
        }
        std::cout << "Wrote " << rows_written << " rows to " << args.output << '\n';
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << '\n';
        return 1;
    }
}
