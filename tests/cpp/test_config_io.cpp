#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "double_ok_gesture/capture_writer.hpp"
#include "double_ok_gesture/cli.hpp"
#include "double_ok_gesture/config.hpp"
#include "double_ok_gesture/json.hpp"
#include "double_ok_gesture/model_io.hpp"

namespace {

#define EXPECT_TRUE(expression)                                                \
    do {                                                                       \
        if (!(expression)) {                                                   \
            throw std::runtime_error(                                          \
                std::string("EXPECT_TRUE failed: ") + #expression);           \
        }                                                                      \
    } while (false)

#define EXPECT_EQ(lhs, rhs)                                                    \
    do {                                                                       \
        if (!((lhs) == (rhs))) {                                               \
            throw std::runtime_error(                                          \
                std::string("EXPECT_EQ failed: ") + #lhs + " != " + #rhs);    \
        }                                                                      \
    } while (false)

#define EXPECT_NEAR(lhs, rhs, epsilon)                                         \
    do {                                                                       \
        if (std::abs((lhs) - (rhs)) > (epsilon)) {                             \
            throw std::runtime_error(                                          \
                std::string("EXPECT_NEAR failed: ") + #lhs + " != " + #rhs); \
        }                                                                      \
    } while (false)

template <typename Exception = std::exception, typename Callable>
void expect_throws(Callable&& callable) {
    bool threw = false;
    try {
        callable();
    } catch (const Exception&) {
        threw = true;
    }
    EXPECT_TRUE(threw);
}

std::filesystem::path fixture_path(const std::string& name) {
    return std::filesystem::temp_directory_path() /
           ("double_ok_" + name);
}

void write_text(const std::filesystem::path& path, const std::string& text) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Cannot create fixture: " + path.string());
    }
    output << text;
    output.close();
    if (!output) {
        throw std::runtime_error("Cannot finish fixture: " + path.string());
    }
}

void test_config_uses_json_scope_and_types() {
    const auto path = fixture_path("scoped_config.json");
    write_text(
        path,
        R"({"max_num_hands":2,"data_capture":{"enabled":true,"output_dir":"captures","cooldown_sec":0.5}})");

    const auto config = double_ok_gesture::load_runtime_config(path);
    EXPECT_TRUE(config.data_capture.enabled);
    EXPECT_EQ(config.data_capture.output_dir.string(), std::string("captures"));

    write_text(path, R"({"stable_window":2.5})");
    expect_throws<std::runtime_error>([&]() {
        (void)double_ok_gesture::load_runtime_config(path);
    });

    write_text(path, R"({"capture_gate":false})");
    expect_throws<std::runtime_error>([&]() {
        (void)double_ok_gesture::load_runtime_config(path);
    });

    write_text(path, R"({"data_capture":{"enable":true}})");
    expect_throws<std::runtime_error>([&]() {
        (void)double_ok_gesture::load_runtime_config(path);
    });
}

void test_config_validation_covers_overrides() {
    double_ok_gesture::RuntimeConfig config;
    config.recognizer.stable_window = 2;
    config.recognizer.stable_min_positive = 3;
    expect_throws<std::invalid_argument>([&]() {
        double_ok_gesture::validate_runtime_config(config);
    });

    config = {};
    expect_throws<std::invalid_argument>([&]() {
        double_ok_gesture::apply_threshold_override(config, 1.1);
    });

    config.data_capture.output_dir.clear();
    expect_throws<std::invalid_argument>([&]() {
        double_ok_gesture::validate_runtime_config(config);
    });
}

void test_json_unicode_and_ambiguous_input() {
    const auto json = double_ok_gesture::parse_json(
        R"({"text":"\u53cc\u624b \uD83D\uDC4C"})");
    EXPECT_EQ(json.get("text")->as_string(), std::string("双手 👌"));

    expect_throws<std::runtime_error>([]() {
        (void)double_ok_gesture::parse_json(R"({"value":1,"value":2})");
    });
    expect_throws<std::runtime_error>([]() {
        (void)double_ok_gesture::parse_json("{\"value\":\"bad\ntext\"}");
    });
    expect_throws<std::runtime_error>([]() {
        (void)double_ok_gesture::parse_json(R"({"value":01})");
    });
}

void test_cli_numbers_require_complete_finite_tokens() {
    EXPECT_EQ(
        double_ok_gesture::parse_int_argument("42", "--count"),
        42);
    EXPECT_TRUE(std::abs(
                    double_ok_gesture::parse_finite_double_argument(
                        "2.5", "--threshold") -
                    2.5) < 1e-12);
    expect_throws<std::invalid_argument>([]() {
        (void)double_ok_gesture::parse_int_argument("42frames", "--count");
    });
    expect_throws<std::invalid_argument>([]() {
        (void)double_ok_gesture::parse_unsigned_argument("-1", "--seed");
    });
    expect_throws<std::invalid_argument>([]() {
        (void)double_ok_gesture::parse_finite_double_argument(
            "nan", "--threshold");
    });
}

class CurrentPathGuard {
public:
    CurrentPathGuard() : original_(std::filesystem::current_path()) {}
    ~CurrentPathGuard() {
        std::error_code ignored;
        std::filesystem::current_path(original_, ignored);
    }

private:
    std::filesystem::path original_;
};

void test_model_round_trip_without_parent_directory() {
    const auto directory = fixture_path("model_io_directory");
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    CurrentPathGuard guard;
    std::filesystem::current_path(directory);

    double_ok_gesture::LinearModelArtifact model;
    model.mean = {0.0, 1.0};
    model.scale = {1.0, 2.0};
    model.coef = {0.5, -0.25};
    model.intercept = 0.1;
    double_ok_gesture::save_model_artifact("model.txt", model);
    const auto loaded = double_ok_gesture::load_model_artifact("model.txt");
    EXPECT_EQ(loaded.coef.size(), 2U);
    EXPECT_TRUE(loaded.feature_columns.empty());
}

void test_model_and_capture_config_reject_invalid_artifacts() {
    const auto path = fixture_path("duplicate_model.txt");
    write_text(
        path,
        "double_ok_model_v1\n"
        "model_type numpy_logreg\n"
        "feature_count 1\n"
        "intercept 0\n"
        "mean 0\n"
        "scale 1\n"
        "coef 1\n"
        "coef 2\n");
    expect_throws<std::runtime_error>([&]() {
        (void)double_ok_gesture::load_model_artifact(path);
    });

    double_ok_gesture::DataCaptureConfig capture;
    capture.cooldown_sec = -1.0;
    expect_throws<std::invalid_argument>([&]() {
        double_ok_gesture::CaptureWriter writer(capture);
    });
}

void test_capture_writer_commits_image_and_metadata_together() {
    const auto directory = fixture_path("capture_writer_directory");
    std::filesystem::remove_all(directory);

    double_ok_gesture::DataCaptureConfig config;
    config.output_dir = directory;
    config.cooldown_sec = 0.0;
    double_ok_gesture::CaptureWriter writer(config);
    double_ok_gesture::CaptureGateDecision decision;
    decision.ready = true;
    decision.reason = double_ok_gesture::GateReason::Ready;
    const cv::Mat frame(8, 8, CV_8UC3, cv::Scalar(10, 20, 30));
    double_ok_gesture::HandPrediction hand;
    hand.handedness = "Unknown";
    hand.ok_score = 0.8;
    hand.is_ok = true;
    double_ok_gesture::LandmarkConfidences visibility{};
    visibility.fill(0.9);
    hand.landmark_confidences = visibility;
    hand.box = double_ok_gesture::HandBoundingBox{
        0.1, 0.2, 0.4, 0.7, 0.85};
    double_ok_gesture::DoubleOKResult result;
    result.hands.push_back(hand);
    result.ok_count = 1;

    const auto path = writer.maybe_save(
        frame,
        result,
        decision,
        "test-backend");
    EXPECT_TRUE(path.has_value());
    EXPECT_TRUE(std::filesystem::is_regular_file(*path));
    EXPECT_TRUE(std::filesystem::is_regular_file(path->string() + ".json"));
    const auto metadata =
        double_ok_gesture::load_json(path->string() + ".json");
    EXPECT_EQ(
        metadata.get("schema")->as_string(),
        std::string("double_ok_capture_v2"));
    EXPECT_EQ(
        metadata.get("label_source")->as_string(),
        std::string("runtime_prediction"));
    EXPECT_EQ(metadata.get("hands")->as_array().size(), 1U);
    const auto& saved_hand = metadata.get("hands")->as_array().front();
    EXPECT_EQ(saved_hand.get("landmarks")->as_array().size(), 21U);
    EXPECT_EQ(saved_hand.get("visibility")->as_array().size(), 21U);
    EXPECT_NEAR(saved_hand.get("box")->get("detection_score")->as_number(),
                0.85,
                1e-12);
    EXPECT_EQ(writer.saved_count(), 1);
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        EXPECT_TRUE(entry.path().filename().string().front() != '.');
    }
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"config_uses_json_scope_and_types", test_config_uses_json_scope_and_types},
        {"config_validation_covers_overrides", test_config_validation_covers_overrides},
        {"json_unicode_and_ambiguous_input", test_json_unicode_and_ambiguous_input},
        {"cli_numbers_require_complete_finite_tokens", test_cli_numbers_require_complete_finite_tokens},
        {"model_round_trip_without_parent_directory", test_model_round_trip_without_parent_directory},
        {"model_and_capture_config_reject_invalid_artifacts", test_model_and_capture_config_reject_invalid_artifacts},
        {"capture_writer_commits_image_and_metadata_together", test_capture_writer_commits_image_and_metadata_together},
    };

    int failures = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            std::cout << "[PASS] " << name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
        }
    }
    return failures == 0 ? 0 : 1;
}
