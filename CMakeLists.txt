cmake_minimum_required(VERSION 3.20)

project(double_ok_gesture VERSION 0.2.0 LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core dnn imgproc imgcodecs videoio)

option(DOUBLE_OK_BUILD_QT_DEMO "Build the Qt live dashboard demo" ON)

set(DOUBLE_OK_CORE_SOURCES
    src/camera.cpp
    src/cli.cpp
    src/capture_writer.cpp
    src/capture_gate.cpp
    src/config.cpp
    src/features.cpp
    src/json.cpp
    src/landmark_provider.cpp
    src/live_ui.cpp
    src/onnx_mediapipe_provider.cpp
    src/recognizer.cpp
    src/runtime.cpp
    src/runtime_pipeline.cpp
)

add_library(double_ok_gesture_core ${DOUBLE_OK_CORE_SOURCES})

target_include_directories(double_ok_gesture_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(double_ok_gesture_core
    PUBLIC
        opencv_core
        opencv_dnn
        opencv_imgproc
        opencv_imgcodecs
        opencv_videoio
)

find_library(OPENCV_FREETYPE_LIBRARY NAMES opencv_freetype HINTS ${OpenCV_LIBRARY_DIRS})
if(OPENCV_FREETYPE_LIBRARY)
    target_compile_definitions(double_ok_gesture_core PRIVATE DOUBLE_OK_HAS_OPENCV_FREETYPE=1)
    target_link_libraries(double_ok_gesture_core PUBLIC ${OPENCV_FREETYPE_LIBRARY})
endif()

target_compile_options(double_ok_gesture_core PRIVATE -Wall -Wextra -Wpedantic)

add_library(double_ok_gesture_demo_app
    src/demo_app.cpp
)

target_link_libraries(double_ok_gesture_demo_app PUBLIC double_ok_gesture_core)
target_compile_options(double_ok_gesture_demo_app PRIVATE -Wall -Wextra -Wpedantic)

function(add_double_ok_executable target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE double_ok_gesture_core)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
endfunction()

add_double_ok_executable(double-ok-camera-check apps/camera_check.cpp)
add_double_ok_executable(double-ok-headless apps/headless.cpp)

target_link_libraries(double-ok-headless PRIVATE double_ok_gesture_demo_app)

if(DOUBLE_OK_BUILD_QT_DEMO)
    find_package(Qt5 5.12 REQUIRED COMPONENTS Widgets)
    add_double_ok_executable(double-ok-demo apps/demo.cpp)
    target_sources(double-ok-demo PRIVATE src/qt_dashboard.cpp)
    target_compile_definitions(double-ok-demo PRIVATE QT_NO_KEYWORDS)
    target_compile_options(double-ok-demo PRIVATE -fPIC)
    target_link_libraries(double-ok-demo PRIVATE double_ok_gesture_demo_app Qt5::Widgets)
endif()

if(BUILD_TESTING)
    function(add_double_ok_test target source)
        add_executable(${target} ${source})
        target_link_libraries(${target} PRIVATE double_ok_gesture_core ${ARGN})
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
        add_test(NAME ${target} COMMAND ${target})
    endfunction()

    add_double_ok_test(
        double_ok_gesture_tests
        tests/cpp/test_core.cpp
        double_ok_gesture_demo_app)
    add_double_ok_test(
        double_ok_config_io_tests
        tests/cpp/test_config_io.cpp)
    add_double_ok_test(
        double_ok_model_replacement_tests
        tests/cpp/test_model_replacement.cpp)
    target_compile_definitions(
        double_ok_model_replacement_tests
        PRIVATE DOUBLE_OK_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
endif()
