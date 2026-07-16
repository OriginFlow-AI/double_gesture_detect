cmake_minimum_required(VERSION 3.20)

project(double_ok_gesture VERSION 0.2.0 LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core dnn imgproc imgcodecs videoio)

option(DOUBLE_OK_BUILD_QT_DEMO "Build the Qt live dashboard demo" ON)
option(DOUBLE_OK_BUILD_CAPTURE_TOOL "Build the desktop OpenCV HighGUI capture tool" ON)
option(DOUBLE_OK_ENABLE_RKNN "Build the RK3588 YOLOv8-Pose RKNN backend" OFF)
set(RKNN_ROOT "" CACHE PATH
    "RKNN Runtime root containing include/rknn_api.h and aarch64/librknnrt.so")

set(DOUBLE_OK_CORE_SOURCES
    src/camera.cpp
    src/cli.cpp
    src/capture_writer.cpp
    src/capture_gate.cpp
    src/config.cpp
    src/features.cpp
    src/hand_attribute_classifier.cpp
    src/hand_detector.cpp
    src/json.cpp
    src/landmark_provider.cpp
    src/live_ui.cpp
    src/model_contract.cpp
    src/model_io.cpp
    src/onnx_yolov8_pose_provider.cpp
    src/recognizer.cpp
    src/report.cpp
    src/runtime.cpp
    src/runtime_pipeline.cpp
    src/training.cpp
    src/yolov8_pose_postprocess.cpp
)

if(DOUBLE_OK_ENABLE_RKNN)
    if(RKNN_ROOT)
        find_path(RKNN_INCLUDE_DIR
            NAMES rknn_api.h
            HINTS "${RKNN_ROOT}"
            PATH_SUFFIXES include
            NO_DEFAULT_PATH
            NO_CMAKE_FIND_ROOT_PATH)
        find_library(RKNN_RUNTIME_LIBRARY
            NAMES rknnrt
            HINTS "${RKNN_ROOT}"
            PATH_SUFFIXES aarch64 lib lib64
            NO_DEFAULT_PATH
            NO_CMAKE_FIND_ROOT_PATH)
    else()
        find_path(RKNN_INCLUDE_DIR NAMES rknn_api.h PATH_SUFFIXES include)
        find_library(RKNN_RUNTIME_LIBRARY NAMES rknnrt)
    endif()
    if(NOT RKNN_INCLUDE_DIR OR NOT EXISTS "${RKNN_INCLUDE_DIR}/rknn_api.h")
        message(FATAL_ERROR
            "DOUBLE_OK_ENABLE_RKNN=ON requires rknn_api.h; set RKNN_ROOT or RKNN_INCLUDE_DIR")
    endif()
    if(NOT RKNN_RUNTIME_LIBRARY OR NOT EXISTS "${RKNN_RUNTIME_LIBRARY}")
        message(FATAL_ERROR
            "DOUBLE_OK_ENABLE_RKNN=ON requires librknnrt.so; set RKNN_ROOT or RKNN_RUNTIME_LIBRARY")
    endif()
    add_library(RKNN::Runtime SHARED IMPORTED GLOBAL)
    set_target_properties(RKNN::Runtime PROPERTIES
        IMPORTED_LOCATION "${RKNN_RUNTIME_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${RKNN_INCLUDE_DIR}")
    list(APPEND DOUBLE_OK_CORE_SOURCES src/rknn_yolov8_pose_provider.cpp)
endif()

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
target_compile_definitions(double_ok_gesture_core
    PUBLIC DOUBLE_OK_ENABLE_RKNN=$<BOOL:${DOUBLE_OK_ENABLE_RKNN}>)
if(DOUBLE_OK_ENABLE_RKNN)
    target_link_libraries(double_ok_gesture_core PRIVATE RKNN::Runtime)
endif()

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
add_double_ok_executable(double-ok-eval apps/evaluate.cpp)
add_double_ok_executable(double-ok-gui apps/gui_report.cpp)
add_double_ok_executable(double-ok-headless apps/headless.cpp)
add_double_ok_executable(double-ok-prepare-hagrid apps/prepare_hagrid.cpp)
add_double_ok_executable(double-ok-train apps/train.cpp)

target_link_libraries(double-ok-headless PRIVATE double_ok_gesture_demo_app)

if(DOUBLE_OK_BUILD_QT_DEMO)
    find_package(Qt5 5.12 REQUIRED COMPONENTS Widgets)
    add_double_ok_executable(double-ok-demo apps/demo.cpp)
    target_sources(double-ok-demo PRIVATE src/qt_dashboard.cpp)
    target_compile_definitions(double-ok-demo PRIVATE QT_NO_KEYWORDS)
    target_compile_options(double-ok-demo PRIVATE -fPIC)
    target_link_libraries(double-ok-demo PRIVATE double_ok_gesture_demo_app Qt5::Widgets)
endif()

if(DOUBLE_OK_BUILD_CAPTURE_TOOL)
    find_package(OpenCV REQUIRED COMPONENTS highgui)
    # Some distro OpenCV packages build HighGUI with Qt but do not expose all
    # Qt libraries through the opencv_highgui CMake target.
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(QT5_HIGHGUI REQUIRED Qt5Widgets Qt5Test Qt5OpenGL)
    add_double_ok_executable(double-ok-capture apps/capture_samples.cpp)
    target_include_directories(
        double-ok-capture PRIVATE ${QT5_HIGHGUI_INCLUDE_DIRS})
    target_compile_options(
        double-ok-capture PRIVATE ${QT5_HIGHGUI_CFLAGS_OTHER})
    target_link_libraries(
        double-ok-capture PRIVATE opencv_highgui ${QT5_HIGHGUI_LIBRARIES})
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
        double_ok_yolov8_pose_tests
        tests/cpp/test_yolov8_pose_postprocess.cpp)
    add_double_ok_test(
        double_ok_model_replacement_tests
        tests/cpp/test_model_replacement.cpp)
    target_compile_definitions(
        double_ok_model_replacement_tests
        PRIVATE DOUBLE_OK_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
endif()
