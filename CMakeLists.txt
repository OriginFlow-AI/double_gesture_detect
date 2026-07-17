cmake_minimum_required(VERSION 3.20)

project(double_ok_gesture VERSION 0.2.0 LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core dnn imgproc imgcodecs videoio)

option(DOUBLE_OK_BUILD_QT_DEMO "Build the Qt live dashboard demo" ON)
option(DOUBLE_OK_ENABLE_RKNN "Enable RKNN NPU backend when the SDK is available" ON)
option(DOUBLE_OK_REQUIRE_RKNN "Fail configuration unless the RKNN SDK is available" OFF)
set(RKNN_SDK_ROOT "" CACHE PATH "RKNN Toolkit2/rknpu2 SDK root")

if(DOUBLE_OK_REQUIRE_RKNN)
    set(DOUBLE_OK_ENABLE_RKNN ON)
endif()

set(DOUBLE_OK_HAS_RKNN OFF)
if(DOUBLE_OK_ENABLE_RKNN)
    find_path(RKNN_INCLUDE_DIR
        NAMES rknn_api.h
        HINTS
            "${RKNN_SDK_ROOT}/include"
            "${RKNN_SDK_ROOT}/runtime/Linux/librknn_api/include"
            "${RKNN_SDK_ROOT}/rknpu2/runtime/Linux/librknn_api/include"
    )
    find_library(RKNN_RUNTIME_LIBRARY
        NAMES rknnrt
        HINTS
            "${RKNN_SDK_ROOT}/lib"
            "${RKNN_SDK_ROOT}/lib64"
            "${RKNN_SDK_ROOT}/runtime/Linux/librknn_api/aarch64"
            "${RKNN_SDK_ROOT}/rknpu2/runtime/Linux/librknn_api/aarch64"
    )
    if(RKNN_INCLUDE_DIR AND RKNN_RUNTIME_LIBRARY)
        set(DOUBLE_OK_HAS_RKNN ON)
        find_package(Threads REQUIRED)
        message(STATUS "RKNN NPU backend enabled: ${RKNN_RUNTIME_LIBRARY}")
    elseif(DOUBLE_OK_REQUIRE_RKNN)
        message(FATAL_ERROR
            "RKNN SDK not found. Set RKNN_SDK_ROOT to the official rknpu2 directory "
            "containing runtime/Linux/librknn_api/include/rknn_api.h and the "
            "aarch64/librknnrt.so runtime library.")
    else()
        message(STATUS "RKNN SDK not found; building OpenCV ONNX backend only")
    endif()
endif()

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

if(DOUBLE_OK_HAS_RKNN)
    target_compile_definitions(double_ok_gesture_core PRIVATE DOUBLE_OK_HAS_RKNN=1)
    target_include_directories(double_ok_gesture_core PRIVATE ${RKNN_INCLUDE_DIR})
    target_link_libraries(double_ok_gesture_core PRIVATE
        ${RKNN_RUNTIME_LIBRARY}
        ${CMAKE_DL_LIBS}
        Threads::Threads
    )
    set_target_properties(double_ok_gesture_core PROPERTIES
        INSTALL_RPATH "$ORIGIN/../lib"
    )
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
