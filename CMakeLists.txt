cmake_minimum_required(VERSION 3.20)

project(double_ok_gesture VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)
find_package(PkgConfig REQUIRED)
pkg_check_modules(QT5_GUI REQUIRED Qt5Widgets Qt5Test Qt5OpenGL)

add_library(double_ok_gesture_core
    src/camera.cpp
    src/capture_gate.cpp
    src/config.cpp
    src/features.cpp
    src/json.cpp
    src/model_io.cpp
    src/recognizer.cpp
    src/runtime.cpp
    src/training.cpp
)

target_include_directories(double_ok_gesture_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(double_ok_gesture_core
    PUBLIC
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
        opencv_videoio
)

target_compile_options(double_ok_gesture_core PRIVATE -Wall -Wextra -Wpedantic)

function(add_double_ok_executable target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE double_ok_gesture_core)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
endfunction()

add_double_ok_executable(double-ok-demo apps/demo.cpp)
add_double_ok_executable(double-ok-camera-check apps/camera_check.cpp)
add_double_ok_executable(double-ok-capture apps/capture_samples.cpp)
add_double_ok_executable(double-ok-eval apps/evaluate.cpp)
add_double_ok_executable(double-ok-gui apps/gui_report.cpp)
add_double_ok_executable(double-ok-prepare-hagrid apps/prepare_hagrid.cpp)
add_double_ok_executable(double-ok-train apps/train.cpp)

target_include_directories(double-ok-demo PRIVATE ${QT5_GUI_INCLUDE_DIRS})
target_compile_options(double-ok-demo PRIVATE ${QT5_GUI_CFLAGS_OTHER})
target_link_libraries(double-ok-demo PRIVATE opencv_highgui ${QT5_GUI_LIBRARIES})

target_include_directories(double-ok-capture PRIVATE ${QT5_GUI_INCLUDE_DIRS})
target_compile_options(double-ok-capture PRIVATE ${QT5_GUI_CFLAGS_OTHER})
target_link_libraries(double-ok-capture PRIVATE opencv_highgui ${QT5_GUI_LIBRARIES})

enable_testing()
add_executable(double_ok_gesture_tests tests/cpp/test_core.cpp)
target_link_libraries(double_ok_gesture_tests PRIVATE double_ok_gesture_core)
target_compile_options(double_ok_gesture_tests PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME double_ok_gesture_tests COMMAND double_ok_gesture_tests)
