cmake_minimum_required(VERSION 3.16)
project(asdun_access_control LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_VULKAN "Enable Vulkan backend for ncnn if available." OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio objdetect)
find_package(SQLite3 REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)
pkg_check_modules(LIBCAMERA QUIET libcamera)

find_package(ncnn QUIET)
if(ncnn_FOUND)
  add_compile_definitions(USE_NCNN=1)
endif()

if(LIBCAMERA_FOUND)
  add_compile_definitions(HAVE_LIBCAMERA=1)
endif()

add_executable(asdun_access
  src/main.cpp
  src/core/App.cpp
  src/core/StateMachine.cpp
  src/camera/CameraManager.cpp
  src/engine/ConfidenceMapper.cpp
  src/engine/FaceDetector.cpp
  src/engine/FaceRecognizer.cpp
  src/engine/EmotionRecognizer.cpp
  src/engine/InferencePipeline.cpp
  src/tracking/TrackManager.cpp
  src/storage/Database.cpp
  src/storage/FileStore.cpp
  src/storage/EmbeddingStore.cpp
  src/quality/FaceQualityGate.cpp
  src/ui/Renderer.cpp
)

target_include_directories(asdun_access
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${GSTREAMER_INCLUDE_DIRS}
    ${LIBCAMERA_INCLUDE_DIRS}
)

target_link_libraries(asdun_access
  PRIVATE
    ${OpenCV_LIBS}
    SQLite::SQLite3
    Threads::Threads
    ${GSTREAMER_LIBRARIES}
    ${LIBCAMERA_LIBRARIES}
)

if(ncnn_FOUND)
  if(TARGET ncnn)
    target_link_libraries(asdun_access PRIVATE ncnn)
  elseif(TARGET ncnn::ncnn)
    target_link_libraries(asdun_access PRIVATE ncnn::ncnn)
  endif()

  if(ENABLE_VULKAN)
    target_compile_definitions(asdun_access PRIVATE NCNN_USE_VULKAN=1)
  endif()
endif()

target_compile_definitions(asdun_access PRIVATE _CRT_SECURE_NO_WARNINGS)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(asdun_access PRIVATE -Wall -Wextra -Wpedantic)
endif()
