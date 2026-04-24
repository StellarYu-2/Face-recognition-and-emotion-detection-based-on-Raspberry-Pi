cmake_minimum_required(VERSION 3.16)
project(asdun_access_control LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_VULKAN "Enable Vulkan backend for ncnn if available." OFF)
option(REQUIRE_NCNN "Require ncnn for detector/recognizer/emotion inference." ON)
option(ENABLE_RPI4_CPU_TUNE "Tune generated code for Raspberry Pi 4 Cortex-A72." ON)
option(ENABLE_CLOUD_CLIENT "Enable the hybrid cloud HTTP client when libcurl is available." ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio objdetect calib3d)
find_package(SQLite3 REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
if(ENABLE_CLOUD_CLIENT)
  find_package(CURL QUIET)
  if(CURL_FOUND)
    add_compile_definitions(USE_CLOUD_CLIENT=1)
    message(STATUS "libcurl found: cloud client enabled")
  else()
    message(WARNING "libcurl was not found: cloud client will build as a no-op fallback")
  endif()
endif()

pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)
pkg_check_modules(LIBCAMERA QUIET libcamera)

find_package(ncnn QUIET)
if(ncnn_FOUND)
  add_compile_definitions(USE_NCNN=1)
  message(STATUS "ncnn found: USE_NCNN enabled")
elseif(REQUIRE_NCNN)
  message(FATAL_ERROR
    "ncnn was not found, so NCNN model inference would be disabled. "
    "Install ncnn on the Raspberry Pi first, or configure with "
    "-DREQUIRE_NCNN=OFF only if you intentionally want the fallback path.")
else()
  message(WARNING "ncnn was not found: building fallback-only binary without USE_NCNN")
endif()

if(LIBCAMERA_FOUND)
  add_compile_definitions(HAVE_LIBCAMERA=1)
endif()

add_executable(asdun_access
  src/main.cpp
  src/core/App.cpp
  src/core/StateMachine.cpp
  src/camera/CameraManager.cpp
  src/cloud/CloudClient.cpp
  src/engine/ConfidenceMapper.cpp
  src/engine/FaceDetector.cpp
  src/engine/FaceLandmarkEstimator.cpp
  src/engine/FaceAligner.cpp
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

if(ENABLE_CLOUD_CLIENT AND CURL_FOUND)
  target_link_libraries(asdun_access PRIVATE CURL::libcurl)
endif()

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
  if(ENABLE_RPI4_CPU_TUNE AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_options(asdun_access PRIVATE -mcpu=cortex-a72 -mtune=cortex-a72)
  endif()
endif()
