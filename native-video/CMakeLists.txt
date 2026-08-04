# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.20)
project(localbooru_native_video LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(nlohmann_json REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
  libavformat libavcodec libavutil libswscale libswresample libva)
pkg_check_modules(SDL2 REQUIRED IMPORTED_TARGET sdl2)
if(UNIX AND NOT APPLE)
  pkg_check_modules(EGL_GLES REQUIRED IMPORTED_TARGET egl glesv2)
endif()

add_library(native-video-core STATIC
  src/audio_decoder.cpp
  src/audio_output.cpp
  src/audio_playback_session.cpp
  src/decoder.cpp
  src/dmabuf_frame_decoder.cpp
  src/dmabuf_frame_uploader.cpp
  src/dmabuf_probe.cpp
  src/dmabuf_surface_producer.cpp
  src/hud.cpp
  src/playback_session.cpp
  src/presenter.cpp
  src/shm_surface_producer.cpp
  src/surface_channel.cpp
  src/surface_pool.cpp
  src/subtitles.cpp
  src/svp_frame_source.cpp
  src/video_frame_decoder.cpp)
target_include_directories(native-video-core PUBLIC src)
target_link_libraries(native-video-core PUBLIC
  PkgConfig::FFMPEG
  PkgConfig::SDL2
  nlohmann_json::nlohmann_json)
target_compile_options(native-video-core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(localbooru-native-video src/main.cpp)
target_include_directories(localbooru-native-video PRIVATE src)
target_link_libraries(localbooru-native-video PRIVATE native-video-core)
target_compile_options(localbooru-native-video PRIVATE -Wall -Wextra -Wpedantic)

add_executable(native-video-dmabuf-probe src/dmabuf_probe_main.cpp)
target_link_libraries(native-video-dmabuf-probe PRIVATE native-video-core)
target_compile_options(native-video-dmabuf-probe PRIVATE -Wall -Wextra -Wpedantic)

if(UNIX AND NOT APPLE)
  add_executable(native-video-dmabuf-egl-import-probe
    src/dmabuf_egl_import_probe_main.cpp)
  target_link_libraries(native-video-dmabuf-egl-import-probe PRIVATE
    native-video-core PkgConfig::EGL_GLES)
  target_compile_options(native-video-dmabuf-egl-import-probe PRIVATE
    -Wall -Wextra -Wpedantic)
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(native-video-svp-smoke tests/svp_frame_source_smoke.cpp)
  target_link_libraries(native-video-svp-smoke PRIVATE native-video-core)
  target_compile_options(native-video-svp-smoke PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(native-video-protocol-tests tests/protocol_tests.cpp)
  target_include_directories(native-video-protocol-tests PRIVATE src)
  target_link_libraries(native-video-protocol-tests PRIVATE nlohmann_json::nlohmann_json)
  target_compile_options(native-video-protocol-tests PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME native-video-protocol COMMAND native-video-protocol-tests)
  add_executable(native-video-frame-pacer-tests tests/frame_pacer_tests.cpp)
  target_include_directories(native-video-frame-pacer-tests PRIVATE src)
  target_compile_options(native-video-frame-pacer-tests PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME native-video-frame-pacer COMMAND native-video-frame-pacer-tests)
  add_executable(native-video-decoder-tests tests/decoder_tests.cpp)
  target_link_libraries(native-video-decoder-tests PRIVATE native-video-core)
  target_compile_options(native-video-decoder-tests PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME native-video-decoder COMMAND native-video-decoder-tests)
  add_executable(native-video-audio-decoder-tests tests/audio_decoder_tests.cpp)
  target_link_libraries(native-video-audio-decoder-tests PRIVATE native-video-core)
  target_compile_options(native-video-audio-decoder-tests PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME native-video-audio-decoder COMMAND native-video-audio-decoder-tests)
  add_executable(native-video-audio-output-tests tests/audio_output_tests.cpp)
  target_link_libraries(native-video-audio-output-tests PRIVATE native-video-core)
  target_compile_options(native-video-audio-output-tests PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME native-video-audio-output COMMAND native-video-audio-output-tests)
  add_executable(native-video-audio-playback-session-tests tests/audio_playback_session_tests.cpp)
  target_link_libraries(native-video-audio-playback-session-tests PRIVATE native-video-core)
  target_compile_options(native-video-audio-playback-session-tests PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME native-video-audio-playback-session COMMAND native-video-audio-playback-session-tests)
  add_executable(native-video-hud-tests tests/hud_tests.cpp)
  target_link_libraries(native-video-hud-tests PRIVATE native-video-core)
  target_compile_options(native-video-hud-tests PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME native-video-hud COMMAND native-video-hud-tests)
  add_executable(native-video-subtitle-tests tests/subtitles_tests.cpp)
  target_link_libraries(native-video-subtitle-tests PRIVATE native-video-core)
  target_compile_options(native-video-subtitle-tests PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME native-video-subtitles COMMAND native-video-subtitle-tests)
  add_executable(native-video-presenter-tests tests/presenter_tests.cpp)
  target_link_libraries(native-video-presenter-tests PRIVATE native-video-core)
  target_compile_options(native-video-presenter-tests PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME native-video-presenter COMMAND native-video-presenter-tests)
  add_executable(native-video-playback-session-tests tests/playback_session_tests.cpp)
  target_link_libraries(native-video-playback-session-tests PRIVATE native-video-core)
  target_compile_options(native-video-playback-session-tests PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME native-video-playback-session COMMAND native-video-playback-session-tests)
  add_executable(native-video-surface-pool-tests tests/surface_pool_tests.cpp)
  target_link_libraries(native-video-surface-pool-tests PRIVATE native-video-core)
  target_compile_options(native-video-surface-pool-tests PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME native-video-surface-pool COMMAND native-video-surface-pool-tests)
  add_executable(native-video-surface-channel-tests tests/surface_channel_tests.cpp)
  target_link_libraries(native-video-surface-channel-tests PRIVATE native-video-core)
  target_compile_options(native-video-surface-channel-tests PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME native-video-surface-channel COMMAND native-video-surface-channel-tests)
  add_executable(native-video-shm-surface-producer-tests tests/shm_surface_producer_tests.cpp)
  target_link_libraries(native-video-shm-surface-producer-tests PRIVATE native-video-core)
  target_compile_options(native-video-shm-surface-producer-tests PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME native-video-shm-surface-producer COMMAND native-video-shm-surface-producer-tests)
  add_executable(native-video-dmabuf-probe-tests tests/dmabuf_probe_tests.cpp)
  target_link_libraries(native-video-dmabuf-probe-tests PRIVATE native-video-core)
  target_compile_options(native-video-dmabuf-probe-tests PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME native-video-dmabuf-probe COMMAND native-video-dmabuf-probe-tests)
  add_executable(native-video-dmabuf-frame-decoder-tests
    tests/dmabuf_frame_decoder_tests.cpp)
  target_link_libraries(native-video-dmabuf-frame-decoder-tests PRIVATE native-video-core)
  target_compile_options(native-video-dmabuf-frame-decoder-tests PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME native-video-dmabuf-frame-decoder
    COMMAND native-video-dmabuf-frame-decoder-tests)
  add_executable(native-video-dmabuf-surface-producer-tests
    tests/dmabuf_surface_producer_tests.cpp)
  target_link_libraries(native-video-dmabuf-surface-producer-tests PRIVATE native-video-core)
  target_compile_options(native-video-dmabuf-surface-producer-tests PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME native-video-dmabuf-surface-producer
    COMMAND native-video-dmabuf-surface-producer-tests)
  add_executable(native-video-dmabuf-pipeline-integration-tests
    tests/dmabuf_pipeline_integration_tests.cpp)
  target_link_libraries(native-video-dmabuf-pipeline-integration-tests PRIVATE
    native-video-core)
  target_compile_options(native-video-dmabuf-pipeline-integration-tests PRIVATE
    -Wall -Wextra -Wpedantic)
  add_test(NAME native-video-dmabuf-pipeline-integration
    COMMAND native-video-dmabuf-pipeline-integration-tests)
  add_executable(native-video-helper-integration-tests tests/helper_integration_tests.cpp)
  target_link_libraries(native-video-helper-integration-tests PRIVATE nlohmann_json::nlohmann_json)
  target_compile_definitions(native-video-helper-integration-tests PRIVATE
    NATIVE_VIDEO_HELPER_PATH="$<TARGET_FILE:localbooru-native-video>")
  target_compile_options(native-video-helper-integration-tests PRIVATE -Wall -Wextra -Wpedantic)
  add_dependencies(native-video-helper-integration-tests localbooru-native-video)
  add_test(NAME native-video-helper-integration COMMAND native-video-helper-integration-tests)
  add_executable(native-video-helper-surface-integration-tests
    tests/helper_surface_integration_tests.cpp)
  target_link_libraries(native-video-helper-surface-integration-tests PRIVATE native-video-core)
  target_compile_definitions(native-video-helper-surface-integration-tests PRIVATE
    NATIVE_VIDEO_HELPER_PATH="$<TARGET_FILE:localbooru-native-video>")
  target_compile_options(native-video-helper-surface-integration-tests PRIVATE -Wall -Wextra -Wpedantic)
  add_dependencies(native-video-helper-surface-integration-tests localbooru-native-video)
  add_test(NAME native-video-helper-surface-integration
    COMMAND native-video-helper-surface-integration-tests)
endif()
