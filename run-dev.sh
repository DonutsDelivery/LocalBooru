#!/bin/bash
source "$HOME/.cargo/env" 2>/dev/null
# Kill stale processes from previous runs
fuser -k 5210/tcp 2>/dev/null
fuser -k 8790/tcp 2>/dev/null
cd "/home/user/Programs/Claude Projects/localbooru"
# GPU video: VA-API preferred (zero-copy DMA-BUF decode), NVDEC fallback, software disabled.
# DMA-BUF renderer + HW accel policy are set in lib.rs — do NOT set
# WEBKIT_DISABLE_COMPOSITING_MODE or WEBKIT_DISABLE_DMABUF_RENDERER here.
export GST_VA_ALL_DRIVERS=1
export LIBVA_DRIVER_NAME=nvidia
export GST_PLUGIN_FEATURE_RANK="vah264dec:MAX,vah265dec:MAX,vaav1dec:MAX,vavp9dec:MAX,nvh264dec:PRIMARY+1,nvh265dec:PRIMARY+1,nvav1dec:PRIMARY+1,nvvp9dec:PRIMARY+1,nvh264sldec:PRIMARY,nvh265sldec:PRIMARY,avdec_h264:NONE,avdec_h265:NONE"
exec npm run tauri:dev >> /tmp/localbooru-dev.log 2>&1
