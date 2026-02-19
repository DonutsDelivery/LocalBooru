#!/bin/bash
source "$HOME/.cargo/env" 2>/dev/null
# Kill stale processes from previous runs
fuser -k 5210/tcp 2>/dev/null
fuser -k 8790/tcp 2>/dev/null
cd "/home/user/Programs/Claude Projects/localbooru"
# GPU video: VA-API preferred (zero-copy DMA-BUF), NVDEC fallback, software last resort.
# HW accel policy is set in lib.rs via WebKitGTK settings API.
export GST_VA_ALL_DRIVERS=1
export LIBVA_DRIVER_NAME=nvidia
export GST_PLUGIN_FEATURE_RANK="vah264dec:MAX,vah265dec:MAX,vaav1dec:MAX,vavp9dec:MAX,nvh264dec:PRIMARY+1,nvh265dec:PRIMARY+1,nvav1dec:PRIMARY+1,nvvp9dec:PRIMARY+1,avdec_h264:MARGINAL,avdec_h265:MARGINAL"
exec npm run tauri:dev >> /tmp/localbooru-dev.log 2>&1
