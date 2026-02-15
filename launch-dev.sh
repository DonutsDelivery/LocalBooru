#!/bin/bash
export WEBKIT_DISABLE_COMPOSITING_MODE=1
export GST_PLUGIN_FEATURE_RANK=nvh264dec:MAX,nvh265dec:MAX,nvvp9dec:MAX,nvav1dec:MAX,avdec_h264:NONE,avdec_h265:NONE
cd "/home/user/Programs/Claude Projects/localbooru"
cargo tauri dev
