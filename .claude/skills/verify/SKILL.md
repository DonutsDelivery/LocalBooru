---
name: verify
summary: Drive LocalBooru responsive frontend and APK behavior
---

# Verify LocalBooru

## Frontend runtime

1. Confirm the embedded backend is available at `http://127.0.0.1:8790/health`.
2. Start Vite with `npm --prefix frontend run dev -- --host 127.0.0.1`.
3. Drive `http://127.0.0.1:5210` with Playwright at 320×568, phone landscape, and 1280×800.
4. For Android inset simulation, set `--android-inset-top/right/bottom/left` on `document.documentElement.style` after each full navigation.
5. Check every route has no document horizontal overflow and that phone drawer open/close controls remain in bounds.
6. Find a video reliably with `/?filename=.mp4`, open `.media-item:has(.video-indicator)`, then exercise toolbar overflow, transport rows, subtitle sheet, auto-hide reveal-only tap, and horizontal seek/metadata arbitration.

The browser build logs expected Tauri bridge errors because it is not running inside Tauri; distinguish these from renderer errors introduced by the change.

## APK runtime

Build with `./scripts/build-android-apk.sh --keep-cache`, install the resulting APK in the Android sandbox/Appium emulator, then verify portrait/landscape, cutouts, gesture/three-button navigation, fullscreen transient bars, and close/resume restoration.
