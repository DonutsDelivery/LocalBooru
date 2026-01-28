#!/usr/bin/env python3
"""Benchmark interpolation performance."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import cv2, time, numpy as np

VIDEO = "/mnt/storage/media/asmr/7192305797841.mp4"

cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
frames = [cap.read()[1] for _ in range(32)]
cap.release()
print(f"Frames: {len(frames)} @ {frames[0].shape[1]}x{frames[0].shape[0]}")

from api.services.optical_flow import GPUNativeInterpolator

interp = GPUNativeInterpolator(preset='fast', use_ipc_worker=False)
interp.initialize(frames[0].shape[1], frames[0].shape[0])

# Warmup
_ = interp.interpolate(frames[0], frames[1], 0.5)

times = []
for i in range(1, len(frames)):
    t0 = time.perf_counter()
    _ = interp.interpolate(frames[i-1], frames[i], 0.5)
    times.append(time.perf_counter() - t0)

avg = sum(times) / len(times) * 1000
print(f"Previous version: {avg:.2f}ms ({1000/avg:.1f} FPS)")
interp.cleanup()
