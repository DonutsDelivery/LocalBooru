#!/usr/bin/env python3
"""Detailed benchmark - GPU-only vs with download."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import cv2, time, numpy as np

VIDEO = "/mnt/storage/media/asmr/7192305797841.mp4"

cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
ret, frame1 = cap.read()
ret, frame2 = cap.read()
cap.release()

h, w = frame1.shape[:2]
print(f"Resolution: {w}x{h}")

# Initialize NVOF directly
nvof = cv2.cuda.NvidiaOpticalFlow_2_0.create(
    (w, h),
    cv2.cuda.NvidiaOpticalFlow_2_0_NV_OF_PERF_LEVEL_FAST,
    cv2.cuda.NvidiaOpticalFlow_2_0_NV_OF_OUTPUT_VECTOR_GRID_SIZE_1,
    cv2.cuda.NvidiaOpticalFlow_2_0_NV_OF_HINT_VECTOR_GRID_SIZE_1,
    False, False, False, 0
)

gpu_f1 = cv2.cuda_GpuMat()
gpu_f2 = cv2.cuda_GpuMat()
t = 0.5
N = 30

# Warmup
gpu_f1.upload(frame1)
gpu_f2.upload(frame2)
g1 = cv2.cuda.cvtColor(gpu_f1, cv2.COLOR_BGR2GRAY)
g2 = cv2.cuda.cvtColor(gpu_f2, cv2.COLOR_BGR2GRAY)
flow, _ = nvof.calc(g1, g2, None)

# Test 1: Full pipeline with download
times_full = []
for _ in range(N):
    t0 = time.perf_counter()
    gpu_f1.upload(frame1)
    gpu_f2.upload(frame2)
    g1 = cv2.cuda.cvtColor(gpu_f1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cuda.cvtColor(gpu_f2, cv2.COLOR_BGR2GRAY)
    flow, _ = nvof.calc(g1, g2, None)
    flow_s = flow.convertTo(cv2.CV_32FC2, alpha=-t/32.0)
    fx, fy = cv2.cuda.split(flow_s)
    warped = cv2.cuda.remap(gpu_f1, fx, fy, cv2.INTER_LINEAR | cv2.WARP_RELATIVE_MAP, borderMode=cv2.BORDER_REPLICATE)
    result = cv2.cuda.addWeighted(warped, 1-t, gpu_f2, t, 0)
    out = result.download()  # Download included
    times_full.append(time.perf_counter() - t0)

# Test 2: GPU-only (no download)
times_gpu = []
for _ in range(N):
    t0 = time.perf_counter()
    gpu_f1.upload(frame1)
    gpu_f2.upload(frame2)
    g1 = cv2.cuda.cvtColor(gpu_f1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cuda.cvtColor(gpu_f2, cv2.COLOR_BGR2GRAY)
    flow, _ = nvof.calc(g1, g2, None)
    flow_s = flow.convertTo(cv2.CV_32FC2, alpha=-t/32.0)
    fx, fy = cv2.cuda.split(flow_s)
    warped = cv2.cuda.remap(gpu_f1, fx, fy, cv2.INTER_LINEAR | cv2.WARP_RELATIVE_MAP, borderMode=cv2.BORDER_REPLICATE)
    result = cv2.cuda.addWeighted(warped, 1-t, gpu_f2, t, 0)
    cv2.cuda.Stream.Null().waitForCompletion()  # Sync but no download
    times_gpu.append(time.perf_counter() - t0)

# Test 3: With frame caching (only upload 1 frame)
gpu_f1.upload(frame1)
times_cached = []
for _ in range(N):
    t0 = time.perf_counter()
    gpu_f2.upload(frame2)  # Only upload new frame
    g1 = cv2.cuda.cvtColor(gpu_f1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cuda.cvtColor(gpu_f2, cv2.COLOR_BGR2GRAY)
    flow, _ = nvof.calc(g1, g2, None)
    flow_s = flow.convertTo(cv2.CV_32FC2, alpha=-t/32.0)
    fx, fy = cv2.cuda.split(flow_s)
    warped = cv2.cuda.remap(gpu_f1, fx, fy, cv2.INTER_LINEAR | cv2.WARP_RELATIVE_MAP, borderMode=cv2.BORDER_REPLICATE)
    result = cv2.cuda.addWeighted(warped, 1-t, gpu_f2, t, 0)
    out = result.download()
    times_cached.append(time.perf_counter() - t0)
    gpu_f1, gpu_f2 = gpu_f2, gpu_f1  # Swap for next

print(f"\nFull (2 uploads + download): {sum(times_full)/len(times_full)*1000:.1f}ms = {1000/(sum(times_full)/len(times_full)):.0f} FPS")
print(f"GPU-only (2 uploads, no download): {sum(times_gpu)/len(times_gpu)*1000:.1f}ms = {1000/(sum(times_gpu)/len(times_gpu)):.0f} FPS")
print(f"Cached (1 upload + download): {sum(times_cached)/len(times_cached)*1000:.1f}ms = {1000/(sum(times_cached)/len(times_cached)):.0f} FPS")
