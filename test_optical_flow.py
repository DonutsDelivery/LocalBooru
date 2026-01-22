#!/usr/bin/env python3
"""
Standalone test for optical flow frame interpolation.
Tests the GPU pipeline without HLS streaming complexity.

Usage:
    python test_optical_flow.py <video_file> [--target-fps 60] [--output out.mp4] [--preview]

Benchmark mode:
    python test_optical_flow.py <video_file> --benchmark --resolution 1080p
"""

import argparse
import time
import sys
import os

# Add api to path
sys.path.insert(0, os.path.dirname(__file__))


def parse_resolution(res_str: str) -> tuple:
    """Parse resolution string to (width, height)."""
    resolutions = {
        '720p': (1280, 720),
        '1080p': (1920, 1080),
        '1440p': (2560, 1440),
        '4k': (3840, 2160),
        '2160p': (3840, 2160),
    }
    if res_str.lower() in resolutions:
        return resolutions[res_str.lower()]
    # Try parsing WxH format
    if 'x' in res_str:
        try:
            w, h = res_str.split('x')
            return (int(w), int(h))
        except ValueError:
            pass
    return None


def main():
    parser = argparse.ArgumentParser(description='Test optical flow interpolation')
    parser.add_argument('video', help='Input video file')
    parser.add_argument('--target-fps', type=int, default=60, help='Target FPS (default: 60)')
    parser.add_argument('--output', '-o', help='Output video file (optional)')
    parser.add_argument('--preview', action='store_true', help='Show preview window')
    parser.add_argument('--quality', choices=['fast', 'balanced', 'quality'], default='fast')
    parser.add_argument('--max-frames', type=int, default=300, help='Max frames to process (default: 300)')
    parser.add_argument('--no-gpu', action='store_true', help='Force CPU mode')
    parser.add_argument('--no-compile', action='store_true', help='Disable torch.compile for comparison')
    parser.add_argument('--fp32', action='store_true', help='Force FP32 instead of FP16')
    parser.add_argument('--resolution', type=str, help='Test at specific resolution (720p, 1080p, 1440p, 4k, or WxH)')
    parser.add_argument('--benchmark', action='store_true', help='Run multiple passes and report statistics')
    parser.add_argument('--benchmark-passes', type=int, default=3, help='Number of benchmark passes (default: 3)')
    parser.add_argument('--warmup-frames', type=int, default=10, help='Warmup frames before timing (default: 10)')
    args = parser.parse_args()

    import cv2
    import numpy as np
    from api.services.optical_flow import FrameInterpolator, get_backend_status

    # Check for torch
    try:
        import torch
        HAS_TORCH = True
        CUDA_AVAILABLE = torch.cuda.is_available()
    except ImportError:
        HAS_TORCH = False
        CUDA_AVAILABLE = False

    # Print backend status
    print("\n=== Backend Status ===")
    status = get_backend_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    # Determine active backend (same priority as FrameInterpolator)
    if status.get('rife_ncnn_available'):
        active_backend = "RIFE-NCNN (Vulkan GPU)"
    elif status.get('rife_torch_available'):
        active_backend = "RIFE PyTorch (CUDA)"
    elif status.get('cv2_cuda_available'):
        active_backend = "OpenCV CUDA Farneback"
    elif status.get('cuda_available') and status.get('torch_available'):
        active_backend = "PyTorch CUDA (LightweightFlowNet)"
    elif status.get('cv2_available'):
        active_backend = "OpenCV CPU Farneback"
    else:
        active_backend = "Simple Blend (fallback)"

    if args.no_gpu:
        active_backend = "OpenCV CPU Farneback" if status.get('cv2_available') else "Simple Blend (fallback)"

    print(f"\n  Active backend: {active_backend}")

    # Print torch.compile and precision status
    if HAS_TORCH:
        compile_status = "disabled" if args.no_compile else "enabled"
        precision = "FP32" if args.fp32 else "FP16 (default)"
        print(f"  torch.compile: {compile_status}")
        print(f"  Precision: {precision}")
    print()

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {args.video}")
        return 1

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Handle resolution override
    target_width, target_height = width, height
    if args.resolution:
        parsed = parse_resolution(args.resolution)
        if parsed:
            target_width, target_height = parsed
            print(f"=== Resolution Override ===")
            print(f"  Original: {width}x{height}")
            print(f"  Target: {target_width}x{target_height}")
            print()
        else:
            print(f"Warning: Could not parse resolution '{args.resolution}', using original")

    print(f"=== Video Info ===")
    print(f"  Resolution: {target_width}x{target_height}")
    print(f"  Source FPS: {source_fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Target FPS: {args.target_fps}")
    print()

    # Calculate interpolation ratio
    fps_ratio = args.target_fps / source_fps
    frames_between = max(0, int(fps_ratio) - 1)
    print(f"  FPS ratio: {fps_ratio:.2f}x")
    print(f"  Interpolated frames per pair: {frames_between}")
    print()

    # Initialize interpolator
    print(f"=== Initializing Interpolator ===")
    print(f"  Quality preset: {args.quality}")
    print(f"  Use GPU: {not args.no_gpu}")

    interpolator = FrameInterpolator(use_gpu=not args.no_gpu, quality=args.quality)
    interpolator.initialize()
    print()

    # Setup output writer if requested
    out_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(args.output, fourcc, args.target_fps, (target_width, target_height))
        print(f"  Output: {args.output}")

    def run_pass(pass_num: int = 1, is_warmup: bool = False, warmup_only: bool = False):
        """Run a single processing pass. Returns (frame_count, output_count, interp_times, elapsed)."""
        nonlocal cap

        # Reset video to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        prev_frame = None
        frame_count = 0
        output_count = 0
        interp_times = []
        warmup_count = 0

        max_frames_this_pass = args.warmup_frames if warmup_only else args.max_frames
        label = "Warming up" if is_warmup or warmup_only else f"Pass {pass_num}"

        if warmup_only:
            print(f"=== Warming up ({args.warmup_frames} frames) ===")
        elif not is_warmup:
            print(f"=== Processing (Pass {pass_num}) ===")

        start_time = time.time()
        last_progress_time = start_time

        while frame_count < max_frames_this_pass:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize if needed
            if target_width != width or target_height != height:
                frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

            frame_count += 1

            # Generate interpolated frames if we have a previous frame
            if prev_frame is not None and frames_between > 0:
                for i in range(1, frames_between + 1):
                    t = i / (frames_between + 1)

                    t0 = time.perf_counter()
                    interp_frame = interpolator.interpolate(prev_frame, frame, t)
                    t1 = time.perf_counter()

                    # Only record times after warmup
                    if not warmup_only and warmup_count >= args.warmup_frames:
                        interp_times.append(t1 - t0)

                    warmup_count += 1
                    output_count += 1

                    if out_writer and not warmup_only:
                        out_writer.write(interp_frame)

                    if args.preview:
                        cv2.imshow('Interpolated', interp_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

            # Write/show original frame
            if out_writer and not warmup_only:
                out_writer.write(frame)
            output_count += 1

            if args.preview:
                cv2.imshow('Original', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            prev_frame = frame.copy()

            # Live FPS indicator (every 0.5 seconds)
            current_time = time.time()
            if current_time - last_progress_time >= 0.5:
                elapsed_so_far = current_time - start_time
                live_fps = frame_count / elapsed_so_far if elapsed_so_far > 0 else 0
                output_fps = output_count / elapsed_so_far if elapsed_so_far > 0 else 0

                # Calculate instantaneous interpolation FPS
                if interp_times:
                    recent_times = interp_times[-10:] if len(interp_times) >= 10 else interp_times
                    instant_interp_fps = 1.0 / (sum(recent_times) / len(recent_times)) if recent_times else 0
                else:
                    instant_interp_fps = 0

                status = "warmup" if warmup_only or warmup_count < args.warmup_frames else "timing"
                print(f"\r  [{status}] Frame {frame_count}/{max_frames_this_pass} | "
                      f"Source: {live_fps:.1f} fps | Output: {output_fps:.1f} fps | "
                      f"Interp: {instant_interp_fps:.1f} fps   ", end='', flush=True)

                last_progress_time = current_time

        elapsed = time.time() - start_time
        print()  # New line after progress

        return frame_count, output_count, interp_times, elapsed

    # Warmup phase
    if args.warmup_frames > 0 and not args.benchmark:
        print("=== Warm-up Phase ===")
        print(f"  Running {args.warmup_frames} frames to let torch.compile optimize...")
        run_pass(pass_num=0, warmup_only=True)
        print("  Warm-up complete.\n")

    # Main processing
    all_interp_times = []

    if args.benchmark:
        print(f"=== Benchmark Mode ({args.benchmark_passes} passes) ===\n")

        # Warmup pass (not counted)
        print("--- Warmup Pass ---")
        run_pass(pass_num=0, is_warmup=True)
        print()

        # Timed passes
        pass_results = []
        for p in range(1, args.benchmark_passes + 1):
            print(f"--- Benchmark Pass {p}/{args.benchmark_passes} ---")
            frame_count, output_count, interp_times, elapsed = run_pass(pass_num=p)
            pass_results.append({
                'frame_count': frame_count,
                'output_count': output_count,
                'interp_times': interp_times,
                'elapsed': elapsed
            })
            all_interp_times.extend(interp_times)
            print()

        # Aggregate results
        total_frames = sum(r['frame_count'] for r in pass_results)
        total_output = sum(r['output_count'] for r in pass_results)
        total_time = sum(r['elapsed'] for r in pass_results)

        print(f"=== Benchmark Results ({args.benchmark_passes} passes) ===")
        print(f"  Total source frames: {total_frames}")
        print(f"  Total output frames: {total_output}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Avg source rate: {total_frames / total_time:.1f} fps")
        print(f"  Avg output rate: {total_output / total_time:.1f} fps")

    else:
        # Single pass mode
        frame_count, output_count, interp_times, elapsed = run_pass(pass_num=1)
        all_interp_times = interp_times

        print()
        print(f"=== Results ===")
        print(f"  Source frames processed: {frame_count}")
        print(f"  Output frames generated: {output_count}")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Source processing rate: {frame_count / elapsed:.1f} fps")
        print(f"  Output generation rate: {output_count / elapsed:.1f} fps")

    # Detailed timing statistics
    if all_interp_times:
        print()
        print(f"=== Interpolation Timing ===")
        avg_interp = sum(all_interp_times) / len(all_interp_times) * 1000
        max_interp = max(all_interp_times) * 1000
        min_interp = min(all_interp_times) * 1000

        print(f"  Samples: {len(all_interp_times)}")
        print(f"  Mean: {avg_interp:.2f}ms")
        print(f"  Min: {min_interp:.2f}ms")
        print(f"  Max: {max_interp:.2f}ms")

        # Percentile stats
        p50 = np.percentile(all_interp_times, 50) * 1000
        p95 = np.percentile(all_interp_times, 95) * 1000
        p99 = np.percentile(all_interp_times, 99) * 1000
        print(f"  p50 (median): {p50:.2f}ms")
        print(f"  p95: {p95:.2f}ms")
        print(f"  p99: {p99:.2f}ms")

        # Standard deviation
        std_dev = np.std(all_interp_times) * 1000
        print(f"  Std Dev: {std_dev:.2f}ms")

        # Real-time analysis
        print()
        print(f"=== Real-time Analysis ===")
        target_frame_time = 1000 / args.target_fps
        print(f"  Target frame time: {target_frame_time:.2f}ms")

        if avg_interp < target_frame_time:
            headroom = ((target_frame_time - avg_interp) / target_frame_time) * 100
            print(f"  [OK] Fast enough for real-time at {args.target_fps} fps")
            print(f"  Headroom: {headroom:.1f}%")
        else:
            max_realtime_fps = 1000 / avg_interp
            print(f"  [SLOW] Too slow for {args.target_fps} fps real-time")
            print(f"  Max sustainable: ~{max_realtime_fps:.0f} fps")

        # Check p99 for worst-case
        if p99 > target_frame_time:
            print(f"  Warning: p99 latency ({p99:.2f}ms) exceeds frame budget - may cause stutters")

    # GPU Memory stats
    if HAS_TORCH and CUDA_AVAILABLE and not args.no_gpu:
        print()
        print(f"=== GPU Memory ===")
        print(f"  GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
        print(f"  GPU Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB peak")

    # Cleanup
    cap.release()
    if out_writer:
        out_writer.release()
    if args.preview:
        cv2.destroyAllWindows()
    interpolator.cleanup()

    return 0


if __name__ == '__main__':
    sys.exit(main())
