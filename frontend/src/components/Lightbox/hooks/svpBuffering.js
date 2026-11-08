// Three 2-second HLS segments, less a half second of slack: MSE reports the
// buffered end a few tens of milliseconds short of the playlist duration, so a
// threshold equal to the playlist total would wait for one more segment.
const MIN_SVP_START_BUFFER_SECONDS = 5.5

// The first video frame lands a frame or two after the audio, so the buffered
// range can open slightly after currentTime 0. hls.js bridges holes up to 0.1 s
// once playing; do not treat such a start gap as "nothing buffered".
const MAX_SVP_START_GAP_SECONDS = 0.25

export function shouldStartSVPPlayback(bufferedAheadSeconds) {
  return bufferedAheadSeconds >= MIN_SVP_START_BUFFER_SECONDS
}

// TimeRanges-like -> seconds buffered ahead of currentTime (0 if none).
export function svpBufferedAhead(ranges, currentTime) {
  for (let i = 0; i < ranges.length; i++) {
    const start = ranges.start(i)
    const end = ranges.end(i)
    if (start - MAX_SVP_START_GAP_SECONDS <= currentTime && end > currentTime) {
      return end - Math.max(currentTime, start)
    }
  }
  return 0
}
