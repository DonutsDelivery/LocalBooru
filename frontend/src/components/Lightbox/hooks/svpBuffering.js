const MIN_SVP_START_BUFFER_SECONDS = 8

export function shouldStartSVPPlayback(bufferedAheadSeconds) {
  return bufferedAheadSeconds >= MIN_SVP_START_BUFFER_SECONDS
}
