export function getQualityOptions(nativeReduction = false) {
  return nativeReduction ? [
    { id: 'original', label: 'Original', description: 'Direct stream, native output', maxHeight: Infinity },
    { id: '1080p', label: '1080p', description: 'Direct stream, playback downscale', maxHeight: 1080 },
    { id: '720p', label: '720p', description: 'Direct stream, playback downscale', maxHeight: 720 },
  ] : [
    { id: 'original', label: 'Original', description: 'No transcoding', maxHeight: Infinity },
    { id: '1440p', label: '1440p (QHD)', description: '30 Mbps', maxHeight: 1440 },
    { id: '1080p_enhanced', label: '1080p Enhanced', description: '20 Mbps', maxHeight: 1080 },
    { id: '1080p', label: '1080p', description: '12 Mbps', maxHeight: 1080 },
    { id: '720p', label: '720p', description: '8 Mbps', maxHeight: 720 },
    { id: '480p', label: '480p', description: '4 Mbps', maxHeight: 480 },
  ]
}