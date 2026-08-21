import test from 'node:test'
import assert from 'node:assert/strict'

import { activeSubtitleText, parseWebVttCues } from './Lightbox/utils/subtitleCues.js'

const SAMPLE = `WEBVTT

00:00:01.000 --> 00:00:03.500
First line

cue-2
00:00:03.500 --> 00:00:05.000 align:center
Second line
continued
`

test('parses cached WebVTT cues for native playback overlay', () => {
  assert.deepEqual(parseWebVttCues(SAMPLE), [
    { start: 1, end: 3.5, text: 'First line' },
    { start: 3.5, end: 5, text: 'Second line\ncontinued' },
  ])
})

test('selects only cues active on the canonical playback clock', () => {
  const cues = parseWebVttCues(SAMPLE)
  assert.equal(activeSubtitleText(cues, 0.9), '')
  assert.equal(activeSubtitleText(cues, 2), 'First line')
  assert.equal(activeSubtitleText(cues, 3.5), 'Second line\ncontinued')
  assert.equal(activeSubtitleText(cues, 5), '')
})
