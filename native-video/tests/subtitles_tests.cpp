// SPDX-License-Identifier: MIT
#include <cassert>
#include <iostream>
#include <vector>

#include "subtitles.h"

using namespace localbooru::native_video;

int main(int argc, char **argv) {
  const auto track = SubtitleTrack::from_webvtt(
      "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nHello 世界\n\n"
      "00:00:02.500 --> 00:00:04.000\nOverlapping\n\n"
      "00:00:05.000 --> 00:00:06.000\n<i>مرحبا &amp; שלום</i>\n\n");
  assert(track.cues().size() == 3);
  assert(track.text_at(0.999).empty());
  assert(track.text_at(1.0) == std::vector<std::string>{"Hello 世界"});
  assert(track.text_at(2.75).size() == 2);
  assert(track.text_at(3.0) == std::vector<std::string>{"Overlapping"});
  assert(track.text_at(4.0).empty());
  assert(track.text_at(5.5) == std::vector<std::string>{"مرحبا & שלום"});
  assert(track.text_at(3.5, 1.0).size() == 2);
  if (argc > 2) {
    const auto embedded =
        SubtitleTrack::from_embedded(argv[1], std::stoi(argv[2]));
    assert(embedded.cues().size() == 2);
    assert(embedded.text_at(0.5) ==
           std::vector<std::string>{"Native subtitle 世界"});
    assert(embedded.text_at(3.2) == std::vector<std::string>{"Second cue"});
  }
  std::cout << "native video subtitle tests passed\n";
}
