/*
 * Copyright (c) the JPEG XL Project Authors.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once
#include <cmath>
#include <algorithm>
#include <cstdint>

namespace butter {

// libjxl thresholds for butteraugli distance
// calculated using ButteraugliFuzzyInverse(1.5) and (0.5)
static const float good_threshold = 0.7170409415f;
static const float bad_threshold = 1.1556389631f;

// From libjxl/lib/jxl/butteraugli/butteraugli.cc
inline void ScoreToRgb(double score, double good_threshold, double bad_threshold, float rgb[3]) {
  double heatmap[12][3] = {
      {0, 0, 0},       {0, 0, 1},
      {0, 1, 1},       {0, 1, 0},  // Good level
      {1, 1, 0},       {1, 0, 0},  // Bad level
      {1, 0, 1},       {0.5, 0.5, 1.0},
      {1.0, 0.5, 0.5},  // Pastel colors for the very bad quality range.
      {1.0, 1.0, 0.5}, {1, 1, 1},
      {1, 1, 1},  // Last color repeated to have a solid range of white.
  };
  if (score < good_threshold) {
    score = (score / good_threshold) * 0.3;
  } else if (score < bad_threshold) {
    score = 0.3 +
            (score - good_threshold) / (bad_threshold - good_threshold) * 0.15;
  } else {
    score = 0.45 + (score - bad_threshold) / (bad_threshold * 12) * 0.5;
  }
  static const int kTableSize = sizeof(heatmap) / sizeof(heatmap[0]);

  score = std::clamp<double>(score * (kTableSize - 1), 0.0, kTableSize - 2);

  int ix = static_cast<int>(score);
  ix = std::clamp(ix, 0, kTableSize - 2);

  double mix = score - ix;
  for (int i = 0; i < 3; ++i) {
    double v = mix * heatmap[ix + 1][i] + (1 - mix) * heatmap[ix][i];
    rgb[i] = pow(v, 0.5);
  }
}

// jxl::CreateHeatMapImage
inline void fill_heatmap(const float* diffmap, uint8_t* dst_ptr[3], int stride, int width, int height) {
    for (int y = 0; y < height; y++) {
        float* r_dst = reinterpret_cast<float*>(dst_ptr[0] + y * stride);
        float* g_dst = reinterpret_cast<float*>(dst_ptr[1] + y * stride);
        float* b_dst = reinterpret_cast<float*>(dst_ptr[2] + y * stride);
        for (int x = 0; x < width; x++) {
            float val = diffmap[y * width + x];
            float rgb[3];
            ScoreToRgb(val, good_threshold, bad_threshold, rgb);
            r_dst[x] = rgb[0];
            g_dst[x] = rgb[1];
            b_dst[x] = rgb[2];
        }
    }
}

}
