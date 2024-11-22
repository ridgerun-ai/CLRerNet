/* Copyright (c) 2018, Grégoire Payen de La Garanderie, Durham University
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
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

#include <torch/extension.h>
#include <torch/types.h>
#include <iostream>
#include <vector>

#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> nms_cpu_forward(
        at::Tensor boxes,
        at::Tensor idx,
        float nms_overlap_thresh,
        unsigned long top_k) {

    auto keep = at::empty({top_k}, torch::kLong);
    auto count = 0;
    auto areas = (boxes.select(1, 2) - boxes.select(1, 0)) *  // (x2 - x1)
                 (boxes.select(1, 3) - boxes.select(1, 1));   // (y2 - y1)

    for (int i = 0; i < idx.size(0); ++i) {
        auto box_idx = idx[i].item<int64_t>();
        bool keep_flag = true;

        for (int j = 0; j < count; ++j) {
            auto kept_idx = keep[j].item<int64_t>();

            // Intersection-over-Union (IoU)
            auto xx1 = std::max(boxes[box_idx][0].item<float>(), boxes[kept_idx][0].item<float>());
            auto yy1 = std::max(boxes[box_idx][1].item<float>(), boxes[kept_idx][1].item<float>());
            auto xx2 = std::min(boxes[box_idx][2].item<float>(), boxes[kept_idx][2].item<float>());
            auto yy2 = std::min(boxes[box_idx][3].item<float>(), boxes[kept_idx][3].item<float>());

            auto w = std::max(0.0f, xx2 - xx1);
            auto h = std::max(0.0f, yy2 - yy1);
            auto inter = w * h;

            auto iou = inter / (areas[box_idx].item<float>() + areas[kept_idx].item<float>() - inter);
            if (iou > nms_overlap_thresh) {
                keep_flag = false;
                break;
            }
        }

        if (keep_flag) {
            keep[count] = box_idx;
            count++;
            if (count >= top_k) {
                break;
            }
        }
    }

    return {keep.narrow(0, 0, count), at::tensor(count), at::empty({})};
}

std::vector<at::Tensor> nms_forward(
        at::Tensor boxes,
        at::Tensor scores,
        float thresh,
        unsigned long top_k) {

    auto idx = std::get<1>(scores.sort(0, true));
    CHECK_INPUT(boxes);
    CHECK_INPUT(idx);

    return nms_cpu_forward(boxes, idx, thresh, top_k);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nms_forward", &nms_forward, "NMS (CPU)");
}
