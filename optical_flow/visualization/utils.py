# Copyright (C) 2017-2019, Arraiy, Inc., all rights reserved.
# Copyright (C) 2019-    , Kornia authors, all rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import Tensor


def hsv_to_rgb(image: Tensor) -> Tensor:
    assert torch.is_tensor(image)

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(
            "Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape)
        )

    h = image[..., 0, :, :]
    s = image[..., 1, :, :]
    v = image[..., 2, :, :]

    hi = torch.floor(h * 6) % 6
    f = ((h * 6) % 6) - hi
    one = torch.tensor(1.0, device=image.device)
    p = v * (one - s)
    q = v * (one - f * s)
    t = v * (one - (one - f) * s)

    hi = hi.long()
    indices = torch.stack([hi, hi + 6, hi + 12], dim=-3)
    out = torch.stack(
        (
            v,
            q,
            p,
            p,
            t,
            v,
            t,
            v,
            v,
            q,
            p,
            p,
            p,
            p,
            t,
            v,
            v,
            q,
        ),
        dim=-3,
    )
    out = torch.gather(out, -3, indices)
    return out
