import torch

from optical_flow import normalize, resize, scale, warp


def test_horizontal_warp():
    img = torch.tensor(
        [
            [[1.0, 2.0]],
        ]
    ).unsqueeze(0)
    flow = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.0, 0.0]],
        ]
    ).unsqueeze(0)
    expected = torch.tensor([[[2.0, 2.0]]]).unsqueeze(0)
    warped = warp(img, normalize(flow))
    assert torch.equal(warped, expected)


def test_vertical_warp():
    img = torch.tensor(
        [
            [[1.0], [2.0]],
        ]
    ).unsqueeze(0)
    flow = torch.tensor(
        [
            [[0.0], [0.0]],
            [[1.0], [0.0]],
        ]
    ).unsqueeze(0)
    expected = torch.tensor([[[2.0], [2.0]]]).unsqueeze(0)

    warped = warp(img, normalize(flow))
    assert torch.equal(warped, expected)


def test_scale():
    flow_x = torch.tensor(
        [
            [[1.0, 3.0], [2.0, 4.0]],
        ]
    ).unsqueeze(0)
    flow_y = torch.tensor([[[-1.0, -2.0], [-3.0, -4.0]]]).unsqueeze(0)
    flow = torch.cat((flow_x, flow_y), 1)

    scaled = scale(flow, 2)
    expected_x = 2 * flow_x
    expected_y = 2 * flow_y
    assert torch.equal(scaled[:, 0], expected_x)
    assert torch.equal(scaled[:, 1], expected_y)

    scaled = scale(flow, (3, -1))
    expected_x = 3 * flow_x
    expected_y = -1 * flow_y
    assert torch.equal(scaled[:, 0], expected_x)
    assert torch.equal(scaled[:, 1], expected_y)


def test_resize():
    flow = torch.tensor(
        [[[1.0, 3.0], [2.0, 4.0]], [[-1.0, -2.0], [-3.0, -4.0]]]
    ).unsqueeze(0)

    resized = resize(flow, scale_factor=2)

    expected = (
        2
        * torch.tensor(
            [
                [
                    [1.0000, 1.5000, 2.5000, 3.0000],
                    [1.2500, 1.7500, 2.7500, 3.2500],
                    [1.7500, 2.2500, 3.2500, 3.7500],
                    [2.0000, 2.5000, 3.5000, 4.0000],
                ],
                [
                    [-1.0000, -1.2500, -1.7500, -2.0000],
                    [-1.5000, -1.7500, -2.2500, -2.5000],
                    [-2.5000, -2.7500, -3.2500, -3.5000],
                    [-3.0000, -3.2500, -3.7500, -4.0000],
                ],
            ]
        ).unsqueeze(0)
    )

    assert torch.equal(resized, expected)


def test_resize_height():
    flow = torch.tensor(
        [[[1.0, 3.0], [2.0, 4.0]], [[-1.0, -2.0], [-3.0, -4.0]]]
    ).unsqueeze(0)

    resized = resize(flow, size=(4, 2))

    expected = torch.tensor(
        [
            [[1.0000, 3.0000], [1.2500, 3.2500], [1.7500, 3.7500], [2.0000, 4.0000]],
            [
                [-1.0000, -2.0000],
                [-1.5000, -2.5000],
                [-2.5000, -3.5000],
                [-3.0000, -4.0000],
            ],
        ]
    ).unsqueeze(0)
    expected[:, 1] *= 2
    assert torch.equal(resized, expected)


def test_resize_width():
    flow = torch.tensor(
        [[[1.0, 3.0], [2.0, 4.0]], [[-1.0, -2.0], [-3.0, -4.0]]]
    ).unsqueeze(0)

    resized = resize(flow, size=(2, 4))

    expected = torch.tensor(
        [
            [[1.0000, 1.5000, 2.5000, 3.0000], [2.0000, 2.5000, 3.5000, 4.0000]],
            [
                [-1.0000, -1.2500, -1.7500, -2.0000],
                [-3.0000, -3.2500, -3.7500, -4.0000],
            ],
        ]
    ).unsqueeze(0)
    expected[:, 0] *= 2
    assert torch.equal(resized, expected)
