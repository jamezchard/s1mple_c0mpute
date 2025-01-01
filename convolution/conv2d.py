import numpy as np
import torch
import torch.nn.functional as F


def conv2d(data, f, r):
    height, width = data.shape[:2]
    height_f, width_f = f.shape[:2]
    assert height_f == 2 * r + 1 and width_f == 2 * r + 1
    data_out = np.zeros_like(data, dtype=data.dtype)
    for row_out in range(height):
        for col_out in range(width):
            value_out = 0
            for row_f in range(height_f):
                for col_f in range(width_f):
                    row_in = row_out - r + row_f
                    col_in = col_out - r + col_f
                    if 0 <= row_in < height and 0 <= col_in < width:
                        value_out += f[row_f][col_f] * data[row_in][col_in]
            data_out[row_out][col_out] = value_out
    return data_out


np.random.seed(42)
input_size = 16
r = 2
data = np.random.uniform(10, 20, (input_size, input_size)).astype(np.float32)
filter_ = np.random.uniform(10, 20, (2 * r + 1, 2 * r + 1)).astype(np.float32)
data.tofile(f"conv2d_data_{input_size}x{input_size}.bin")
filter_.tofile(f"conv2d_filter_{2 * r + 1}x{2 * r + 1}.bin")
np.save(f"conv2d_data_{input_size}x{input_size}.npy", data)
np.save(f"conv2d_filter_{2 * r + 1}x{2 * r + 1}.npy", filter_)
print(data)
print(filter_)

my_output = conv2d(data, filter_, r)
my_output.tofile(f"conv2d_output_{input_size}x{input_size}.bin")
np.save(f"conv2d_output_{input_size}x{input_size}.npy", my_output)

data_torch = torch.tensor(data).unsqueeze(0).unsqueeze(0)  # (N, C, H, W)
filter_torch = torch.tensor(filter_).unsqueeze(0).unsqueeze(0)  # (out_channels, in_channels, H, W)
torch_output = F.conv2d(data_torch, filter_torch, padding=r).squeeze().numpy()

difference = np.abs(my_output - torch_output)
max_diff = np.max(difference)
print(f"max diff between conv2d and torch conv2d: {max_diff}")
