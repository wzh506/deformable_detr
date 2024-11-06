import torch

def get_reference_points(spatial_shapes, valid_ratios, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):#lvl是编号，从0开始

        ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                        torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
        ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None] * valid_ratios[:, None]
    return reference_points

def test_get_reference_points():
    # 定义测试输入
    spatial_shapes = [(4, 4), (2, 2)]
    valid_ratios = torch.tensor([[[[1.0, 1.0], [0.5, 0.5]]]], dtype=torch.float32)
    device = torch.device('cpu')
    
    # 调用函数
    reference_points = get_reference_points(spatial_shapes, valid_ratios, device)
    
    # 检查输出形状
    assert reference_points.shape == (1, 20, 1, 2), f"Shape mismatch: {reference_points.shape}"
    
    # 检查输出值
    expected_values = torch.tensor(
        [[[[0.1250, 0.1250]], [[0.3750, 0.1250]], [[0.6250, 0.1250]], [[0.8750, 0.1250]],
          [[0.1250, 0.3750]], [[0.3750, 0.3750]], [[0.6250, 0.3750]], [[0.8750, 0.3750]],
          [[0.1250, 0.6250]], [[0.3750, 0.6250]], [[0.6250, 0.6250]], [[0.8750, 0.6250]],
          [[0.1250, 0.8750]], [[0.3750, 0.8750]], [[0.6250, 0.8750]], [[0.8750, 0.8750]],
          [[0.2500, 0.2500]], [[0.7500, 0.2500]], [[0.2500, 0.7500]], [[0.7500, 0.7500]]]],
        dtype=torch.float32
    )
    assert torch.allclose(reference_points, expected_values, atol=1e-4), "Values mismatch"
    
    print("All tests passed!")

# 运行测试
spatial_shapes = torch.tensor([[16, 16], [8, 8], [4, 4]], dtype=torch.int32)  # [H, W] for each level
print('spatial_shapes ',spatial_shapes.shape)
valid_ratios = torch.tensor([[[1.0, 1.0], [0.8, 0.9], [0.7, 0.75]]], dtype=torch.float32)  # [B, 3, 2]
print('valid_ratios  ',valid_ratios .shape)
device = torch.device('cpu')  # 或者 'cuda' 如果你想在GPU上测试

level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))

# 调用函数
reference_points = get_reference_points(spatial_shapes, valid_ratios, device)
print('reference_points ',reference_points.shape)
# 检查输出的形状是否正确，应该是 [B, N, 1, 2]
assert reference_points.shape == (1, sum([H*W for H, W in spatial_shapes]), 3, 2), \
    f"Expected shape (1, N, 3, 2) but got {reference_points.shape}"

# 你可以根据实际需要添加更多的断言
assert torch.is_tensor(reference_points), "The output should be a tensor"

# 可以打印结果，或者验证某些具体的参考点值（这里可以根据你的期望输出来进行断言）
print(reference_points)
