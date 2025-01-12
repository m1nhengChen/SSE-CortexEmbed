import torch
import torch.nn as nn


def mse(target, predict, hot_num):
    loss = 0
    for hot in range(hot_num):
        loss += torch.nn.functional.mse_loss(predict[:, hot, :], target[:, hot, :])
    return loss


# def rmse(target, predict, hot_num, lamda=2):
#     """
#     revised mse loss function which penalizes reconstruction errors in non-zero elements
#     """
#     loss = 0
#     weight = torch.ones_like(target) + lamda * target
#     for hot in range(hot_num):
#         loss += torch.nn.functional.mse_loss(
#             predict[:, hot, :] * weight[:, hot, :],
#             target[:, hot, :] * weight[:, hot, :],
#         )
#     return loss

def rmse(target, predict, lamda=2):
    """
    revised mse loss function which penalizes reconstruction errors in non-zero elements
    """
    loss = 0
    weight = torch.ones_like(target) + lamda * target
    loss = torch.nn.functional.mse_loss(
            predict * weight,
            target* weight
        )
    return loss

def test_rmse():
    # set random seed
    torch.manual_seed(0)

    # input tensor for testing
    target = torch.tensor([[[1.0, 2.0], [0.0, 1.0]], [[2.0, 0.0], [3.0, 1.0]]])
    predict = torch.tensor([[[1.5, 2.5], [0.5, 1.5]], [[1.5, 0.5], [2.5, 1.5]]])
    hot_num = 2
    lamda = 2

    # calculate RMSE loss
    loss = rmse(target, predict, lamda)
    mse_l = mse(target, predict, hot_num)
    print('mse loss: ',mse_l.item())
    # print result for visual inspection
    print(f"Computed loss: {loss.item()}")

    # ground truth calculation
    weight = torch.ones_like(target) + lamda * target
    expected_loss = 0
    for hot in range(hot_num):
        expected_loss += torch.nn.functional.mse_loss(
            predict[:, hot, :] * weight[:, hot, :],
            target[:, hot, :] * weight[:, hot, :],
        )

    # Check if the calculated loss is close to the expected value(ground truth)
    assert torch.isclose(
        loss, expected_loss
    ), f"Test failed! Expected: {expected_loss.item()}, Got: {loss.item()}"

    print("Test passed!")


# 运行测试用例
# test_rmse()
