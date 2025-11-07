import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def mse(output, target):
    with torch.no_grad():
        se = torch.mean((target - output) ** 2)
    return se


def min_mse(output, target):
    """
    selects one closest cell and computes the loss

    the target is the set of velocity target candidates,
    find the closest in them.

    output: torch.tensor e.g. (128, 2000)
    target: torch.tensor e.g. (128, 30, 2000)
    """
    with torch.no_grad():
        distance = torch.pow(
            target - torch.unsqueeze(output, 1), exponent=2
        )  # (128, 30, 2000)
        distance = torch.sum(distance, dim=2)  # (128, 30)
        min_distance = torch.min(distance, dim=1)[0]  # (128,)

        # loss = torch.mean(torch.max(torch.tensor(alpha).float(), min_distance))
        se = torch.mean(min_distance)

    return se
