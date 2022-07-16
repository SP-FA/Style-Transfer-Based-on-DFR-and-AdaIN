from torch import mm, svd


def _calc_mean_std(mat, eps: float = 1e-5):
    """
    Calculate the mean and standard value of a given matrix.

    PARAMETER:
      @ mat: input matrix
      @ eps: A small value added to the variance to avoid divide-by-zero.

    RETURN:
      @ mean value
      @ standard value
    """
    size = mat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    std = (mat.view(N, C, -1).var(dim=2) + eps).sqrt().view(N, C, 1, 1)
    mean = mat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return mean, std


def _AdaIN(cFeat, sFeat):
    """
    Calculate the Adaptive Instance Normalization.

    PARAMETER:
      @ cFeat: content feature
      @ sFeat: style feature
    """
    assert (cFeat.size()[:2] == sFeat.size()[:2])
    size = cFeat.size()

    cMean, cStd = _calc_mean_std(cFeat)
    sMean, sStd = _calc_mean_std(sFeat)

    # normalized feature
    nFeat = (cFeat - cMean.expand(size)) / cStd.expand(size)
    return nFeat * sStd.expand(size) + sMean.expand(size)


def _mat_sqrt(mat):
    U, D, V = svd(mat)
    return mm(mm(U, D.pow(0.5).diag()), V.t())


def _adjust_learning_rate(optimizer, i: int, lr: float, decay: float):
    """
    Imitating the original implementation

    PARAMETER:
      @ i: iteration count
      @ lr: learning rate
    """
    lr = lr / (1.0 + decay * i)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr