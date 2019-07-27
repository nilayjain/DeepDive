import torch.nn as nn
import torch.tensor as T
import torch.functional as F


class ConvLayer(nn.Module):
    """
    A 2d conv layer, followed by bn, and relu nonlinearity.
    N, C_in, H_in, W_in  ->  N, C_out, H_out, W_out
    """

    def __init__(self, c_in: int, c_out: int, k: int, s: int = 1, p: int = 0):
        """
        :param c_in: num input channels
        :param c_out: num output channels
        :param k: kernel size
        :param s: stride
        :param p: padding
        """
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x: T):
        return F.relu(self.bn(self.conv(x)))


class LinearLayer(nn.Module):
    """
    A linear layer, followed by bn, and relu nonlinearity.
    """

    def __init__(self, c_in: int, c_out: int):
        """
        :param c_in: in_features
        :param c_out: out_features
        """
        super().__init__()
        self.linear = nn.Linear(c_in, c_out, bias=False)
        self.bn = nn.BatchNorm1d(c_out)

    def forward(self, x: T):
        return F.relu(self.bn(self.linear(x)))


class ResBlock(nn.Module):
    """
    implements a residual block with a skip connection.
    connects 2 instances of a layer with a skip connection.
    # TODO: clone or just take the layer function as input
    """

    def __init__(self, layer: Type[nn.Module]):

    def forward(self, *input):
        pass

