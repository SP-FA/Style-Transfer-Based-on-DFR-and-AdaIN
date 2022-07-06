import torch.nn as nn
from typing import *

from .function import _AdaIN as adain
from .function import _calc_mean_std
from .rotation import _DFR as dfr

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


class Net(nn.Module):
    def __init__(self, encoder, decoder):
        super(Net, self).__init__()
        layers = list(encoder.children())
        self.step1 = nn.Sequential(*layers[:4])  # input   -> relu1_1
        self.step2 = nn.Sequential(*layers[4:11])  # relu1_1 -> relu2_1
        self.step3 = nn.Sequential(*layers[11:18])  # relu2_1 -> relu3_1
        self.step4 = nn.Sequential(*layers[18:31])  # relu3_1 -> relu4_1
        self.decoder = decoder
        self.MSELoss = nn.MSELoss()

        # fix the encoder
        for name in ['step1', 'step2', 'step3', 'step4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def encode(self, input, last: bool = True) -> List:
        """
        Extract step 1-4 from input image.

        PARAMETER:
          @ last: only extract step 4

        RETURN:
          A list which elements stand for step 1-4 orderly.
        """
        results = [input]
        for i in range(4):
            func = getattr(self, 'step{:d}'.format(i + 1))
            results.append(func(results[-1]))
        if last:
            return results[-1]
        else:
            return results[1:]

    def _calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.MSELoss(input, target)

    def _calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        inputMean, inputStd = _calc_mean_std(input)
        targetMean, targetStd = _calc_mean_std(target)
        return self.MSELoss(inputMean, targetMean) + self.MSELoss(inputStd, targetStd)

    def forward(self, content, style, angles: List[float] = [0], alpha: float = 1.0):
        """
        PARAMETER:
          @ content: content img
          @ style: style img
          @ angles: a list which elements are angles the matrix will rotate to.
          @ alpha

        RETURN:
          content loss and style loss
        """
        assert 0 <= alpha <= 1
        cFeat = self.encode(content)
        sFeats = self.encode(style, False)

        rotFeat = sum(dfr(sFeats[-1], angles)) / len(angles)
        adainFeat = adain(cFeat, rotFeat)
        adainFeat = alpha * adainFeat + (1 - alpha) * cFeat

        genImg = self.decoder(adainFeat)
        gFeats = self.encode(genImg, False)

        sLoss = 0
        cLoss = self._calc_content_loss(gFeats[-1], adainFeat)
        for i in range(4):
            sLoss += self._calc_style_loss(gFeats[i], sFeats[i])
        return cLoss, sLoss
