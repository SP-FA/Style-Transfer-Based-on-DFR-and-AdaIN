import torch.nn as nn
import argparse
import torch
import time
import os

from torchvision.utils import save_image
from torchvision import transforms
from pathlib import Path
from PIL import Image
from typing import *

from package.function import _AdaIN as adain
from package.rotation import _DFR as dfr
from package.util import get_images
from package import net


def data_transform(size: int):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha: float):
    assert (0.0 <= alpha <= 1.0)
    cFeat = vgg(content)
    sFeat = vgg(style)
    adainFeat = adain(cFeat, sFeat)
    adainFeat = alpha * adainFeat + (1 - alpha) * cFeat
    return decoder(adainFeat)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.synchronize()
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--content', type=str, default='data/content', help='Folder path to content images')
    parser.add_argument('--style',   type=str, default='data/style',   help='Folder path to style images')
    parser.add_argument('--encoder', type=str, default='models/encoder.pth')
    parser.add_argument('--decoder', type=str, default='models/decoder.pth')
    parser.add_argument('--output',  type=str, default='output', help='Folder to save the output images')

    parser.add_argument('--content_size', type=int, default=512, help='New size for the content image')
    parser.add_argument('--style_size',   type=int, default=512, help='New size for the style image')
    parser.add_argument('--save_ext', type=str, default='.jpg', help='The extension name of the output image')
    parser.add_argument('--alpha', type=float, default=1.0, help='Controlling the degree of stylization. [0,1]')
    args = parser.parse_args()

    outputPath = Path(args.output)
    outputPath.mkdir(exist_ok=True, parents=True)

    cImages = get_images(args.content)
    sImages = get_images(args.style)

    decoder = net.decoder
    decoder.eval()
    decoder.load_state_dict(torch.load(args.decoder))
    decoder.to(device)

    vgg = net.vgg
    vgg.eval()
    vgg.load_state_dict(torch.load(args.encoder))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    vgg.to(device)

    cTransform = data_transform(args.content_size)
    sTransfrom = data_transform(args.style_size)

    for cName, cImg in cImages.items():
        for sName, sImg in sImages.items():
            sImage = sTransfrom(sImg).to(device).unsqueeze(0)
            cImage = cTransform(cImg).to(device).unsqueeze(0)
            with torch.no_grad():
                output = style_transfer(vgg, decoder, cImage, sImage, alpha=args.alpha)
            output = output.cpu()
            outputName = outputPath / '{:s}_stylized_{:s}{:s}'.format(cName, sName, args.save_ext)
            save_image(output, str(outputName))

    torch.cuda.synchronize()
    end = time.time()
    print(end - start)
