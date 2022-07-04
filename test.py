import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from package import net
from package.function import adaptive_instance_normalization

import time


torch.cuda.synchronize()
start = time.time()


def test_data_transform(size):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


def get_images(image, image_dir):
    assert (image or image_dir)
    interpolation_weights = None
    if image:
        imgPaths = image.split(',')
        if len(imgPaths) == 1:
            imgPaths = imgPaths
        else:
            assert (args.style_interpolation_weights != ''), \
                'Please specify interpolation weights.'
            weights = [int(i) for i in args.style_interpolation_weights.split(',')]
            interpolation_weights = [w / sum(weights) for w in weights]
    else:
        image_dir = Path(image_dir)
        imgPaths = [f for f in image_dir.glob('*')]
    return imgPaths, interpolation_weights


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str, help='File path to the content image')
parser.add_argument('--content_dir', type=str, help='Directory path to a batch of content images')
parser.add_argument('--style', type=str, help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style interpolation or spatial control')
parser.add_argument('--style_dir', type=str, help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder.pth')

# Additional options
parser.add_argument('--content_size', type=int, default=512, help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=512, help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--save_ext', default='.jpg', help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output', help='Directory to save the output image(s)')

# Advanced options
parser.add_argument('--alpha', type=float, default=1.0, help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
parser.add_argument('--style_interpolation_weights', type=str, default='', help='The weight for blending the style of multiple style images')
args = parser.parse_args()

do_interpolation = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)


content_paths, _ = get_images(args.content, args.content_dir)
style_paths, interpolation_weights = get_images(args.style  , args.style_dir  )

decoder = net.decoder
decoder.eval()
decoder.load_state_dict(torch.load(args.decoder))
decoder.to(device)

vgg = net.vgg
vgg.eval()
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])
vgg.to(device)

content_tf = test_data_transform(args.content_size)
style_tf = test_data_transform(args.style_size)

for content_path in content_paths:
    if do_interpolation:  # one content image, N style image
        style = torch.stack([style_tf(Image.open(str(p))) for p in style_paths])
        content = content_tf(Image.open(str(content_path))).unsqueeze(0).expand_as(style)
        style = style.to(device)
        content = content.to(device)
        with torch.no_grad():
            output = style_transfer(vgg, decoder, content, style,
                                    args.alpha, interpolation_weights)
        output = output.cpu()
        output_name = output_dir / '{:s}_interpolation{:s}'.format(
            content_path.stem, args.save_ext)
        save_image(output, str(output_name))

    else:  # process one content and one style
        for style_path in style_paths:
            print(type(content_path))
            # if type(content_path).__name__ == 'list':
            #     path = content_path[0]
            # else:
            #     path = content_path
            #
            # if type(style_path).__name__ == 'list':
            #     path2 = style_path[0]
            # else:
            #     path2 = style_path

            content = content_tf(Image.open(str(content_path)))
            style = style_tf(Image.open(str(style_path)))
            style = style.to(device).unsqueeze(0)
            content = content.to(device).unsqueeze(0)
            with torch.no_grad():
                output = style_transfer(vgg, decoder, content, style,
                                        args.alpha)
            output = output.cpu()

            output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(
                content_path.stem, style_path.stem, args.save_ext)
            save_image(output, str(output_name))


torch.cuda.synchronize()
end = time.time()

print(end - start)