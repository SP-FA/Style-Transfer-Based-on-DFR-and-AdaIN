import torch.backends.cudnn as cudnn
import torch.nn as nn
import argparse
import torch
import os

from tensorboardX import SummaryWriter
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm

from package.function import _adjust_learning_rate as adjustlr
from package.util import get_iter, getDataset
from package import net


def train_transform():
    transformList = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transformList)


if __name__ == "__main__":
    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--content', type=str, default='data/content', help='Folder path to content images')
    parser.add_argument('--style',   type=str, default='data/style')
    parser.add_argument('--encoder', type=str, default='models/encoder.pth')

    parser.add_argument('--save', default='./experiments', help='Directory to save the model')
    parser.add_argument('--log',  default='./logs', help='Directory to save the log')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--decay', type=float, default=5e-5)
    parser.add_argument('--epoch', type=int, default=160000)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--content_weight', type=float, default=1.0)
    parser.add_argument('--style_weight', type=float, default=10.0)
    parser.add_argument('--n_threads', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=10000)
    args = parser.parse_args()

    log_dir = Path(args.log)
    save_dir = Path(args.save)
    log_dir.mkdir(exist_ok=True, parents=True)
    save_dir.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir=args.log)

    decoder = net.decoder
    vgg = net.vgg

    vgg.load_state_dict(torch.load(args.encoder), False)
    vgg = nn.Sequential(*list(vgg.children())[:31])
    network = net.Net(vgg, decoder)
    network.train()
    network.to(device)

    cDataset = getDataset(args.content, train_transform())
    sDataset = getDataset(args.style, train_transform())
    cIter = get_iter(cDataset, args.batch, args.n_threads)
    sIter = get_iter(sDataset, args.batch, args.n_threads)

    optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)

    for i in tqdm(range(args.epoch)):
        adjustlr(optimizer, i, args.lr, args.decay)
        cImages = next(cIter).to(device)
        sImages = next(sIter).to(device)
        cLoss, sLoss = network(cImages, sImages)
        cLoss = args.content_weight * cLoss
        sLoss = args.style_weight * sLoss
        loss = cLoss + sLoss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('loss_content', cLoss.item(), i + 1)
        writer.add_scalar('loss_style', sLoss.item(), i + 1)

        if (i + 1) % args.save_interval == 0 or (i + 1) == args.epoch:
            state_dict = net.decoder.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict, save_dir / 'decoder_iter_%d.pth' % (i + 1))
    writer.close()
