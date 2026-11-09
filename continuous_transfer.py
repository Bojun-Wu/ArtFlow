import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
from pathlib import Path


import time
import numpy as np
import random
from datetime import datetime


def test_transform(img, size):
    transform_list = []
    h, w, _ = np.shape(img)
    divisor = 2 ** args.n_block
    if size == 0:
        # Preserve the original resolution. Only pad to satisfy the flow blocks.
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad((0, 0, (-w) % divisor, (-h) % divisor), padding_mode='edge'),
        ])
    if h<w:
        newh = size
        neww = w/h*size
    else:
        neww = size
        newh = h/w*size
    neww = max(divisor, int(neww//divisor*divisor))
    newh = max(divisor, int(newh//divisor*divisor))
    transform_list.append(transforms.Resize((newh, neww)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str, default=None,
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str, default='input/content',
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str, default=None,
                    help='Single style image to reuse in every round')
parser.add_argument('--style_dir', type=str, default='input/style',
                    help='Directory path to a batch of style images')
parser.add_argument('--decoder', type=str, default='experiments/decoder2.pth.tar')

# Additional options
parser.add_argument('--size', type=int, default=512,
                    help='New size for the content and style images, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')
parser.add_argument('--rounds', type=int, default=20,
                    help='Number of rounds for a single style; maximum styles in directory mode')

# glow parameters
parser.add_argument('--operator', type=str, default='adain',
                    help='style feature transfer operator')
parser.add_argument('--n_flow', default=8, type=int, help='number of flows in each block')# 32
parser.add_argument('--n_block', default=2, type=int, help='number of blocks')# 4
parser.add_argument('--no_lu', action='store_true', help='use plain convolution instead of LU decomposed version')
parser.add_argument('--affine', default=False, type=bool, help='use affine coupling instead of additive')

args = parser.parse_args()
if args.rounds < 1 or args.size < 0 or 0 < args.size < 4:
    parser.error('rounds must be positive; size must be 0 or at least 4')
for path in (args.content, args.style, args.decoder):
    if path is not None and not Path(path).is_file():
        parser.error('File not found: {}'.format(path))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.operator == 'wct':
    from glow_wct import Glow
elif args.operator == 'adain':
    from glow_adain import Glow
elif args.operator == 'decorator':
    from glow_decorator import Glow
else:
    raise('Not implemented operator', args.operator)


output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

assert (args.content or args.content_dir)
assert (args.style or args.style_dir)

content_dir = Path(args.content_dir)
content_paths = [Path(args.content)] if args.content else [f for f in content_dir.glob('*')]
style_dir = Path(args.style_dir)
style_paths = [Path(args.style)] * args.rounds if args.style else [f for f in style_dir.glob('*')][1:args.rounds + 1]
if not content_paths or not style_paths:
    parser.error('No content or style images selected')
run_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')



# glow
glow = Glow(3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu)

# -----------------------resume training------------------------
if os.path.isfile(args.decoder):
    print("--------loading checkpoint----------")
    checkpoint = torch.load(args.decoder, map_location='cpu')
    args.start_iter = checkpoint['iter']
    glow.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}'".format(args.decoder))
else:
    raise FileNotFoundError(args.decoder)
glow = glow.to(device)
glow.eval()

# -----------------------start------------------------
for content_path in content_paths:
    count = 0
    first_output = None
    for style_path in style_paths:
        if count==0:
            content = Image.open(str(content_path)).convert('RGB')
            original_width, original_height = content.size
            img_transform = test_transform(content, args.size)
            content = img_transform(content)
            content = content.to(device).unsqueeze(0)
        else:
            content = output.to(device)

        style = Image.open(str(style_path)).convert('RGB')
        img_transform = test_transform(style, args.size)
        style = img_transform(style)
        style = style.to(device).unsqueeze(0)
        
        with torch.no_grad():
            # content/style ---> z ---> stylized 
            z_c = glow(content, forward=True)
            z_s = glow(style, forward=True)
            output = glow(z_c, forward=False, style=z_s)
            output = output.cpu()

        count += 1
        if first_output is None:
            first_output = output.clone()
        output_name = output_dir / '{}_ArtFlow-{}_{}_stylized_{}_round_{:03d}_of_{:03d}{}'.format(
            run_id, 'AdaIN' if args.operator == 'adain' else args.operator.upper(),
            content_path.stem, style_path.stem, count, len(style_paths), args.save_ext)
        # Keep full floating-point output for feedback; crop/quantize only a preview.
        preview = output[:, :, :original_height, :original_width] if args.size == 0 else output
        save_image(preview, str(output_name))
        print(output_name, flush=True)
        
        next_path = output_name
            
