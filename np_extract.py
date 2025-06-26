from np_field import UDFNetwork
import os
import torch
from argparse import ArgumentParser
import sys

udf_field = UDFNetwork(range_=1.3, multires=6, scale=1, last_layer='abs').cuda()

parser = ArgumentParser()
parser.add_argument("--outdir", type=str, default='output_v1')
parser.add_argument("--epoch", type=int, nargs='+', default=[30000])
parser.add_argument("--select", action='store_true', default=False)
parser.add_argument("--file", type=str, nargs='+', default=[])
args = parser.parse_args(sys.argv[1:])
dirname = args.outdir


allfile = os.listdir(dirname)

if args.select:
   allfile = args.file

for file in allfile:
  print(file)

  path = os.path.join(dirname, file)
  for epoch in args.epoch: 
      ckpt_path = os.path.join(path, 'field%d.pth'%epoch)
      if not os.path.exists(ckpt_path):
        continue
      ckpt, _ = torch.load(ckpt_path)

      udf_field.load_state_dict(ckpt)
      mesh_path = path
      threshs = [0.0]
      thresh = 0.0
      udf_field.nudf_extract(resolution=512, base_exp_dir=mesh_path, dist_threshold_ratio=2.0, threshold=thresh, iteration=epoch, voxel_origin=[-0.55,-0.55,-0.55])
