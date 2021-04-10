import argparse
import os
from datetime import datetime
import yaml
from easydict import EasyDict as edict
import torch
from src.utils.utils import init_logger
from train import Train


def train_model(cfg):
  timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
  save_dir = cfg.save_path
  save_dir += "/BIRDS_%d_TEXT_TUNE_%d_CYCLE_%d" % (cfg.batch_size, int(cfg.TEXTtune), int(cfg.CYCLEloss))
  logger = init_logger(save_dir)
  logger.info("Hyper-parameters: %s" % str(cfg))
  cfg.save_path = save_dir
  torch.set_num_threads(cfg.cpu_workers)
  model = Train(cfg)
  model.train()
  

parser = argparse.ArgumentParser(description="Text Guided Image Manipulation")
parser.add_argument("--cfg_file", dest="cfg_file", default="cfg/birds.yml", type=str,
                    help="Path to a config file listing reader, model and solver parameters.")
# learning schedule
parser.add_argument("--model_name", dest="model_name", default="OURS", type=str, help="Our model name")
parser.add_argument("--data_name", dest='data_name', default="birds", type=str, help="birds")
parser.add_argument("--gpu", dest='gpu', default="1", type=str, help="GPUs ")
parser.add_argument("--batch_size", dest='batch_size', default=32, type=int, help="TaGAN default 64")
parser.add_argument("--save_step", dest='save_step', default=13800, type=int, help="save iteration")
parser.add_argument("--niter", dest='niter', default=13800, type=int, help="lr decay per niter ")

parser.add_argument("--iter_G", dest='iter_G', default=1, type=int, help="learning iter for G")
parser.add_argument("--iter_D", dest='iter_D', default=1, type=int, help="learning iter for D")
# networks
parser.add_argument('--generator_encoder', type=str, default='tagan', help="tagan|relgan")
parser.add_argument('--generator_decoder', type=str, default='tagan', help="[tagan]")
parser.add_argument('--text_encoder', type=str, default='bert', help='(|bert|)')


# loss & optimization (fixed with True)
parser.add_argument('--gan_mode', type=str, default='original', help='(|original|ls|wgan|hinge)')
parser.add_argument('--lr_decay', action='store_false', help='Use learning rate decay or weight decay')
# fixed with False
parser.add_argument('--ATTNloss', action='store_true', help='Use attnmap loss')
parser.add_argument('--CYCLEloss', action='store_true', help='Use cycle loss for G')
parser.add_argument('--TEXTtune', action='store_true', help='if not: Freeze Text encoder')


if __name__ == '__main__':
  args = parser.parse_args()
  
  with open(args.cfg_file) as f:
    conf = yaml.safe_load(f)
  cfg = edict(conf)
  print("=" * 80)
  print(f"train {args.model_name} model on {args.data_name}")
  print("=" * 80)
  cfg.update(vars(args))
    
  os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
  cfg.gpu_ids = list(range(len(cfg.gpu.split(","))))
  train_model(cfg)
  
  
 






