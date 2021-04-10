import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from setproctitle import setproctitle
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.data import BirdBertDataset, BirdBertEvalDataset
from src.models.TGIM_model import TGIM_Model
from src.utils.utils import set_seed


class Train(nn.Module):
	def __init__(self, cfg):
		super(Train, self).__init__()
		self.cfg = cfg
		set_seed(self.cfg.seed_num)
		torch.set_num_threads(cfg.cpu_workers)
	
		title = "BIRDS_%d_TEXT_TUNE_%d_CYCLE_%d" % (self.cfg.batch_size, int(self.cfg.TEXTtune), int(self.cfg.CYCLEloss))
		setproctitle(title)
		self._logger = logging.getLogger(__name__)
	
	def _build_dataloader(self):
		if self.cfg.data_name =="birds":
			print("=" * 50)
			print("BIRDS")
			print("=" * 50)
			self.train_dataset = BirdBertDataset(self.cfg, split="train")
			trn_collate_fn = None
			self.eval_dataset = BirdBertEvalDataset(self.cfg)
		else:
			raise ValueError("|coco_one or birds| is only valid")
		
		self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, drop_last=True, shuffle=True, num_workers=int(self.cfg.cpu_workers), collate_fn=trn_collate_fn)
		self.eval_dataloader = DataLoader(self.eval_dataset, batch_size=10, drop_last=False, shuffle=False,num_workers=int(self.cfg.cpu_workers))
		
		self.iterations = len(self.train_dataset) // self.cfg.batch_size
		self.total_iter = self.iterations * self.cfg.num_epoch
		self.cfg.total_iter = self.total_iter
		print("-" * 30)
		print("DATALOADER FINISHED")
		print("-" * 30)
	
	def _build_model(self):
		print("=" * 80)
		print(f'Building model...{self.cfg.model_name}')
		print(f"GPU: {torch.cuda.is_available()}")
		if self.cfg.previous_weight == "":
			self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
			self.start_epoch = 1
			self.model = TGIM_Model(self.cfg, self.device).to(self.device)
		
		print("-" * 30)
		print("BUILD MODEL FINISHED")
		print("-" * 30)
	
	def _setup_training(self):
		save_checkpoint = "%s/%s" % (self.cfg.save_path, "checkpoints")
		tensorboard = "%s/%s" % (self.cfg.save_path, "tensorboard")
		if not os.path.exists(tensorboard):
			os.makedirs(tensorboard)
		if not os.path.exists(save_checkpoint):
			os.makedirs(save_checkpoint)
		self.cfg.save_dirpath = save_checkpoint
		self.summary_writer = SummaryWriter(tensorboard)
		
		print("-" * 30)
		print("TRAINING SETUP FINISHED")
		print("-" * 30)
	
	def train(self):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self._build_dataloader()
		self._build_model()
		self._setup_training()
		
		description_key = {
			'd_real_loss'     : 'D_real',
			'd_fake_loss'     : 'D_fake',
			'd_real_c_loss'   : 'D_cond',
			'd_fake_attn_loss': 'D_attn',
			'g_fake_loss'     : 'G_fake',
			'g_fake_c_loss'   : 'G_cond',
			'g_fake_attn_loss': 'G_attn',
			'g_recon'         : 'G_rec',
			'g_cycle'         : 'G_cycle',
		}
		
		start_time = datetime.now().strftime('%H:%M:%S')
		self._logger.info("Start train model at %s" % start_time)
		self.global_iteration_step = 0
		train_begin = datetime.utcnow()  # New
		eval_data = next(iter(self.eval_dataloader))
		
		for epoch in range(self.start_epoch, self.cfg.num_epoch + 1):
			self.model.train()
			tqdm_batch_iterator = tqdm(self.train_dataloader)
			for batch_idx, batch in enumerate(tqdm_batch_iterator):
				src_img, errD, fake_img = self.model.train_D(batch)
				if self.cfg.CYCLEloss:
					return_text, errG, recon_img, cycle_img = self.model.train_G(batch)
				else:
					return_text, errG, recon_img = self.model.train_G(batch)
					
				description = "[{}][Iter: {:6d}/{:6d}]".format(
					datetime.utcnow() - train_begin,
					self.global_iteration_step, self.total_iter
				)
				for key in errD.keys():
					description += "[{}: {:4f}]".format(description_key[key], errD[key])
				for key in errG.keys():
					description += "[{}: {:4f}]".format(description_key[key], errG[key])
				description += "[G lr: {:7f}][D lr: {:7f}][lr: {:7f}]".format(self.model.G_lr, self.model.D_lr,
				                                                              self.model.old_lr)
				tqdm_batch_iterator.set_description(description)
				# tensorboard
				if (self.global_iteration_step + 1) % self.cfg.summary_step == 0:
					self._logger.info(description)
					img_m = torch.cat((src_img[-1:,:,:,:],src_img[:-1,:,:,:]),0)
					txt_m =[]
					txt_m.append(return_text[-1])
					txt_m.extend(return_text[:-1])
					fake_fig = self.visualize_output(src_img, img_m, fake_img.detach(), return_text, txt_m)
					recon_fig = self.visualize_output(src_img, src_img, recon_img.detach(), return_text, return_text)
					self.summary_writer.add_figure('fake_gen_image', fake_fig, self.global_iteration_step + 1)
					self.summary_writer.add_figure('recon_image', recon_fig, self.global_iteration_step + 1)
					
					if self.cfg.CYCLEloss:
						cycle_fig = self.visualize_output(src_img, fake_img.detach(), cycle_img.detach(), return_text, return_text)
						self.summary_writer.add_figure('cycle_image', cycle_fig, self.global_iteration_step + 1)
					self._train_summaries(errG, errD, batch)
					
					# eval images
					eval_src_cap, eval_tar_cap, eval_src_img, recon, mani= self.model.eval_G(eval_data)
					eval_fig = self.mani_visualize_output(eval_src_img.detach(), recon.detach(), mani.detach(), eval_src_cap, eval_tar_cap)
					self.summary_writer.add_figure('eval_image', eval_fig, self.global_iteration_step + 1)
					
				# save model
				if (self.global_iteration_step + 1) % self.cfg.save_step == 0:
					filename = "%s/%d_net.pth" % (self.cfg.save_dirpath, self.global_iteration_step)
					self.model.save(filename)
				
				description = self.model.update_learning_rate(self.cfg.lr_decay, self.global_iteration_step)
				if description:
					self._logger.info(description)
				
				self.global_iteration_step += 1
				

		return None
	
	def _train_summaries(self, errG, errD, batch):
		self.summary_writer.add_scalar("gen/fake", errG["g_fake_loss"], self.global_iteration_step)
		self.summary_writer.add_scalar("gen/cond", errG["g_fake_c_loss"], self.global_iteration_step)
		if self.cfg.ATTNloss:
			self.summary_writer.add_scalar("gen/attn", errG["g_fake_attn_loss"], self.global_iteration_step)
		if self.cfg.CYCLEloss:
			self.summary_writer.add_scalar("gen/cycle", errG["g_cycle"], self.global_iteration_step)
		self.summary_writer.add_scalar("gen/recon", errG["g_recon"], self.global_iteration_step)
		
		self.summary_writer.add_scalar("dis/real", errD["d_real_loss"], self.global_iteration_step)
		self.summary_writer.add_scalar("dis/fake", errD["d_fake_loss"], self.global_iteration_step)
		self.summary_writer.add_scalar("dis/cond", errD["d_real_c_loss"], self.global_iteration_step)
		if self.cfg.ATTNloss:
			self.summary_writer.add_scalar("dis/attn", errD["d_fake_attn_loss"], self.global_iteration_step)
		self.summary_writer.add_scalar("LR/g_lr", self.model.learning_rate['g_lr'], self.global_iteration_step)
		self.summary_writer.add_scalar("LR/d_lr", self.model.learning_rate['d_lr'], self.global_iteration_step)
		for name, param in self.model.named_parameters():
			self.summary_writer.add_histogram(name, param.clone().cpu().data.numpy(), self.global_iteration_step)
	
	def visualize_output(self, src_img, trg_img, gen_img, src_cap, trg_cap):
		batch_size = src_img.shape[0]
		src_img = (src_img.cpu().numpy() + 1) / 2
		trg_img = (trg_img.cpu().numpy() + 1) / 2
		gen_img = (gen_img.cpu().numpy() + 1) / 2
		fig = plt.figure(figsize=(6 * 3, 4 * batch_size))
		
		for idx in range(batch_size):
			ax = fig.add_subplot(batch_size, 3, idx * 3 + 1)
			ax.imshow(np.transpose(src_img[idx], (1, 2, 0)))
			ax.set_title('Source Image')
			ax.text(0.5, -0.1, src_cap[idx], size=12, ha="center",
			        transform=ax.transAxes, color='blue')
			
			ax.axis('off')
			ax = fig.add_subplot(batch_size, 3, idx * 3 + 2)
			ax.imshow(np.transpose(trg_img[idx], (1, 2, 0)))
			ax.set_title('Target Image')
			
			ax.axis('off')
			ax = fig.add_subplot(batch_size, 3, idx * 3 + 3)
			ax.imshow(np.transpose(gen_img[idx], (1, 2, 0)))
			ax.set_title('Generated Image')
			ax.text(0.5, -0.1, trg_cap[idx], size=12, ha="center",
			        transform=ax.transAxes, color='red')
			ax.axis('off')
		
		return fig
	
	def mani_visualize_output(self, src_img, recon_img, mani_img, src_cap, trg_cap):
		batch_size = src_img.shape[0]
		src_img = (src_img.cpu().numpy() + 1) / 2
		recon_img = (recon_img.cpu().numpy() + 1) / 2
		mani_img = (mani_img.cpu().numpy() + 1) / 2
		fig = plt.figure(figsize=(6 * 3, 4 * batch_size))
		
		for idx in range(batch_size):
			ax = fig.add_subplot(batch_size, 3, idx * 3 + 1)
			ax.imshow(np.transpose(src_img[idx], (1, 2, 0)))
			ax.set_title('Source Image')
			ax.text(0.5, -0.1, src_cap[idx], size=12, ha="center",
			        transform=ax.transAxes, color='blue')
			
			ax.axis('off')
			ax = fig.add_subplot(batch_size, 3, idx * 3 + 2)
			ax.imshow(np.transpose(recon_img[idx], (1, 2, 0)))
			ax.set_title('Target Image')
			
			ax.axis('off')
			ax = fig.add_subplot(batch_size, 3, idx * 3 + 3)
			ax.imshow(np.transpose(mani_img[idx], (1, 2, 0)))
			ax.set_title('Generated Image')
			ax.text(0.5, -0.1, trg_cap[idx], size=12, ha="center",
			        transform=ax.transAxes, color='red')
			ax.axis('off')
		
		return fig