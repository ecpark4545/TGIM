import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from .networks.discriminator import Discriminator
from .networks.generator import Generator


class TGIM_Model(torch.nn.Module):
	def __init__(self, cfg, device):
		print('Building TGIM_Model...')
		super().__init__()
		# hyper parameters
		self.cfg = cfg
		self.device = device
		self.tokenize = AutoTokenizer.from_pretrained(self.cfg.bert_pretrain)
		
		self.old_lr = self.cfg.lr
		self.learning_rate = {}
		self.learning_rate['lr'] = self.old_lr
		
		# Models

		self.netG = Generator(cfg)
		self.netD = Discriminator(cfg)
		
		self.netG.to(self.device)
		self.netD.to(self.device)
		
		if not self.cfg.TEXTtune:
			self.netG.text_init.eval()
			self.netD.text_init.eval()
		
		# Criterion & Optimizer
		self.criterionL1Pixel = nn.L1Loss()
		self._build_optimizer()
		
		# initialization
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					nn.init.constant_(m.bias.data, 0)
			elif isinstance(m, nn.ConvTranspose2d):
				nn.init.kaiming_normal_(m.weight.data)
				if m.bias is not None:
					nn.init.constant_(m.bias.data, 0)
			else:
				pass
	
	def train_G(self, batch):
		src_img, src_cap, src_cls_id = batch
		errG = {}
		
		# Text data preprocess
		src_encoding = self.tokenize(list(src_cap), padding='max_length', truncation=True, max_length=self.cfg.max_len,return_tensors="pt")
		src_img = src_img.to(self.device)
		src_cap, src_mask, src_ids = src_encoding["input_ids"].to(self.device), src_encoding["attention_mask"].to(self.device), src_encoding["token_type_ids"].to(self.device)
		
		# word for D
		D_src_words = self.netD.text_init(src_cap, src_mask, src_ids)
		source_w_emb = self.netD.text_encoder(D_src_words)
		# sent for G
		G_src_words = self.netG.text_init(src_cap, src_mask, src_ids)
		source_s_emb = self.netG.text_encoder(G_src_words)
		
		n_w_embs = torch.cat((source_w_emb[-1:, :, :], source_w_emb[:-1, :, :]), 0)
		n_mask = torch.cat((src_mask[-1:, :], src_mask[:-1, :]), 0)
		n_s_embs = torch.cat((source_s_emb[-1:, :], source_s_emb[:-1, :]), 0)
		
		# Image data preprocess
		src_img_emb = self.netG.encoder(src_img)
		
		# G -> D
		fake, (z_mean, z_log_stddev) = self.netG.decoder(src_img_emb, n_s_embs)
		kld = torch.mean(-z_log_stddev + 0.5 * (torch.exp(2 * z_log_stddev) + torch.pow(z_mean, 2) - 1))
		fake_features = self.netD.encoder(fake)
		src_features = self.netD.encoder(src_img)
		
		fake_logit = self.netD.adv(fake_features[-1])
		fake_loss = F.binary_cross_entropy_with_logits(fake_logit, self.ones_like(fake_logit))
		errG['g_fake_loss'] = fake_loss.item()
		
		fake_c_prob = self.netD.TaGAN_D(fake_features, n_w_embs, n_mask, negative=False)
		fake_c_loss = F.binary_cross_entropy(fake_c_prob, self.ones_like(fake_c_prob))
		errG['g_fake_c_loss'] = fake_c_loss.item()
		
		
		if self.cfg.ATTNloss:
			tar_region_word_attn, tar_region_word_attn_label = self.netD.word_region(
				fake_features[0], src_features[0], source_w_emb, n_w_embs, src_mask, n_mask)
			fake_attn_loss = self.criterionL1Pixel(tar_region_word_attn, tar_region_word_attn_label)
			errG['g_fake_attn_loss'] = fake_attn_loss.item()
		else:
			fake_attn_loss = 0
			
			
		if self.cfg.CYCLEloss:
			fake_emb = self.netG.encoder(fake)
			cycle_recon, (c_z_mean, c_z_log_stddev) = self.netG.decoder(fake_emb, source_s_emb)
			c_kld = torch.mean(-c_z_log_stddev + 0.5 * (torch.exp(2 * c_z_log_stddev) + torch.pow(c_z_mean, 2) - 1))
			cycle_loss = self.criterionL1Pixel(cycle_recon, src_img)
			errG['g_cycle'] = cycle_loss.item()
		else:
			cycle_loss = 0
			c_kld = 0
			
			
		# Reconstruction for matching case
		recon, (r_z_mean, r_z_log_stddev) = self.netG.decoder(src_img_emb, source_s_emb)
		r_kld = torch.mean(-r_z_log_stddev + 0.5 * (torch.exp(2 * r_z_log_stddev) + torch.pow(r_z_mean, 2) - 1))
		recon_loss = self.criterionL1Pixel(recon, src_img)
		errG['g_recon'] = recon_loss.item()
		
		fake_c_loss_total = self.cfg.lambda_cond_loss * fake_c_loss
		cycle_loss_total = self.cfg.lambda_cycle * cycle_loss
		recon_loss_total = self.cfg.lambda_rec * recon_loss
		attn_loss_total = self.cfg.lambda_attn * fake_attn_loss
		kld_total = 0.5 * (kld + c_kld + r_kld)
		g_loss = fake_loss + fake_c_loss_total + cycle_loss_total + recon_loss_total + attn_loss_total + kld_total
		
		self.optimizer_G.zero_grad()
		g_loss.backward()
		self.optimizer_G.step()
		
		
		self.learning_rate['lr'] = self.old_lr
		self.learning_rate['g_lr'] = self.G_lr
		
		return_text = []
		for i, cap in enumerate(src_cap):
			mask = int(src_mask[i].sum())
			cap_cut = cap[:mask]
			text = self.tokenize.decode(cap_cut)
			return_text.append(text)
		
		if self.cfg.CYCLEloss:
			return return_text[:4], errG, recon[:4, :, :, :], cycle_recon[:4,:,:,:]
		else:
			return return_text[:4], errG, recon[:4, :, :, :]
	
	def train_D(self, batch):
		src_img, src_cap, src_cls_id = batch
		errD = {}
		
		# Text data process
		src_encoding = self.tokenize(list(src_cap), padding='max_length', truncation=True, max_length=self.cfg.max_len,return_tensors="pt")
		src_cap, src_mask, src_ids = src_encoding["input_ids"].to(self.device), src_encoding["attention_mask"].to(self.device) , src_encoding["token_type_ids"].to(self.device)
		
		# word for D
		D_src_words = self.netD.text_init(src_cap, src_mask, src_ids)
		source_w_emb = self.netD.text_encoder(D_src_words)
		# sent for G
		G_src_words = self.netG.text_init(src_cap, src_mask, src_ids)
		source_s_emb = self.netG.text_encoder(G_src_words)
		
		n_s_embs = torch.cat((source_s_emb[-1:, :], source_s_emb[:-1, :]), 0)
		n_w_embs = torch.cat((source_w_emb[-1:, :, :], source_w_emb[:-1, :, :]), 0)
		n_mask = torch.cat((src_mask[-1:, :], src_mask[:-1, :]), 0)
		
		
		
		# Image data process
		src_img = src_img.to(self.device)
		
		# Real
		src_features = self.netD.encoder(src_img)
		real_logit = self.netD.adv(src_features[-1])
		real_loss = F.binary_cross_entropy_with_logits(real_logit, self.ones_like(real_logit))
		
		real_c_prob, real_c_prob_n = self.netD.TaGAN_D(src_features, source_w_emb, src_mask, negative=True)
		real_c_loss = (F.binary_cross_entropy(real_c_prob, self.ones_like(real_c_prob)) + \
		               F.binary_cross_entropy(real_c_prob_n, self.zeros_like(real_c_prob_n))) / 2
		
		errD['d_real_loss'] = real_loss.item()
		errD['d_real_c_loss'] = real_c_loss.item()
		
		
		# Fake
		src_img_emb = self.netG.encoder(src_img)
		fake, _ = self.netG.decoder(src_img_emb.detach(), n_s_embs)
		fake_features = self.netD.encoder(fake.detach())
		fake_logit = self.netD.adv(fake_features[-1])
		
		fake_loss = F.binary_cross_entropy_with_logits(fake_logit, self.zeros_like(fake_logit))
		errD['d_fake_loss'] = fake_loss.item()
		
		if self.cfg.ATTNloss:
			tar_region_word_attn, tar_region_word_attn_label = self.netD.word_region(
				fake_features[0], src_features[0], source_w_emb, n_w_embs, src_mask, n_mask)
			fake_attn_loss = self.criterionL1Pixel(tar_region_word_attn, tar_region_word_attn_label)
			errD['d_fake_attn_loss'] = fake_attn_loss.item()
		else:
			fake_attn_loss = 0
		
		d_loss = real_loss + (real_c_loss * self.cfg.lambda_cond_loss) + fake_loss + self.cfg.lambda_attn * fake_attn_loss
		self.optimizer_D.zero_grad()
		d_loss.backward()
		self.optimizer_D.step()
		
		self.learning_rate['lr'] = self.old_lr
		self.learning_rate['d_lr'] = self.D_lr
		return src_img[:4, :, :, :], errD, fake[:4, :, :, :]
	
	def eval_G(self, batch):
		self.eval()
		src_img, source_cap, target_cap = batch
		
		# Text data preprocess
		src_encoding = self.tokenize(list(source_cap), padding='max_length', truncation=True, max_length=self.cfg.max_len,
		                             return_tensors="pt")
		tar_encoding = self.tokenize(list(target_cap), padding='max_length', truncation=True, max_length=self.cfg.max_len,
		                             return_tensors="pt")
		src_img = src_img.to(self.device)
		src_cap, src_mask, src_ids = src_encoding["input_ids"].to(self.device), src_encoding["attention_mask"].to(
			self.device), src_encoding["token_type_ids"].to(self.device)
		tar_cap, tar_mask, tar_ids = tar_encoding["input_ids"].to(self.device), tar_encoding["attention_mask"].to(
			self.device), tar_encoding["token_type_ids"].to(self.device)
		
		with torch.no_grad():
			G_src_words = self.netG.text_init(src_cap, src_mask, src_ids)
			G_source_s_emb = self.netG.text_encoder(G_src_words)
			
			G_tar_words = self.netG.text_init(tar_cap, tar_mask, tar_ids)
			G_target_s_emb = self.netG.text_encoder(G_tar_words)
			
		# Image data preprocess
		src_img_emb = self.netG.encoder(src_img)
		
		
		recon,_ = self.netG.decoder(src_img_emb, G_source_s_emb)
		mani, _ = self.netG.decoder(src_img_emb, G_target_s_emb)
		
		self.train()
		return source_cap, target_cap, src_img, recon, mani
		
	
	def label_like(self, label, x):
		assert label == 0 or label == 1
		v = torch.zeros_like(x) if label == 0 else torch.ones_like(x)
		v = v.to(x.device)
		return v
	
	def zeros_like(self, x):
		return self.label_like(0, x)
	
	def ones_like(self, x):
		return self.label_like(1, x)
	
	def train(self):
		self.netG.train()
		self.netD.train()
	
	def eval(self):
		self.netG.eval()
		self.netD.eval()
	
	def _build_optimizer(self):
		G_params = list(self.netG.parameters())
		D_params = list(self.netD.parameters())
		
		beta1, beta2 = self.cfg.beta1, self.cfg.beta2
		self.G_lr, self.D_lr = self.old_lr, self.old_lr
		
		if self.cfg.lr_decay:
			optimizer_G = torch.optim.Adam(G_params, lr=self.G_lr, betas=(beta1, beta2))
			optimizer_D = torch.optim.Adam(D_params, lr=self.D_lr, betas=(beta1, beta2))
			
		self.optimizer_G = optimizer_G
		self.optimizer_D = optimizer_D
	
	def update_learning_rate(self, lr_decay, iter):
		if self.cfg.lr_decay:
			if (iter+1) % self.cfg.niter == 0:
				new_lr = self.old_lr * self.cfg.niter_decay
			else:
				new_lr = self.old_lr
			
			if new_lr != self.old_lr:
				self.G_lr = new_lr
				self.D_lr = new_lr
				
				for param_group in self.optimizer_D.param_groups:
					param_group['lr'] = self.D_lr
				for param_group in self.optimizer_G.param_groups:
					param_group['lr'] = self.G_lr
				
				description = ("=" * 50)
				description += "\n learning rate decay"
				
				description += '\n update learning rate: %f -> %f' % (self.old_lr, new_lr)
				self.old_lr = new_lr
				return description
			else:
				return None
		else:
			return None
	
	def save(self, file):
		print('Saving weights to', file, '...')
		torch.save({
			'netG'       : self.netG.state_dict(),
			'netD'       : self.netD.state_dict(),
			'optimizer_G': self.optimizer_G.state_dict(),
			'optimizer_D': self.optimizer_D.state_dict()
		}, file)
		
	