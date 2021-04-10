import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel
from .ops import ResidualBlock_TaGAN


class TaGANEncoder(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		# encoder
		self.encoder = nn.Sequential(
			nn.Conv2d(3, 64, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 128, 4, 2, padding=1, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 256, 4, 2, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 512, 4, 2, padding=1, bias=False),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True)
		)
	
	def forward(self, img):
		img = self.encoder(img)  # bs, 512, 32, 32
		return img


def GeneratorEncoder(cfg):
	if cfg.generator_encoder == "tagan":
		return TaGANEncoder(cfg)
	else:
		raise Exception('there is no model')


def init_weights(m):
	if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
		if m.weight.requires_grad:
			m.weight.data.normal_(std=0.02)
		if m.bias is not None and m.bias.requires_grad:
			m.bias.data.fill_(0)
	elif isinstance(m, nn.BatchNorm2d) and m.affine:
		if m.weight.requires_grad:
			m.weight.data.normal_(1, 0.02)
		if m.bias.requires_grad:
			m.bias.data.fill_(0)


class TaGANDecoder(nn.Module):
	def __init__(self, cfg):
		super(TaGANDecoder, self).__init__()
		self.cfg = cfg
		self.w_dim = self.cfg.w_dim
		self.z_dim = self.cfg.z_dim
		
		# residual blocks
		self.residual_blocks = nn.Sequential(
			nn.Conv2d(512 + self.z_dim, 512, 3, padding=1, bias=False),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
			ResidualBlock_TaGAN(512),
			ResidualBlock_TaGAN(512),
			ResidualBlock_TaGAN(512),
			ResidualBlock_TaGAN(512)
		)
		
		# decoder
		self.decoder = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.Conv2d(512, 256, 3, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.Conv2d(256, 128, 3, padding=1, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.Conv2d(128, 64, 3, padding=1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 3, 3, padding=1),
			nn.Tanh()
		)
		# conditioning augmentation
		self.mu = nn.Sequential(
			nn.Linear(self.w_dim, self.z_dim),
			nn.LeakyReLU(0.2, inplace=True)
		)
		self.log_sigma = nn.Sequential(
			nn.Linear(self.w_dim, self.z_dim),
			nn.LeakyReLU(0.2, inplace=True)
		)
		self.apply(init_weights)
	
	def forward(self, img_feat, sent):
		z_mean = self.mu(sent)
		z_log_stddev = self.log_sigma(sent)
		z = torch.randn(sent.size(0), self.z_dim, device=img_feat.device)
		cond = z_mean + z_log_stddev.exp() * z
		cond = cond.unsqueeze(-1).unsqueeze(-1)
		merge = self.residual_blocks(
			torch.cat((img_feat, cond.repeat(1, 1, img_feat.size(2), img_feat.size(3))), 1))  # 4 512 32 32
		
		# decoder
		d = self.decoder(img_feat + merge)
		return d, (z_mean, z_log_stddev)


def GeneratorDecoder(cfg):
	if cfg.generator_decoder == 'tagan':
		return TaGANDecoder(cfg)
	else:
		raise Exception('there is no model')


class G_BERT(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.bert = BertModel.from_pretrained(self.cfg.bert_pretrain)
		if not self.cfg.TEXTtune:
			for param in self.bert.parameters():
				param.requires_grad = False
	
	def forward(self, text, mask=None, type_id=None):
		# text = [batch size, sent len]
		embedded = self.bert(input_ids=text, attention_mask=mask, token_type_ids=type_id)[0]
		return embedded


class G_text(nn.Module):
	def __init__(self, cfg):
		super(G_text, self).__init__()
		self.cfg = cfg
		self.w_dim = self.cfg.w_dim
		self.fc = nn.Sequential(
			nn.Linear(self.w_dim, self.w_dim),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.5),
		)
	
	def forward(self, text_emb):
		words = self.fc(text_emb)
		sent = words[:, 0, :]
		
		return sent

class Generator(nn.Module):
	def __init__(self, cfg):
		super(Generator, self).__init__()
		self.cfg = cfg
		self.encoder = GeneratorEncoder(self.cfg)
		self.decoder = GeneratorDecoder(self.cfg)
		self.text_init = G_BERT(self.cfg)
		self.text_encoder = G_text(self.cfg)
