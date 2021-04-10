import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from .ops import *


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


class D_encoder(nn.Module):
	def __init__(self, cfg):
		super(D_encoder, self).__init__()
		self.encoder_1 = nn.Sequential(
			nn.Conv2d(3, 64, 4, 2, padding=1),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(64, 128, 4, 2, padding=1, bias=False),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(128, 256, 4, 2, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2, inplace=True)
		)
		self.encoder_2 = nn.Sequential(
			nn.Conv2d(256, 256, 4, 2, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(256, 512, 4, 2, padding=1, bias=False),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2, inplace=True)
		)
		self.encoder_3 = nn.Sequential(
			nn.Conv2d(512, 512, 4, 2, padding=1, bias=False),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2, inplace=True)
		)
	
	def forward(self, img):
		img_feat_1 = self.encoder_1(img)  # bs, 256, 32, 32
		img_feat_2 = self.encoder_2(img_feat_1)  # bs, 512, 8, 8
		img_feat_3 = self.encoder_3(img_feat_2)  # bs, 512, 4, 4
		return img_feat_1, img_feat_2, img_feat_3


class TaGANDiscriminator(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.w_dim = self.cfg.w_dim
		self.eps = self.cfg.eps
		self.softmax = nn.Softmax(-1)
		
		self.gen_filter = nn.ModuleList([
			nn.Linear(self.w_dim, 256 + 1),
			nn.Linear(self.w_dim, 512 + 1),
			nn.Linear(self.w_dim, 512 + 1)
		])
		
		self.gen_weight = nn.Sequential(
			nn.Linear(self.w_dim, 3),
			nn.Softmax(-1)
		)
		
		self.GAP_1 = nn.Sequential(
			nn.Conv2d(256, 256, 3, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2, inplace=True)
		)
		self.GAP_2 = nn.Sequential(
			nn.Conv2d(512, 512, 3, padding=1, bias=False),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2, inplace=True)
		)
		self.GAP_3 = nn.Sequential(
			nn.Conv2d(512, 512, 3, padding=1, bias=False),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2, inplace=True)
		)
		
		self.apply(init_weights)
	
	def forward(self, img_features, txt_w, txt_mask, negative=False):
		img_feats = [self.GAP_1(img_features[0]), self.GAP_2(img_features[1]), self.GAP_3(img_features[2])]
		
		u, alpha = self._encode_txt(txt_w, txt_mask)
		weight = self.gen_weight(u).permute(2, 0, 1)  # 3, bs, seq_len
		
		sim = 0
		sim_n = 0
		idx = np.arange(0, img_feats[0].size(0))
		idx_n = torch.tensor(np.roll(idx, 1), dtype=torch.long, device=txt_w.device)
		
		for i in range(3):
			img_feat = img_feats[i]  # bs 256 16 16
			W_cond = self.gen_filter[i](u)  # bs seq_len w_dim
			W_cond, b_cond = W_cond[:, :, :-1], W_cond[:, :, -1].unsqueeze(-1)
			img_feat = img_feat.mean(-1).mean(-1).unsqueeze(-1)  # bs 256 1
			# img_feat = F.adaptive_avg_pool2d(img_feat,(1,1)).unsqueeze(-1)  # bs 256 1
			
			if negative:
				W_cond_n, b_cond_n, weight_n = W_cond[idx_n], b_cond[idx_n], weight[i][idx_n]
				sim_n += torch.sigmoid(torch.bmm(W_cond_n, img_feat) + b_cond_n).squeeze(-1) * weight_n
			sim += torch.sigmoid(torch.bmm(W_cond, img_feat) + b_cond).squeeze(-1) * weight[i]
		
		if negative:
			alpha_n = alpha.permute(1, 0)[:, idx_n]
			sim_n = torch.clamp(sim_n + self.eps, max=1).t().pow(alpha_n).prod(0)
		sim = torch.clamp(sim + self.eps, max=1).t().pow(alpha.permute(1, 0)).prod(0)
		
		if negative:
			return sim, sim_n
		return sim
	
	def _encode_txt(self, txt, txt_mask):
		"""
		input
		txt:      bs seq_len w_dim
		txt_mask: bs seq_len

		output
		u             bs seq_len w_dim
		att_score:    bs seq_len
		"""
		m = txt[:, :1, :]    # CLS token
		u = txt[:, 1:-1, :]  # except CLS and SEP
		u_mask = txt_mask[:, 1:-1]
		
		att_txt = ((u * u_mask.unsqueeze(-1)) * (m)).sum(-1)
		att_mask = (u_mask.float() - 1.0) * 10000
		att_score = self.softmax(att_txt + att_mask)
		return u, att_score


class D_WORD_REGION(nn.Module):
	def __init__(self, cfg):
		super(D_WORD_REGION, self).__init__()
		self.cfg = cfg
		self.fc_w = nn.Linear(self.cfg.w_dim, 256)
		self.conv = nn.Conv2d(256, 256, 3, 1, 1)
		self.softmax = nn.Softmax(dim=-1)
	
	def forward(self, src_img_feature, tar_img_feature, src_word_embs, tar_word_embs, src_mask, tar_mask):
		N, C, H, W = src_img_feature.shape
		
		src_img_feature = self.conv(src_img_feature)
		tar_img_feature = self.conv(tar_img_feature)
		src_img_feature = src_img_feature.view(N, H * W, C)
		tar_img_feature = tar_img_feature.view(N, H * W, C)
		
		src_word_embs = src_word_embs[:, 1:-1, :] * src_mask[:, 1:-1].unsqueeze(-1)
		tar_word_embs = tar_word_embs[:, 1:-1, :] * tar_mask[:, 1:-1].unsqueeze(-1)
		
		src_word_embs = self.fc_w(src_word_embs)
		tar_word_embs = self.fc_w(tar_word_embs)
		
		src_t_trans = src_word_embs.transpose(1, 2)
		tar_t_trans = tar_word_embs.transpose(1, 2)
		
		src_region_word_attn = torch.matmul(src_img_feature, src_t_trans)
		src_region_word_attn = self.softmax(src_region_word_attn)
		
		src_tar_word_attn = torch.matmul(src_word_embs, tar_t_trans)
		src_tar_word_attn = self.softmax(src_tar_word_attn)
		
		tar_region_word_attn_label = torch.matmul(src_region_word_attn, src_tar_word_attn)
		
		tar_region_word_attn = torch.matmul(tar_img_feature, tar_t_trans)
		tar_region_word_attn = self.softmax(tar_region_word_attn)
		
		return tar_region_word_attn, tar_region_word_attn_label


class D_adv(nn.Module):
	def __init__(self, cfg):
		super(D_adv, self).__init__()
		self.cfg = cfg
		self.classifier = nn.Conv2d(512, 1, 4)
	
	def forward(self, img_feature):
		return self.classifier(img_feature).squeeze(1)


class D_BERT(nn.Module):
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


class D_text(nn.Module):
	def __init__(self, cfg):
		super(D_text, self).__init__()
		self.cfg = cfg
		self.w_dim = self.cfg.w_dim
		self.fc = nn.Sequential(
			nn.Linear(self.w_dim, self.w_dim),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.5),
		)
	
	def forward(self, text_emb):
		words = self.fc(text_emb)
		
		return words

class Discriminator(nn.Module):
	def __init__(self, cfg):
		super(Discriminator, self).__init__()
		self.cfg = cfg
		self.text_init = D_BERT(self.cfg)
		self.text_encoder = D_text(self.cfg)
		self.encoder = D_encoder(self.cfg)
		self.TaGAN_D = TaGANDiscriminator(self.cfg)
		self.word_region = D_WORD_REGION(self.cfg)
		self.adv = D_adv(self.cfg)