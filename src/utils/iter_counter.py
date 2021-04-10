import os
import time

import numpy as np


## Helper class that keeps track of training iterations
class IterationCounter():
	def __init__(self, cfg, dataset_size):
		self.cfg = cfg
		self.dataset_size = dataset_size
		
		self.first_epoch = 1
		self.total_epochs = cfg.niter + cfg.niter_decay
		self.epoch_iter = 0  # iter number within each epoch
		self.iter_record_path = os.path.join(self.cfg.checkpoints_dir, self.cfg.name, 'iter.txt')
	
		self.total_steps_so_far = (self.first_epoch - 1) * dataset_size + self.epoch_iter
		
		self.num_gpu = num_gpu
		self.rank = rank
	
	# return the iterator of epochs for the training
	def training_epochs(self):
		return range(self.first_epoch, self.total_epochs + 1)
	
	def record_epoch_start(self, epoch):
		self.epoch_start_time = time.time()
		self.epoch_iter = 0
		self.last_iter_time = time.time()
		self.current_epoch = epoch
	
	def record_one_iteration(self):
		current_time = time.time()
		
		self.time_per_iter = (current_time - self.last_iter_time) / (self.cfg.batchSize * self.num_gpu)
		self.last_iter_time = current_time
		self.total_steps_so_far += self.cfg.batchSize * self.num_gpu
		self.epoch_iter += self.cfg.batchSize * self.num_gpu
	
	def record_iteration_end(self):
		current_time = time.time()
		
		self.model_time_per_iter = (current_time - self.last_iter_time) / (self.cfg.batchSize * self.num_gpu)
	
	def record_epoch_end(self):
		current_time = time.time()
		self.time_per_epoch = current_time - self.epoch_start_time
		if self.rank == 0:
			print('End of epoch %d / %d \t Time Taken: %d sec' %
			      (self.current_epoch, self.total_epochs, self.time_per_epoch))
			if self.current_epoch % self.cfg.save_epoch_freq == 0:
				np.savetxt(self.iter_record_path, (self.current_epoch + 1, 0),
				           delimiter=',', fmt='%d')
				print('Saved current iteration count at %s.' % self.iter_record_path)
	
	def record_current_iter(self):
		if self.rank == 0:
			np.savetxt(self.iter_record_path, (self.current_epoch, self.epoch_iter),
			           delimiter=',', fmt='%d')
			print('Saved current iteration count at %s.' % self.iter_record_path)
	
	def needs_printing(self):
		return (self.total_steps_so_far % self.cfg.print_freq) < self.cfg.batchSize * self.num_gpu
