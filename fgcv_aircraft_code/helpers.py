# NAI

# Helper file for very general functions like saving checkpoints and running a test pass

import numpy as np
import random
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.autograd.gradcheck import zero_gradients
from collections import defaultdict
import random
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Save model checkpoints
def save_checkpoint(state, is_best, checkpoint_prefix):
	filepath = checkpoint_prefix+"_checkpoint.pth.tar"
	torch.save(state, filepath)
	if is_best:
		print("New best file! Saving.")
		shutil.copyfile(filepath, checkpoint_prefix+'_checkpoint_best.pth.tar')

def create_learning_rate_table(initial_lr, decay_schedule, gamma, epochs):
	lr_table = np.zeros((epochs))
	prev_lr = initial_lr
	for i in range(epochs):
		if i in decay_schedule:
			prev_lr *= gamma
		lr_table[i] = prev_lr
	return lr_table

def adjust_learning_rate(optimizer, curr_epoch, learning_rate_table):
	for param_group in optimizer.param_groups:
		param_group['lr'] = learning_rate_table[curr_epoch]

# Test the input model on data from the loader. Used in training script
def test_model(net,device,loader,mean,std):
	net.eval()
	# Stat keepers
	running_clean_correct = 0.
	running_clean_loss = 0.
	running_total = 0.
	with torch.no_grad():
		#for batch_idx,(data,labels) in enumerate(loader):
		for batch_idx,(pkg) in enumerate(loader):
			data = pkg[0].to(device); labels = pkg[1].to(device)
			#plt.figure(figsize=(20,8))
			#plt.subplot(1,6,1);plt.imshow(data[0].cpu().numpy().squeeze().transpose(1,2,0)); plt.axis("off")
			#plt.subplot(1,6,2);plt.imshow(data[1].cpu().numpy().squeeze().transpose(1,2,0)); plt.axis("off")
			#plt.subplot(1,6,3);plt.imshow(data[2].cpu().numpy().squeeze().transpose(1,2,0)); plt.axis("off")
			#plt.subplot(1,6,4);plt.imshow(data[3].cpu().numpy().squeeze().transpose(1,2,0)); plt.axis("off")
			#plt.subplot(1,6,5);plt.imshow(data[4].cpu().numpy().squeeze().transpose(1,2,0)); plt.axis("off")
			#plt.subplot(1,6,6);plt.imshow(data[5].cpu().numpy().squeeze().transpose(1,2,0)); plt.axis("off")
			#plt.tight_layout(); plt.show()
			#exit()
			clean_outputs = net((data-mean)/std)
			clean_loss = F.cross_entropy(clean_outputs, labels)
			_,clean_preds = clean_outputs.max(1)
			running_clean_correct += clean_preds.eq(labels).sum().item()
			running_clean_loss += clean_loss.item()
			running_total += labels.size(0)
		clean_acc = running_clean_correct/running_total
		clean_loss = running_clean_loss/running_total
	net.train()
	return clean_acc,clean_loss

"""
def split_dataset_into_trainval(dset, num_val_per_class):

	# Go through dset and record the indices of each unique class
	dset_info = defaultdict(list)
	for idx in range(len(dset)):
		cls = dset.imgs[idx][1]
		dset_info[cls].append(idx)

	# Split each class into train/val components
	traininds = []; valinds = []
	for cls in dset_info.keys():
		tmp_inds = random.sample(list(range(len(dset_info[cls]))), num_val_per_class)
		for i in range(len(dset_info[cls])):
			if i in tmp_inds:
				valinds.append(dset_info[cls][i])
			else:
				traininds.append(dset_info[cls][i])
		print("Class: {}\t# train: {}\t#val: {}".format(cls, len(dset_info[cls])-num_val_per_class, num_val_per_class))
	assert(set(valinds).isdisjoint(traininds))
	return traininds, valinds
"""



