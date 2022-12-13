# NAI

# This file contains helper functions specifically for training a DNN on in-distribution
#    data from some training dataloader. 


from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients

device = 'cuda' if torch.cuda.is_available() else 'cpu'

############################################################################################################
#### LABEL SMOOTHING STUFF
# Transform the true "Long" labels to softlabels. The confidence of the gt class is 
#  1-smoothing, and the rest of the probability (i.e. smoothing) is uniformly distributed
#  across the non-gt classes. Note, this is slightly different than standard smoothing
#  notation.  
def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
	# if smoothing == 0, it's one-hot method
	# if 0 < smoothing < 1, it's smooth method
	assert 0 <= smoothing < 1
	confidence = 1.0 - smoothing
	label_shape = torch.Size((true_labels.size(0), classes))
	with torch.no_grad():
		true_dist = torch.empty(size=label_shape, device=true_labels.device)
		true_dist.fill_(smoothing / (classes - 1))
		true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
	return true_dist.float()
def xent_with_soft_targets(logit_preds, targets):
	logsmax = F.log_softmax(logit_preds, dim=1)
	batch_loss = targets * logsmax
	batch_loss =  -1*batch_loss.sum(dim=1)
	return batch_loss.mean()
def xent_with_soft_targets_noreduce(logit_preds, targets):
	logsmax = F.log_softmax(logit_preds, dim=1)
	batch_loss = targets * logsmax
	batch_loss =  -1*batch_loss.sum(dim=1)
	return batch_loss

"""
############################################################################################################
#### MIXUP STUFF
def mixup_data(x, y, alpha=1.0, use_cuda=True):
	# Returns mixed inputs, pairs of targets, and lambda
	if alpha > 0:
		lam = np.random.beta(alpha, alpha)
	else:
		lam = 1
	batch_size = x.size()[0]
	if use_cuda:
		index = torch.randperm(batch_size).cuda()
	else:
		index = torch.randperm(batch_size)
	mixed_x = lam * x + (1 - lam) * x[index, :]
	y_a, y_b = y, y[index]
	return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
	return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
"""
"""
############################################################################################################
#### ADVERSARIAL TRAINING STUFF

# Gradient - Forward pass data through model and return gradient w.r.t. data
# model - pytorch model to be used for forward pass
# device - device the model is running on
# data -  NCHW tensor of range [0.,1.]
# lbl - label to calculate loss against (usually GT label or target label for targeted attacks)
# returns gradient of loss w.r.t data
# Note: It is important this is treated as an atomic operation because of the
#       normalization. We carry the data around unnormalized, so in this fxn we
#       normalize, forward pass, then unnorm the gradients before returning. Any
#       fxn that uses this method should handle the data in [0,1] range
def gradient_wrt_data(model,device,data,lbl):
	# Manually Normalize
	mean = torch.tensor([0.5], dtype=torch.float32).view([1,1,1]).to(device)
	std = torch.tensor([0.5], dtype=torch.float32).view([1,1,1]).to(device)
	dat = (data-mean)/std
	# Forward pass through the model
	dat.requires_grad = True
	out = model(dat)
	# Calculate loss
	loss = F.cross_entropy(out,lbl)
	# zero all old gradients in the model
	model.zero_grad()
	# Back prop the loss to calculate gradients
	loss.backward()
	# Extract gradient of loss w.r.t data
	data_grad = dat.grad.data
	# Unnorm gradients back into [0,1] space
	#   As shown in foolbox/models/base.py
	grad = data_grad / std
	return grad.data.detach()

def gradient_wrt_data_OE(model,device,data,lbl):
	# Manually Normalize
	mean = torch.tensor([0.5], dtype=torch.float32).view([1,1,1]).to(device)
	std = torch.tensor([0.5], dtype=torch.float32).view([1,1,1]).to(device)
	dat = (data-mean)/std
	# Forward pass through the model
	dat.requires_grad = True
	out = model(dat)
	# Calculate loss
	loss =  xent_with_soft_targets(out, lbl)
	# zero all old gradients in the model
	model.zero_grad()
	# Back prop the loss to calculate gradients
	loss.backward()
	# Extract gradient of loss w.r.t data
	data_grad = dat.grad.data
	# Unnorm gradients back into [0,1] space
	#   As shown in foolbox/models/base.py
	grad = data_grad / std
	return grad.data.detach()

# Projected Gradient Descent Attack (PGD) with random start
def PGD_Linf_attack(model, device, dat, lbl, eps, alpha, iters, withOE=False):
	x_nat = dat.clone().detach()
	# Randomly perturb within small eps-norm ball
	x_adv = dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).to(device)
	x_adv = torch.clamp(x_adv,0.,1.) # respect image bounds
	# Iteratively Perturb data	
	for i in range(iters):
		zero_gradients(x_adv)
		model.zero_grad()		
		# Calculate gradient w.r.t. data
		grad=None
		if withOE:
			grad = gradient_wrt_data_OE(model,device,x_adv.clone().detach(),lbl.clone().detach())
		else:
			grad = gradient_wrt_data(model,device,x_adv.clone().detach(),lbl)
		# Perturb by the small amount a
		x_adv = x_adv + alpha*grad.sign()
		# Clip the perturbations w.r.t. the original data so we still satisfy l_infinity
		x_adv = torch.max(torch.min(x_adv, x_nat+eps), x_nat-eps)
		# Make sure we are still in bounds
		x_adv = torch.clamp(x_adv, 0., 1.)
	return x_adv.data.clone().detach()
"""


############################################################################################################
#### ONE EPOCH OF CLASSIFIER MODEL TRAINING
def train_model(net, optimizer, trainloader, num_classes, norm_mean, norm_std, lblsm=None):

	net.train()

	#AT_EPS = None
	#AT_EPS = 2./255.; AT_ALPHA = 0.5/255.; AT_ITERS = 7
	#AT_EPS = 4./255.; AT_ALPHA = 1./255. ; AT_ITERS = 7
	#AT_EPS = 8./255.; AT_ALPHA = 2./255. ; AT_ITERS = 7

	running_correct = 0.
	running_total = 0.
	running_loss_sum = 0.

	for batch_idx,(pkg) in enumerate(trainloader):
		data = pkg[0].to(device); labels = pkg[1].to(device)

		# ADVERSARIALLY PERTURB DATA
		#if AT_EPS is not None:
		#	exit("AT not checked yet")
		#	data = PGD_Linf_attack(net, device, data.clone().detach(), labels, eps=AT_EPS, alpha=AT_ALPHA, iters=AT_ITERS)

		# Plot some training samples	
		#plt.figure(figsize=(10,3))
		#plt.subplot(1,6,1);plt.imshow(data[0].cpu().numpy().squeeze().transpose(1,2,0)); plt.axis("off")
		#plt.subplot(1,6,2);plt.imshow(data[1].cpu().numpy().squeeze().transpose(1,2,0)); plt.axis("off")
		#plt.subplot(1,6,3);plt.imshow(data[2].cpu().numpy().squeeze().transpose(1,2,0)); plt.axis("off")
		#plt.subplot(1,6,4);plt.imshow(data[3].cpu().numpy().squeeze().transpose(1,2,0)); plt.axis("off")
		#plt.subplot(1,6,5);plt.imshow(data[4].cpu().numpy().squeeze().transpose(1,2,0)); plt.axis("off")
		#plt.subplot(1,6,6);plt.imshow(data[5].cpu().numpy().squeeze().transpose(1,2,0)); plt.axis("off")
		#plt.show()
		#exit()

		# Forward pass data through model. Normalize before forward pass
		outputs = net((data-norm_mean)/norm_std)

		loss = None
		if lblsm is None:
			# VANILLA CROSS-ENTROPY
			loss = F.cross_entropy(outputs, labels);
		else:		
			# LABEL SMOOTHING LOSS
			sl = smooth_one_hot(labels,num_classes,smoothing=lblsm)
			loss =  xent_with_soft_targets(outputs, sl)

		# Calculate gradient and update parameters
		optimizer.zero_grad()
		net.zero_grad()
		loss.backward()
		optimizer.step()

		# Measure accuracy and loss for this batch
		_,preds = outputs.max(1)
		running_total += labels.size(0)
		running_correct += preds.eq(labels).sum().item()
		running_loss_sum += loss.item()		

		#if batch_idx % 10 == 0:
		#	print("[{}/{}] loss: {}, acc: {}".format(batch_idx,len(trainloader),running_loss_sum/(batch_idx+1), running_correct/running_total))
		#if batch_idx == 30:
		#	break

	# Return train_acc, train_loss, %measured_traindata
	return running_correct/running_total, running_loss_sum/running_total

"""
def train_model_outlierexposure(net, optimizer, idtrainloader, oetrainloader, num_classes, norm_mean, norm_std):

	net.train()
	AT_EPS = None

	### DATA AUGMENTATION AND LOSS FXN CONFIGS
	OE_LAMBDA=0.5
	#AT_EPS = 2./255.; AT_ALPHA = 0.5/255.; AT_ITERS = 7
	#AT_EPS = 4./255.; AT_ALPHA = 1./255. ; AT_ITERS = 7
	#AT_EPS = 8./255.; AT_ALPHA = 2./255. ; AT_ITERS = 7
	#AT_EPS = 10./255.; AT_ALPHA = 2.5/255. ; AT_ITERS = 7

	running_correct = 0.
	running_total = 0.
	running_loss_sum = 0.
	running_real_cnt = 0.

	batch_idx = -1
	#for batch_idx,(data,labels,pth) in enumerate(trainloader):
	for in_set, out_set in zip(idtrainloader,oetrainloader):
		batch_idx += 1
		idata,ilabels = in_set
		odata = out_set[0]

		# Plot some samples from the OE set
		#plt.figure(figsize=(15,9))
		#for jj in range(15):
		#	plt.subplot(3,5,jj+1); plt.imshow(idata[jj].cpu().squeeze().numpy(),cmap='gray'); plt.axis("off")
		#	plt.subplot(3,5,jj+1); plt.imshow(odata[jj].cpu().squeeze().numpy(),cmap='gray'); plt.axis("off")
		#plt.tight_layout(); plt.show(); exit("BREAK")

		onehot_ilabels = smooth_one_hot(ilabels,num_classes,smoothing=0.)
		uniform_olabels = torch.zeros((odata.size(0),num_classes)).fill_(1./num_classes)

		full_data = torch.cat((idata,odata),0)
		full_labels = torch.cat((onehot_ilabels,uniform_olabels),0)
		full_data = full_data.to(device); full_labels = full_labels.to(device); ilabels=ilabels.to(device)

		# ADVERSARIALLY PERTURB DATA
		if AT_EPS is not None:
			exit("AT not checked yet")
			full_data = PGD_Linf_attack(net, device, full_data.clone().detach(), full_labels.clone().detach(), eps=AT_EPS, alpha=AT_ALPHA, iters=AT_ITERS, withOE=True)

		# Forward pass data through model. Normalize before forward pass
		outputs = net((full_data-norm_mean)/norm_std)

		# CROSS-ENTROPY
		batch_loss =  xent_with_soft_targets_noreduce(outputs, full_labels)
		loss = batch_loss[:ilabels.size(0)].mean() + OE_LAMBDA*batch_loss[ilabels.size(0):].mean()

		# Calculate gradient and update parameters
		optimizer.zero_grad()
		net.zero_grad()
		loss.backward()
		optimizer.step()

		# Measure accuracy and loss for this batch
		_,preds = outputs.max(1)
		preds = preds[:ilabels.size(0)]
		running_total += ilabels.size(0)
		running_correct += preds.eq(ilabels).sum().item()
		running_loss_sum += loss.item()		

		#if batch_idx % 10 == 0:
		#	print("{} / {}".format(batch_idx, len(idtrainloader)))

	return running_correct/running_total, running_loss_sum/running_total
"""

