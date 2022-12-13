# NAI

# Main file for OOD detection with the FGCV-Aircraft dataset

from __future__ import print_function
import numpy as np
import sys
import os
from collections import defaultdict
import random
#import timm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as utilsdata
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as tvdatasets
import torchvision.models as tvmodels
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Custom
import Dataset_fromTxtFile as txtfiledataset
import helpers
import training_helpers
#import mixupoe_training_helpers
#import ood_helpers
#import calculate_log as callog
#import my_models

# Constants
cudnn.benchmark = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
DSIZE = 224


# TODO: Move this out of here
def compute_average_OOD_stats(id_scores, ood_scores, num_splits=10):
	# Calculate how many id/ood samples we will compute metrics against
	sample_size = min(len(id_scores), len(ood_scores))
	# initialize some stat keepers
	auroc = []; tnr = []; dtacc = []; auprin = []; auprout = []
	# Compute OOD stats over a few random splits
	for i in range(num_splits):
		# Calculate OOD stats for this split
		metric_results = callog.metric( np.array(random.sample(id_scores, sample_size)) , np.array(random.sample(ood_scores, sample_size)) )
		# update stat keepers for auroc, tnr, aupr
		auroc.append(metric_results['TMP']['AUROC'])
		tnr.append( metric_results['TMP']['TNR'])
		dtacc.append( metric_results['TMP']['DTACC'])
		auprin.append(metric_results['TMP']['AUIN'])
		auprout.append(metric_results['TMP']['AUOUT'])
	# compute and print average detection rates for this class
	auroc = np.array(auroc)
	tnr = np.array(tnr)
	dtacc = np.array(dtacc)
	auprin = np.array(auprin)
	auprout = np.array(auprout)
	return auroc.mean(), tnr.mean(), dtacc.mean(), auprin.mean(), auprout.mean()


#####################################################################################################################
# Inputs and Initializations
#####################################################################################################################

# Pick the split
dataset_root = "/raid/inkawhna/WORK/fgvc-aircraft-2013b/data/splits/split1"; NUM_ID_CLASSES=79
#dataset_root = "/raid/inkawhna/WORK/fgvc-aircraft-2013b/data/splits/split2"; NUM_ID_CLASSES=80
#dataset_root = "/raid/inkawhna/WORK/fgvc-aircraft-2013b/data/splits/split3"; NUM_ID_CLASSES=78
#dataset_root = "/raid/inkawhna/WORK/fgvc-aircraft-2013b/data/splits/split4"; NUM_ID_CLASSES=80

DATASETS = ["ID","aircraft_ood"]

# Number of separate models to train and evaluate
REPEAT_ITERS = 4

SEED = 2740572 # Tester
#SEED = 15726423
#SEED = 23740567
#SEED = 37493712
#SEED = 74057289


# FGCV-Aircraft Training Params
num_epochs = 100
batch_size = 32
learning_rate = 0.001
learning_rate_decay_schedule = [30,60,90]
gamma = 0.1

# Use ImageNet normalization
MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view([3,1,1]).to(device)
STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view([3,1,1]).to(device)


#### Initialize stat keepers
STAT_accuracy = []
STAT_ood_baseline = {}
STAT_ood_odin = {}
#STAT_ood_mahala = {}

for dd in range(1,len(DATASETS)):
	STAT_ood_baseline[DATASETS[dd]] = defaultdict(list)
	STAT_ood_odin[DATASETS[dd]] = defaultdict(list)
#	STAT_ood_mahala[DATASETS[dd]] = defaultdict(list)


#####################################################################################################################
# Create dataset splits
#####################################################################################################################
random.seed(SEED)
torch.manual_seed(SEED)
print("Creating Datasets with SEED = ",SEED)

transform_train = transforms.Compose([
	#transforms.RandomResizedCrop(224,scale=(0.08,1.0),ratio=(0.75,1.3333)),
	transforms.Resize((256,256)),#JYZ
	transforms.RandomCrop((224,224)),#JYZ
	transforms.RandomHorizontalFlip(p=0.5),
	#transforms.ColorJitter(brightness=(0.6,1.4), contrast=(0.6,1.4), saturation=(0.6,1.4)),
	transforms.ColorJitter(brightness=32./255., saturation=0.5),
	transforms.ToTensor(),
]) 
transform_test = transforms.Compose([
	#transforms.Resize(224),
	transforms.Resize((256,256)),
	transforms.CenterCrop((224,224)),
	transforms.ToTensor(),
])  

# Construct dataloaders for the In-Distribution data
ID_trainloader = utilsdata.DataLoader(
	txtfiledataset.Dataset_fromTxtFile(dataset_root+"/ID_trainset.txt", transform_train, is_fgcvaircraft=True),
	batch_size=batch_size,shuffle=True,num_workers=4,timeout=1000,
)
ID_testloader = utilsdata.DataLoader(
	txtfiledataset.Dataset_fromTxtFile(dataset_root+"/ID_testset.txt", transform_test, is_fgcvaircraft=True),
	batch_size=batch_size,shuffle=False,num_workers=2,timeout=1000,
)

# Create Outlier Exposure Dataset
#OE_trainloader  = torch.utils.data.DataLoader(
#	tvdatasets.ImageFolder("/zero1/data1/imagenet21K/splits/imagenet21k_NOAIRPLANES", transform_train), # ImageNet21K
#	#tvdatasets.ImageFolder("/bigdata/NFTI/datasets/Places365/places365_standard/train", transform_train), # Places365
#	batch_size=2*batch_size,shuffle=True,num_workers=4,timeout=1600
#)


#####################################################################################################################
# Main Loop over classifers - for each of these loops we train a model from scratch on the sample
#    data then test its OOD detection performance
#####################################################################################################################

for ITER in range(REPEAT_ITERS):

	print("**********************************************************")
	print("Starting Iter: {} / {}".format(ITER,REPEAT_ITERS))
	print("**********************************************************")

	#################################################################################################################
	# Load Model


	## From scratch 
	#net = tvmodels.resnet34(pretrained=False, num_classes=NUM_ID_CLASSES).to(device)
	#net = tvmodels.resnet50(pretrained=False, num_classes=NUM_ID_CLASSES).to(device)

	## Use ImageNet pre-trained
	#net = tvmodels.resnet34(pretrained=True).to(device)
	#net.fc = nn.Linear(512, NUM_ID_CLASSES).to(device)
	net = tvmodels.resnet50(pretrained=True).to(device)
	net.fc = nn.Linear(2048, NUM_ID_CLASSES).to(device)

	# Optimizer
	#optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
	optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.00001)
	learning_rate_table = helpers.create_learning_rate_table(learning_rate,learning_rate_decay_schedule,gamma,num_epochs)



	#################################################################################################################
	# Training Loop

	final_test_acc = 0.
	for epoch in range(num_epochs):

		# Decay learning rate according to decay schedule 
		helpers.adjust_learning_rate(optimizer, epoch, learning_rate_table)
		print("Starting Epoch {}/{}. lr = {}".format(epoch,num_epochs,learning_rate_table[epoch]))

		# Train
		train_acc, train_loss = training_helpers.train_model(net, optimizer, ID_trainloader, NUM_ID_CLASSES, MEAN, STD, lblsm=None)
		###train_acc, train_loss = training_helpers.train_model_outlierexposure(net, optimizer, ID_trainloader, OE_trainloader, NUM_ID_CLASSES, MEAN, STD)
		###train_acc, train_loss, _ = mixupoe_training_helpers.train_model_mixupoutlierexposure(net, optimizer, ID_trainloader, OE_trainloader, NUM_ID_CLASSES, MEAN, STD)
		print("[{}] Epoch [ {} / {} ]; lr: {} TrainAccuracy: {} TrainLoss: {}".format(ITER,epoch,num_epochs,learning_rate_table[epoch],train_acc,train_loss))

		# Test
		test_acc,test_loss = helpers.test_model(net,device,ID_testloader,MEAN,STD)
		print("\t[{}] Epoch [ {} / {} ]; TestAccuracy: {} TestLoss: {}".format(ITER,epoch,num_epochs,test_acc,test_loss))
		final_test_acc = test_acc

	STAT_accuracy.append(final_test_acc)
	net.eval()
	exit("burp")

	# Optional
	#helpers.save_checkpoint({'test_acc': final_test_acc,'state_dict': net.state_dict()}, False, "checkpoints/rn34_standard_mlrsnetv3_seed{}_iter{}".format(SEED,ITER))
	#helpers.save_checkpoint({'test_acc': final_test_acc,'state_dict': net.state_dict()}, False, "checkpoints/rn34_OEfMoW_mlrsnetv3_seed{}_iter{}".format(SEED,ITER))
	#helpers.save_checkpoint({'test_acc': final_test_acc,'state_dict': net.state_dict()}, False, "checkpoints/rn34_MixupOEfMoW_mlrsnetv3_seed{}_iter{}".format(SEED,ITER))
	#continue

	#################################################################################################################
	# Test ID and OOD data

	### Hook into Net for Mahalanobis layer
	"""
	MAHALA_LAYER_SET = [ 
		#"resnet18_2",
		#"resnet18_2_2",
		#"resnet18_2_2_2",
		#"resnet18_2_2_2_2",
		#"resnet34_3",
		#"resnet34_3_4",
		#"resnet34_3_4_6",
		"resnet34_3_4_6_3",
	]

	hooked_activation_dict={}; hook_handles = []
	def get_activation(name):
		def hook(module, input, output):
			hooked_activation_dict[name]=output
		return hook
	for l in MAHALA_LAYER_SET:
		hook_handles.append(ood_helpers.set_model_hook(l, net, get_activation(l), device))
	with torch.no_grad():
		dd = torch.zeros(1,3,224,224).uniform_().to(device)
		net(dd)
		dd = None
	for l in MAHALA_LAYER_SET:
		print("Hooked Layer: {}; Shape: {}; Min: {}; Max: {}".format(l,hooked_activation_dict[l].shape,hooked_activation_dict[l].min(),hooked_activation_dict[l].max()))

	### Calculate means and covariance stats for Mahalanobis detector

	layer_means, layer_precisions = ood_helpers.compute_empirical_means_and_precision(net, hooked_activation_dict, MAHALA_LAYER_SET, ID_trainloader, NUM_ID_CLASSES, MEAN, STD)
	"""

	### Measure OOD scores with state-of-the-art methods

	baseline_scores_dict = {}
	baseline_classes_dict = {}
	odin_scores_dict = {}
	#odin_ipp_scores_dict = {}
	#mahala_scores_dict = {}
	#mahala_classes_dict = {}
	#mahala_ipp_scores_dict = {}

	for dset in DATASETS:
		print("Computing OOD scores for {}...".format(dset))
		currloader = None
		if dset == "ID":
			currloader = ID_testloader
		else:
			currloader = ood_helpers.get_ood_dataloader_AERIALIMAGERY(dset,DSIZE,bsize=128,shuf=True)

		##### COMPUTE OOD SCORES FOR THIS DATASET
		#base_scores,odin_scores,odin_ipp_scores = ood_helpers.generate_ood_scores_ODIN_and_BASELINE(net, currloader, 1000., MEAN, STD, IPP=True, eps=0.01)
		#mahalanobis_scores,mahalanobis_ipp_scores = ood_helpers.generate_ood_scores_MAHALANOBIS(net, currloader, hooked_activation_dict, MAHALA_LAYER_SET, layer_means, layer_precisions, MEAN, STD, IPP=True, eps=0.01)
		base_scores,odin_scores,_,base_classes = ood_helpers.generate_ood_scores_ODIN_and_BASELINE(net, currloader, 1000., MEAN, STD, IPP=False)
		#mahalanobis_scores,_,mahalanobis_classes = ood_helpers.generate_ood_scores_MAHALANOBIS(net, currloader, hooked_activation_dict, MAHALA_LAYER_SET, layer_means, layer_precisions, MEAN, STD, IPP=False)
		#####

		# Save raw OOD scores into dictionaries
		baseline_scores_dict[dset] = base_scores
		baseline_classes_dict[dset] = base_classes
		odin_scores_dict[dset] = odin_scores
		#odin_ipp_scores_dict[dset] = odin_ipp_scores
		#mahala_scores_dict[dset] = mahalanobis_scores
		#mahala_classes_dict[dset] = mahalanobis_classes
		#mahala_ipp_scores_dict[dset] = mahalanobis_ipp_scores

		print("\tDSET: {}\t# scores:{}".format(dset,len(base_scores)))

	####################################################################################################
	# Compute all OOD statistics for this model over all the tested datasets 

	print("Computing OOD Statistics...")
	for dd in range(1,len(DATASETS)):
		print("** DATASET: {} **".format(DATASETS[dd]))
		
		## Baseline
		tmppkg = compute_average_OOD_stats(baseline_scores_dict["ID"].copy(), baseline_scores_dict[DATASETS[dd]].copy(), num_splits=50)
		print("\tBaseline.          AUROC: {:.4f}. TNR@95TPR: {:.4f}. DetAcc: {:.4f}. AUPR-IN: {:.4f}. AUPR-OUT: {:.4f}".format(tmppkg[0],tmppkg[1],tmppkg[2],tmppkg[3],tmppkg[4]))
		STAT_ood_baseline[DATASETS[dd]]["auroc"].append(tmppkg[0])
		STAT_ood_baseline[DATASETS[dd]]["tnr"].append(tmppkg[1])
		STAT_ood_baseline[DATASETS[dd]]["auin"].append(tmppkg[3])
		STAT_ood_baseline[DATASETS[dd]]["auout"].append(tmppkg[4])
		if DATASETS[dd] not in ["places365","random"]:
			compute_per_class_OOD_stats(baseline_scores_dict["ID"] , baseline_scores_dict[DATASETS[dd]], baseline_classes_dict[DATASETS[dd]], num_splits=50, print_prefix="Baseline --" )

		## ODIN
		tmppkg = compute_average_OOD_stats(odin_scores_dict["ID"].copy(), odin_scores_dict[DATASETS[dd]].copy(), num_splits=50)
		print("\tODIN (T=1000).     AUROC: {:.4f}. TNR@95TPR: {:.4f}. DetAcc: {:.4f}. AUPR-IN: {:.4f}. AUPR-OUT: {:.4f}".format(tmppkg[0],tmppkg[1],tmppkg[2],tmppkg[3],tmppkg[4]))
		STAT_ood_odin[DATASETS[dd]]["auroc"].append(tmppkg[0])
		STAT_ood_odin[DATASETS[dd]]["tnr"].append(tmppkg[1])
		STAT_ood_odin[DATASETS[dd]]["auin"].append(tmppkg[3])
		STAT_ood_odin[DATASETS[dd]]["auout"].append(tmppkg[4])
		if DATASETS[dd] not in ["places365","random"]:
			compute_per_class_OOD_stats(odin_scores_dict["ID"] , odin_scores_dict[DATASETS[dd]], baseline_classes_dict[DATASETS[dd]], num_splits=50, print_prefix="ODIN (T=1000) --")


#####################################################################################################################
# Print Final Run Statistics
#####################################################################################################################

# Helper to print min/max/avg/std/len of values in a list
def print_stats_of_list(prefix,dat):
	dat = np.array(dat)
	print("{} Min: {:.4f}; Max: {:.4f}; Avg: {:.4f}; Std: {:.4f}; Len: {}".format(
			prefix, dat.min(), dat.max(), dat.mean(), dat.std(), len(dat))
	)

print("\n\n")
#print("Printing Final Accuracy + OOD Detection stats for {} Runs with K = {} and J = {}".format(REPEAT_ITERS, K, NUM_HOLDOUT_CLASSES))
print("*****************************************************************************")
print("*****************************************************************************")
print("*****************************************************************************")
print()
print_stats_of_list("Accuracy: ",STAT_accuracy)

for dset in DATASETS:
	if dset == "ID":
		continue
	print("*"*70)
	print("* Dataset: {}".format(dset))
	print("*"*70)
	print_stats_of_list("\tBaseline (AUROC):     ",STAT_ood_baseline[dset]["auroc"])
	print_stats_of_list("\tBaseline (TNR@TPR95): ",STAT_ood_baseline[dset]["tnr"])
	print_stats_of_list("\tBaseline (AUIN):      ",STAT_ood_baseline[dset]["auin"])
	print_stats_of_list("\tBaseline (AUOUT):     ",STAT_ood_baseline[dset]["auout"])
	print()
	print_stats_of_list("\tODIN (T=1000) (AUROC):     ",STAT_ood_odin[dset]["auroc"])
	print_stats_of_list("\tODIN (T=1000) (TNR@TPR95): ",STAT_ood_odin[dset]["tnr"])
	print_stats_of_list("\tODIN (T=1000) (AUIN):      ",STAT_ood_odin[dset]["auin"])
	print_stats_of_list("\tODIN (T=1000) (AUOUT):     ",STAT_ood_odin[dset]["auout"])
	print()

exit("DONE")





