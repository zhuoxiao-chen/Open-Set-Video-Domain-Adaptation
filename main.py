# import argparse
import os
import sys
import time
import shutil
import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

from utils.visualization import visualize_TSNE

from scipy.stats import genextreme as gev

from dataset import TSNDataSet, VideoRecord
from models import VideoModel
from loss import *
from opts import parser
from utils.utils import randSelectBatch
import math

from colorama import init
from colorama import Fore, Back, Style
import numpy as np
from tensorboardX import SummaryWriter


# auto hyperparameters tuning
import optuna
from optuna.trial import TrialState
from optuna.samplers import RandomSampler

import logging



os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
torch.backends.cudnn.enabled=False
# torch.cuda.set_device('1')

best_prec1 = 0
gpu_count = torch.cuda.device_count()
gpu_count = 1

def main(trial):

	# To run CEVT model only, and get the similar result in paper,
	# You only need to change code BLOCK 1 for dataset selecting,
	# and BLOCK 2 for hyper-parameters selecting.
	# Please search BLOCK 1 or BLOCK 2 in this file.

	np.random.seed(1)
	torch.manual_seed(1)
	torch.cuda.manual_seed_all(1)

	init(autoreset=True)

	global args, best_prec1, writer
	best_prec1 = 0
	sys.argv += 'data/classInd_ucf_olympic.txt RGB /media/bigdata/uqyluo/MM2020Data/olympic/list_olympic_train_ucf_olympic-feature.txt /media/bigdata/uqyluo/MM2020Data/ucf101/list_ucf101_train_ucf_olympic-feature.txt /media/bigdata/uqyluo/MM2020Data/ucf101/list_ucf101_val_ucf_olympic-feature.txt --exp_path action/Testexp-SGD-share_params_Y-lr_3e-2-bS_32_129/ucf_olympic-16seg-disDA_none-alpha_0-advDA_none-beta_1_0.75_0.5-useBN_none-addlossDA_none-gamma_0.3-ensDA_none-mu_0-useAttn_none-n_attn_1/ --arch resnet101 --pretrained none --baseline_type video --frame_aggregation avgpool --num_segments 16 --val_segments 16 --add_fc 1 --fc_dim 512 --dropout_i 0.5 --dropout_v 0.5 --use_target uSv --share_params Y --dis_DA none --alpha 0 --place_dis N Y N --adv_DA none --beta 1 0.75 0.5 --place_adv N Y Y --use_bn none --add_loss_DA none --gamma 0.3 --ens_DA none --mu 0 --use_attn none --n_attn 1 --use_attn_frame none --gd 20 --lr 3e-2 --lr_decay 10 --lr_adaptive dann --lr_steps 10 20 --epochs 50 --optimizer SGD --n_rnn 1 --rnn_cell LSTM --n_directions 1 --n_ts 5 -b 32 129 32 -j 4 -ef 1 -pf 50 -sf 50 --copy_list N N'.split()

	# arguments setting
	parser.add_argument('--open_method', default="cls EVT", choices=['OSVM', 'cls OSVM', 'EVT', 'cls EVT'])
	parser.add_argument('--OSVM_threshold', default=0.995)  #args.OSVM_threshold python main.py
	parser.add_argument('--EVT_threshold', default=0.53) # python main.py
	parser.add_argument('--EVT_power', default=1)
	parser.add_argument('--mix_interval_length', default=0.5)

	# Entropy Maximization Loss
	parser.add_argument('--entropy_max_loss', default=False)
	parser.add_argument('--lambda_', default=1.5)
	parser.add_argument('--adv_param', default=0.1)

	parser.add_argument('--tsne', default=False)
	parser.add_argument('--auto', default=False)
	
	args = parser.parse_args()


	########## BLOCK 1: Change Here for Different Datasets ##########
	args.class_file = "data/classInd_ucf_olympic.txt"
	args.train_source_list = "dataset/olympic/list_olympic_train_ucf_olympic-feature.txt"
	args.train_target_list = "dataset/ucf101/list_ucf101_train_ucf_olympic-feature.txt"
	args.val_list = "dataset/ucf101/list_ucf101_val_ucf_olympic-feature.txt"
	########## END OF BLOCK 1 ##########


	method = "CEVT" # CEVT | TA2N | TA3N | JAN | DAN | MCD | AdaBN
	args.auto = False # True | False

	if method == "CEVT":
		args.entropy_max_loss = True # True | False
		args.open_method = "cls EVT"
		args.adv_DA = "RevGrad" # RevGrad | none
		if args.auto:
			args.lambda_ = trial.suggest_loguniform("lambda_", 1e-2, 1e1)
			args.adv_param = trial.suggest_loguniform("adv_param", 1e-2, 2e1)
			args.EVT_threshold = trial.suggest_uniform("EVT_threshold", 0.2, 0.7)
		else:


			########## BLOCK 2: Change Here for Different Hyper-Parameters ##########
			args.lambda_ = 0.214 # H->U 0.7 | U->O 0.19 | O->U 0.214
			args.adv_param = 5 # H->U 0.10 | U->O 1.83 | O->U 5
			args.EVT_threshold = 0.3 # H->U 0.45 | U->O  0.565 | O->U 0.3
			########## END OF BLOCK 2 ##########



	elif method == "TA2N" or method == "TA3N":
		args.frame_aggregation = 'trn-m'
		args.adv_DA = "RevGrad"
		args.use_attn = "TransAttn"
		args.place_adv[0] = 'Y'
		if args.auto:
			if method == "TA3N":
				args.add_loss_DA = "attentive_entropy"
				args.gamma = trial.suggest_loguniform("c", 1e-3, 1e1)
			segments = trial.suggest_int("segments", 2, 5)
			args.OSVM_threshold = trial.suggest_uniform("OSVM_threshold", 0.6, 0.99)
		else: 
			if method == "TA3N":
				args.add_loss_DA = "attentive_entropy"
				args.gamma = 0.009903869666099867
			segments = 4
			args.OSVM_threshold = 0.8990617407877891
		args.num_segments = segments
		args.val_segments = segments
	elif method == "JAN" or method == "DAN":
		args.dis_DA = method
		args.alpha = trial.suggest_uniform("alpha", 0.01, 2) if args.auto else 0.1 # 1 | 0.1 | 0.01
		args.OSVM_threshold = trial.suggest_uniform("OSVM_threshold", 0.5, 0.99) if args.auto else 0.99
	elif method == "MCD" or method == "AdaBN":
		args.ens_DA = method if method == "MCD" else "none"
		args.use_bn = method if method == "AdaBN" else "none"
		args.OSVM_threshold = trial.suggest_uniform("OSVM_threshold", 0.5, 0.99) if args.auto else  0.9999

	if args.auto == False:
		if args.open_method == "cls EVT":
			print(Fore.GREEN + 'adv_param:', args.adv_param)

		if args.open_method == "EVT" or args.open_method == "cls EVT":
			print(Fore.GREEN + 'EVT_threshold:', args.EVT_threshold)	
			print(Fore.GREEN + 'EVT_power:', args.EVT_power)
			print(Fore.GREEN + 'mix_interval_length:', args.mix_interval_length)

		if args.entropy_max_loss:
			print()
			print(Fore.GREEN + "Using entropy maximization loss ...")
			print(Fore.GREEN + 'lambda_:', args.lambda_)
			print()
		
		if args.open_method == "OSVM":
			print(Fore.GREEN + 'OSVM_threshold:', args.OSVM_threshold)
		
		print(Fore.GREEN + 'Baseline:', args.baseline_type)
		print(Fore.GREEN + 'Frame aggregation method:', args.frame_aggregation)

		print(Fore.GREEN + 'target data usage:', args.use_target)

		if args.use_target == 'none':
			print(Fore.GREEN + 'no Domain Adaptation')
		else:
			if args.dis_DA != 'none':
				print(Fore.GREEN + 'Apply the discrepancy-based Domain Adaptation approach:', args.dis_DA)
				if len(args.place_dis) != args.add_fc + 2:
					raise ValueError(Back.RED + 'len(place_dis) should be equal to add_fc + 2')

			if args.adv_DA != 'none':
				print(Fore.GREEN + 'Apply the adversarial-based Domain Adaptation approach:', args.adv_DA)

			if args.use_bn != 'none':
				print(Fore.GREEN + 'Apply the adaptive normalization approach:', args.use_bn)

	# determine the categories
	class_names = [line.strip().split(' ', 1)[1] for line in open(args.class_file)]
	num_class = len(class_names) + 1 if args.open_method == "OSBP" else len(class_names)

	class_names.append('UNK')

	entropy_cuts = torch.zeros(num_class)
	entropy_cuts[entropy_cuts==0] = 100

	meter = AverageMeterIvan(num_class) if args.open_method == "OSBP" \
		else AverageMeterIvan(num_class + 1)

	#=== check the folder existence ===#
	path_exp = args.exp_path + args.modality + '/'
	if not os.path.isdir(path_exp):
		os.makedirs(path_exp)

	if args.tensorboard:
		writer = SummaryWriter(path_exp + '/tensorboard')  # for tensorboardX

	#=== initialize the model ===#
	if args.auto == False:
		print(Fore.CYAN + 'preparing the model......')
	model = VideoModel(num_class, args.baseline_type, args.frame_aggregation, args.modality,
				train_segments=args.num_segments, val_segments=args.val_segments, 
				base_model=args.arch, path_pretrained=args.pretrained,
				add_fc=args.add_fc, fc_dim = args.fc_dim,
				dropout_i=args.dropout_i, dropout_v=args.dropout_v, partial_bn=not args.no_partialbn,
				use_bn=args.use_bn if args.use_target != 'none' else 'none', ens_DA=args.ens_DA if args.use_target != 'none' else 'none',
				n_rnn=args.n_rnn, rnn_cell=args.rnn_cell, n_directions=args.n_directions, n_ts=args.n_ts,
				use_attn=args.use_attn, n_attn=args.n_attn, use_attn_frame=args.use_attn_frame,
				verbose=args.verbose, share_params=args.share_params)

	model = torch.nn.DataParallel(model).cuda()

	
	if args.optimizer == 'SGD':
		# print(Fore.YELLOW + 'using SGD')
		optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
	elif args.optimizer == 'Adam':
		# print(Fore.YELLOW + 'using Adam')
		optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
	else:
		print(Back.RED + 'optimizer not support or specified!!!')
		exit()

	#=== check point ===#
	start_epoch = 1
	# print(Fore.CYAN + 'checking the checkpoint......')
	if args.resume:
		if os.path.isfile(args.resume):
			checkpoint = torch.load(args.resume)
			start_epoch = checkpoint['epoch'] + 1
			best_prec1 = checkpoint['best_prec1']
			model.load_state_dict(checkpoint['state_dict'])
			print(("=> loaded checkpoint '{}' (epoch {})"
				  .format(args.resume, checkpoint['epoch'])))
			if args.resume_hp:
				print("=> loaded checkpoint hyper-parameters")
				optimizer.load_state_dict(checkpoint['optimizer'])
		else:
			print(Back.RED + "=> no checkpoint found at '{}'".format(args.resume))

	cudnn.benchmark = True

	#--- open log files ---#
	if not args.evaluate:
		if args.resume:
			train_file = open(path_exp + 'train.log', 'a')
			train_short_file = open(path_exp + 'train_short.log', 'a')
			val_file = open(path_exp + 'val.log', 'a')
			val_short_file = open(path_exp + 'val_short.log', 'a')
			train_file.write('========== start: ' + str(start_epoch) + '\n')  # separation line
			train_short_file.write('========== start: ' + str(start_epoch) + '\n')
			val_file.write('========== start: ' + str(start_epoch) + '\n')
			val_short_file.write('========== start: ' + str(start_epoch) + '\n')
		else:
			train_short_file = open(path_exp + 'train_short.log', 'w')
			val_short_file = open(path_exp + 'val_short.log', 'w')
			train_file = open(path_exp + 'train.log', 'w')
			val_file = open(path_exp + 'val.log', 'w')

		val_best_file = open(args.save_best_log, 'a')

	else:
		test_short_file = open(path_exp + 'test_short.log', 'w')
		test_file = open(path_exp + 'test.log', 'w')

	#=== Data loading ===#
	if args.auto == False:
		print(Fore.CYAN + 'loading data......')

	if args.use_opencv:
		print("use opencv functions")

	if args.modality == 'RGB':
		data_length = 1
	elif args.modality in ['Flow', 'RGBDiff', 'RGBDiff2', 'RGBDiffplus']:
		data_length = 5

	# calculate the number of videos to load for training in each list ==> make sure the iteration # of source & target are same
	num_source = sum(1 for i in open(args.train_source_list))
	num_target = sum(1 for i in open(args.train_target_list))
	num_val = sum(1 for i in open(args.val_list))

	num_iter_source = num_source / args.batch_size[0]
	num_iter_target = num_target / args.batch_size[1]
	num_max_iter = max(num_iter_source, num_iter_target)
	num_source_train = round(num_max_iter*args.batch_size[0]) if args.copy_list[0] == 'Y' else num_source
	num_target_train = round(num_max_iter*args.batch_size[1]) if args.copy_list[1] == 'Y' else num_target

	# calculate the weight for each class
	class_id_list = [int(line.strip().split(' ')[2]) for line in open(args.train_source_list)]
	class_id, class_data_counts = np.unique(np.array(class_id_list), return_counts=True)
	class_freq = (class_data_counts / class_data_counts.sum()).tolist()

	weight_source_class = torch.ones(num_class).cuda()
	weight_domain_loss = torch.Tensor([1, 1]).cuda()

	if args.weighted_class_loss == 'Y':
		weight_source_class = 1 / torch.Tensor(class_freq).cuda()

	if args.weighted_class_loss_DA == 'Y':
		weight_domain_loss = torch.Tensor([1/num_source_train, 1/num_target_train]).cuda()

	# data loading (always need to load the testing data)
	val_segments = args.val_segments if args.val_segments > 0 else args.num_segments
	val_set = TSNDataSet("", args.val_list, num_dataload=num_val, num_segments=val_segments,
						 new_length=data_length, modality=args.modality,
						 image_tmpl="img_{:05d}.t7" if args.modality in ["RGB", "RGBDiff", "RGBDiff2",
																		  "RGBDiffplus"] else args.flow_prefix + "{}_{:05d}.t7",
						 random_shift=False,
						 test_mode=True,
						 )
	val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size[2], shuffle=False,
											 num_workers=args.workers, pin_memory=True)

	if not args.evaluate:
		source_set = TSNDataSet("", args.train_source_list, num_dataload=num_source_train, num_segments=args.num_segments,
								new_length=data_length, modality=args.modality,
								image_tmpl="img_{:05d}.t7" if args.modality in ["RGB", "RGBDiff", "RGBDiff2", "RGBDiffplus"] else args.flow_prefix+"{}_{:05d}.t7",
								random_shift=False,
								test_mode=True,
								)

		source_sampler = torch.utils.data.sampler.RandomSampler(source_set)
		source_loader = torch.utils.data.DataLoader(source_set, batch_size=args.batch_size[0], shuffle=False, sampler=source_sampler, num_workers=args.workers, pin_memory=True)

		target_set = TSNDataSet("", args.train_target_list, num_dataload=num_target_train, num_segments=args.num_segments,
								new_length=data_length, modality=args.modality,
								image_tmpl="img_{:05d}.t7" if args.modality in ["RGB", "RGBDiff", "RGBDiff2", "RGBDiffplus"] else args.flow_prefix + "{}_{:05d}.t7",
								random_shift=False,
								test_mode=True,
								)

		target_sampler = torch.utils.data.sampler.RandomSampler(target_set)
		target_loader = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size[1], shuffle=False, sampler=target_sampler, num_workers=args.workers, pin_memory=True)

	# --- Optimizer ---#
	# define loss function (criterion) and optimizer
	if args.loss_type == 'nll':
		criterion = torch.nn.CrossEntropyLoss(weight=weight_source_class).cuda()
		criterion_domain = torch.nn.CrossEntropyLoss(weight=weight_domain_loss).cuda()
	else:
		raise ValueError("Unknown loss type")

	if args.evaluate:
		print(Fore.CYAN + 'evaluation only......')
		prec1, entropy_cuts = validate(val_loader, model, criterion, num_class, 0, test_file, meter)
		test_short_file.write('%.3f\n' % prec1)
		return

	#=== Training ===#

	start_train = time.time()
	if args.auto == False:
		print(Fore.CYAN + 'start training......')
	beta = args.beta
	gamma = args.gamma
	mu = args.mu
	loss_c_current = 999 # random large number
	loss_c_previous = 999 # random large number

	attn_source_all = torch.Tensor()
	attn_target_all = torch.Tensor()

	for epoch in range(start_epoch, args.epochs+1):

		## schedule for parameters
		alpha = 2 / (1 + math.exp(-1 * (epoch) / args.epochs)) - 1 if args.alpha < 0 else args.alpha

		## schedule for learning rate
		if args.lr_adaptive == 'loss':
			adjust_learning_rate_loss(optimizer, args.lr_decay, loss_c_current, loss_c_previous, '>')
		elif args.lr_adaptive == 'none' and epoch in args.lr_steps:
			adjust_learning_rate(optimizer, args.lr_decay)

		# train for one epoch
		loss_c, attn_epoch_source, attn_epoch_target = train(num_class, source_loader, target_loader, model, criterion, criterion_domain, optimizer, epoch, train_file, train_short_file, alpha, beta, gamma, mu, entropy_cuts, class_names)
		
		if args.save_attention >= 0:
			attn_source_all = torch.cat((attn_source_all, attn_epoch_source.unsqueeze(0)))  # save the attention values
			attn_target_all = torch.cat((attn_target_all, attn_epoch_target.unsqueeze(0)))  # save the attention values

		# update the recorded loss_c
		loss_c_previous = loss_c_current
		loss_c_current = loss_c

		# evaluate on validation set
		if epoch % args.eval_freq == 0 or epoch == args.epochs:
			prec1, entropy_cuts = validate(val_loader, model, criterion, num_class, epoch, val_file, meter)

			# auto
			trial.report(prec1, epoch)
			# Handle pruning based on the intermediate value.
			if trial.should_prune():
				raise optuna.exceptions.TrialPruned()
			
			

			# remember best prec@1 and save checkpoint
			is_best = prec1 > best_prec1
			if args.auto == False:
				line_update = ' ==> updating the best accuracy' if is_best else ''
				line_best = "Best score {} vs current score {}".format(best_prec1, prec1) + line_update
				print(Fore.YELLOW + line_best)
			val_short_file.write('%.3f\n' % prec1)

			best_prec1 = max(prec1, best_prec1)

			if epoch >= 20 and best_prec1 < 0.7:
				break

			if epoch >= 30 and best_prec1 - prec1 > 0.2:
				break

			if args.tensorboard:
				writer.add_text('Best_Accuracy', str(best_prec1), epoch)

			if args.save_model:
				save_checkpoint({
					'epoch': epoch,
					'arch': args.arch,
					'state_dict': model.state_dict(),
					'optimizer' : optimizer.state_dict(),
					'best_prec1': best_prec1,
					'prec1': prec1,
				}, is_best, path_exp)
			
	

	end_train = time.time()
	
	# print(Fore.CYAN + 'total training time:', end_train - start_train)
	val_best_file.write('%.3f\n' % best_prec1)

	# --- write the total time to log files ---#
	line_time = 'total time: {:.3f} '.format(end_train - start_train)
	if not args.evaluate:
		train_file.write(line_time)
		train_short_file.write(line_time)
		val_file.write(line_time)
		val_short_file.write(line_time)
	else:
		test_file.write(line_time)
		test_short_file.write(line_time)

	#--- close log files ---#
	if not args.evaluate:
		train_file.close()
		train_short_file.close()
		val_file.close()
		val_short_file.close()
	else:
		test_file.close()
		test_short_file.close()

	if args.tensorboard:
		writer.close()

	if args.save_attention >= 0:
		np.savetxt('attn_source_' + str(args.save_attention) + '.log', attn_source_all.cpu().detach().numpy(), fmt="%s")
		np.savetxt('attn_target_' + str(args.save_attention) + '.log', attn_target_all.cpu().detach().numpy(), fmt="%s")

	# auto	
	return best_prec1


def train(num_class, source_loader, target_loader, model, criterion, criterion_domain, optimizer, epoch, log, log_short, alpha, beta, gamma, mu, entropy_cuts, class_names):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses_a = AverageMeter()  # adversarial loss
	losses_d = AverageMeter()  # discrepancy loss
	losses_e = AverageMeter()  # entropy loss
	losses_s = AverageMeter()  # ensemble loss
	losses_c = AverageMeter()  # classification loss
	losses_h = AverageMeter()  # Entropy Maximization loss
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	if args.no_partialbn:
		model.module.partialBN(False)
	else:
		model.module.partialBN(True)

	# switch to train mode
	model.train()

	end = time.time()
	data_loader = enumerate(zip(source_loader, target_loader))

	# step info
	start_steps = epoch * len(source_loader)
	total_steps = args.epochs * len(source_loader)

	# initialize the embedding
	if args.tensorboard:
		feat_source_display = None
		label_source_display = None
		label_source_domain_display = None

		feat_target_display = None
		label_target_display = None
		label_target_domain_display = None

	attn_epoch_source = torch.Tensor()
	attn_epoch_target = torch.Tensor()

	if args.tsne:
		feat_source_list = []
		feat_target_list = []
		source_label_list = []
		target_label_list = []

	for i, ((source_data, source_record),(target_data, target_record)) in data_loader:

		
		source_label, target_label = torch.tensor([int(i) for i in source_record[2]]), torch.FloatTensor([int(i) for i in target_record[2]])
		

		# setup hyperparameters
		p = float(i + start_steps) / total_steps
		beta_dann = 2. / (1. + np.exp(-10 * p)) - 1
		beta = [beta_dann if beta[i] < 0 else beta[i] for i in range(len(beta))] # replace the default beta if value < 0

		source_size_ori = source_data.size()  # original shape
		target_size_ori = target_data.size()  # original shape
		batch_source_ori = source_size_ori[0]
		batch_target_ori = target_size_ori[0]
		# add dummy tensors to keep the same batch size for each epoch (for the last epoch)
		if batch_source_ori < args.batch_size[0]:
			source_data_dummy = torch.zeros(args.batch_size[0] - batch_source_ori, source_size_ori[1], source_size_ori[2])
			source_data = torch.cat((source_data, source_data_dummy))
		if batch_target_ori < args.batch_size[1]:
			target_data_dummy = torch.zeros(args.batch_size[1] - batch_target_ori, target_size_ori[1], target_size_ori[2])
			target_data = torch.cat((target_data, target_data_dummy))

		# add dummy tensors to make sure batch size can be divided by gpu #
		if source_data.size(0) % gpu_count != 0:
			source_data_dummy = torch.zeros(gpu_count - source_data.size(0) % gpu_count, source_data.size(1), source_data.size(2))
			source_data = torch.cat((source_data, source_data_dummy))
		if target_data.size(0) % gpu_count != 0:
			target_data_dummy = torch.zeros(gpu_count - target_data.size(0) % gpu_count, target_data.size(1), target_data.size(2))
			target_data = torch.cat((target_data, target_data_dummy))

		# measure data loading time
		data_time.update(time.time() - end)

		source_label = source_label.cuda(non_blocking=True) # pytorch 0.4.X
		target_label = target_label.cuda(non_blocking=True) # pytorch 0.4.X

		if args.baseline_type == 'frame':
			source_label_frame = source_label.unsqueeze(1).repeat(1,args.num_segments).view(-1) # expand the size for all the frames
			target_label_frame = target_label.unsqueeze(1).repeat(1, args.num_segments).view(-1)

		label_source = source_label_frame if args.baseline_type == 'frame' else source_label  # determine the label for calculating the loss function
		label_target = target_label_frame if args.baseline_type == 'frame' else target_label

		#====== pre-train source data ======#
		if args.pretrain_source:
			#------ forward pass data again ------#
			_, out_source, out_source_2, _, _, _, _, _, _, _ = model(source_data, target_data, beta, mu, is_train=True, reverse=False)

			# ignore dummy tensors
			out_source = out_source[:batch_source_ori]
			out_source_2 = out_source_2[:batch_source_ori]

			#------ calculate the loss function ------#
			# 1. calculate the classification loss
			out = out_source
			label = label_source

			loss = criterion(out, label)
			if args.ens_DA == 'MCD' and args.use_target != 'none':
				loss += criterion(out_source_2, label)

			# compute gradient and do SGD step
			optimizer.zero_grad()
			loss.backward()

			if args.clip_gradient is not None:
				total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
				if total_norm > args.clip_gradient and args.verbose:
					print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

			optimizer.step()


		#====== forward pass data ======#
		attn_source, out_source, out_source_2, pred_domain_source, feat_source, attn_target, out_target, out_target_2, pred_domain_target, feat_target = model(source_data, target_data, beta, mu, is_train=True, reverse=False)

		# ignore dummy tensors
		attn_source, out_source, out_source_2, pred_domain_source, feat_source = removeDummy(attn_source, out_source, out_source_2, pred_domain_source, feat_source, batch_source_ori)
		attn_target, out_target, out_target_2, pred_domain_target, feat_target = removeDummy(attn_target, out_target, out_target_2, pred_domain_target, feat_target, batch_target_ori)

		if args.pred_normalize == 'Y': # use the uncertainly method (in contruction...)
			out_source = out_source / out_source.var().log()
			out_target = out_target / out_target.var().log()


		if args.tsne:
			feat_source_list.append(out_source.cpu())
			feat_target_list.append(out_target.cpu())
			source_label_list.append(source_label.cpu())
			target_label_list.append(target_label.cpu())

		# store the embedding
		if args.tensorboard:
			feat_source_display = feat_source[1] if i==0 else torch.cat((feat_source_display, feat_source[1]), 0)
			label_source_display = label_source if i==0 else torch.cat((label_source_display, label_source), 0)
			label_source_domain_display = torch.zeros(label_source.size(0)) if i==0 else torch.cat((label_source_domain_display, torch.zeros(label_source.size(0))), 0)
			feat_target_display = feat_target[1] if i==0 else torch.cat((feat_target_display, feat_target[1]), 0)
			label_target_display = label_target if i==0 else torch.cat((label_target_display, label_target), 0)
			label_target_domain_display = torch.ones(label_target.size(0)) if i==0 else torch.cat((label_target_domain_display, torch.ones(label_target.size(0))), 0)

		#====== calculate the loss function ======#
		# 1. calculate the classification loss
		out = out_source
		label = label_source

		if args.use_target == 'Sv':
			out = torch.cat((out, out_target))
			label = torch.cat((label, label_target))

		loss_classification = criterion(out, label)
		if args.ens_DA == 'MCD' and args.use_target != 'none':
			loss_classification += criterion(out_source_2, label)

		losses_c.update(loss_classification.item(), out_source.size(0)) # pytorch 0.4.X
		loss = loss_classification

		# 2. calculate the loss for DA
		# (I) discrepancy-based approach: discrepancy loss
		if args.dis_DA != 'none' and args.use_target != 'none':
			loss_discrepancy = 0

			kernel_muls = [2.0]*2
			kernel_nums = [2, 5]
			fix_sigma_list = [None]*2

			if args.dis_DA == 'JAN':
				# ignore the features from shared layers
				feat_source_sel = feat_source[:-args.add_fc]
				feat_target_sel = feat_target[:-args.add_fc]

				size_loss = min(feat_source_sel[0].size(0), feat_target_sel[0].size(0))  # choose the smaller number
				feat_source_sel = [feat[:size_loss] for feat in feat_source_sel]
				feat_target_sel = [feat[:size_loss] for feat in feat_target_sel]

				loss_discrepancy += JAN(feat_source_sel, feat_target_sel, kernel_muls=kernel_muls, kernel_nums=kernel_nums, fix_sigma_list=fix_sigma_list, ver=2)

			else:
				# extend the parameter list for shared layers
				kernel_muls.extend([kernel_muls[-1]]*args.add_fc)
				kernel_nums.extend([kernel_nums[-1]]*args.add_fc)
				fix_sigma_list.extend([fix_sigma_list[-1]]*args.add_fc)

				for l in range(0, args.add_fc + 2):  # loss from all the features (+2 because of frame-aggregation layer + final fc layer)
					if args.place_dis[l] == 'Y':
						# select the data for calculating the loss (make sure source # == target #)
						size_loss = min(feat_source[l].size(0), feat_target[l].size(0)) # choose the smaller number
						# select
						feat_source_sel = feat_source[l][:size_loss]
						feat_target_sel = feat_target[l][:size_loss]

						# break into multiple batches to avoid "out of memory" issue
						size_batch = min(256,feat_source_sel.size(0))
						feat_source_sel = feat_source_sel.view((-1,size_batch) + feat_source_sel.size()[1:])
						feat_target_sel = feat_target_sel.view((-1,size_batch) + feat_target_sel.size()[1:])

						if args.dis_DA == 'CORAL':
							losses_coral = [CORAL(feat_source_sel[t], feat_target_sel[t]) for t in range(feat_source_sel.size(0))]
							loss_coral = sum(losses_coral)/len(losses_coral)
							loss_discrepancy += loss_coral
						elif args.dis_DA == 'DAN':
							losses_mmd = [mmd_rbf(feat_source_sel[t], feat_target_sel[t], kernel_mul=kernel_muls[l], kernel_num=kernel_nums[l], fix_sigma=fix_sigma_list[l], ver=2) for t in range(feat_source_sel.size(0))]
							loss_mmd = sum(losses_mmd) / len(losses_mmd)

							loss_discrepancy += loss_mmd
						else:
							raise NameError('not in dis_DA!!!')

			losses_d.update(loss_discrepancy.item(), feat_source[0].size(0))
			loss += alpha * loss_discrepancy

		# (II) adversarial discriminative model: adversarial loss
		if args.adv_DA != 'none' and args.use_target != 'none' and args.open_method != 'cls EVT':
			loss_adversarial = 0
			pred_domain_all = []
			pred_domain_target_all = []

			for l in range(len(args.place_adv)):
				if args.place_adv[l] == 'Y':

					# reshape the features (e.g. 128x5x2 --> 640x2)
					pred_domain_source_single = pred_domain_source[l].view(-1, pred_domain_source[l].size()[-1])
					pred_domain_target_single = pred_domain_target[l].view(-1, pred_domain_target[l].size()[-1])

					# prepare domain labels
					source_domain_label = torch.zeros(pred_domain_source_single.size(0)).long()
					target_domain_label = torch.ones(pred_domain_target_single.size(0)).long()
					domain_label = torch.cat((source_domain_label,target_domain_label),0)

					domain_label = domain_label.cuda(non_blocking=True)

					pred_domain = torch.cat((pred_domain_source_single, pred_domain_target_single),0)
					pred_domain_all.append(pred_domain)
					pred_domain_target_all.append(pred_domain_target_single)

					if args.pred_normalize == 'Y':  # use the uncertainly method (in construction......)
						pred_domain = pred_domain / pred_domain.var().log()
					loss_adversarial_single = criterion_domain(pred_domain, domain_label)
					# loss_adversarial_single = ivan_CEL(pred_domain,
					# 										   domain_label)
					loss_adversarial += loss_adversarial_single

			losses_a.update(loss_adversarial.item(), pred_domain.size(0))
			loss += loss_adversarial

		# (II) adversarial discriminative model: adversarial loss - weighted
		if args.adv_DA != 'none' and args.use_target != 'none' and args.open_method == 'cls EVT':
			loss_adversarial = 0
			pred_domain_all = []
			pred_domain_target_all = []

			for l in range(len(args.place_adv)):
				if args.place_adv[l] == 'Y':

					# reshape the features (e.g. 128x5x2 --> 640x2)
					pred_domain_source_single = pred_domain_source[l].view(-1, pred_domain_source[l].size()[-1])
					pred_domain_target_single = pred_domain_target[l].view(-1, pred_domain_target[l].size()[-1])

					# prepare domain labels
					source_domain_label = torch.zeros(pred_domain_source_single.size(0)).long()
					target_domain_label = torch.ones(pred_domain_target_single.size(0)).long()
					domain_label = torch.cat((source_domain_label,target_domain_label),0)

					domain_label = domain_label.cuda(non_blocking=True)

					pred_domain = torch.cat((pred_domain_source_single, pred_domain_target_single),0)
					pred_domain_all.append(pred_domain)
					pred_domain_target_all.append(pred_domain_target_single)

					if args.pred_normalize == 'Y':  # use the uncertainly method (in construction......)
						pred_domain = pred_domain / pred_domain.var().log()
					loss_adversarial_single = criterion_domain(pred_domain, domain_label)
					# loss_adversarial_single = ivan_CEL(pred_domain,
					# 										   domain_label)

					loss_adversarial_single_source = criterion_domain(pred_domain_source_single, source_domain_label.cuda(non_blocking=True))



					#------- Computing Target Advserial Weights -------# 
					target_entropy = -(F.softmax(out_target, dim=1) * F.log_softmax(out_target, dim=1)).sum(dim=1)	

					# th-based
					# 1. change mininum weight to 0
					# 2. remove weight upper bound
					weights = []
					even_tensor = torch.ones(num_class)/num_class
					entropy_max = -(F.softmax(even_tensor, dim=0)* F.log_softmax(even_tensor, dim=0)).sum()
					# entropy_cuts[entropy_cuts==100] = -0.5 # 100 means 
					for i_ in range(list(out_target.shape)[0]):
						max_class = int(out_target.data.max(1)[1][i_])
						entropy_cut = float(entropy_cuts[max_class]) 

						if entropy_cut > float(entropy_max)/2: 
							hard_interval_half = (float(entropy_max) - entropy_cut)/2
						else: 
							hard_interval_half = entropy_cut/2

						# # use fixed value 0.5 as interval 
						# weight = entropy_cut - float(target_entropy[i_]) + 0.5
						

						# max = 2.2 t=1.4 half = 0.4 
						# given h = 1.4: 0.5 + 0.5 * (1.4-1.4)/0.4 = 0.5
						# given h = 1.6: 0.5 + 0.5 * (1.4-1.6)/0.4 = 0.25
						# given h = 1: 0.5 + 0.5 * (1.4-1)/0.4 = 0
						weight = 0.5 + args.mix_interval_length * (entropy_cut - float(target_entropy[i_])) / hard_interval_half # 0.25 0.75 1
						weight = 0 if weight < 0 else weight
						weight = 1 if weight > 1 else weight
						weights.append(weight)

					weights = torch.tensor(weights)
					# weights = F.normalize(weights.float(), dim=0, p=1) * list(weights.shape)[0]

					# entropy-based

					# # even_tensor = torch.ones(num_class)/num_class
					# # entropy_max = -(F.softmax(even_tensor, dim=0)* F.log_softmax(even_tensor, dim=0)).sum()
					# weights = entropy_max - target_entropy 
					# weights = F.normalize(weights.float(), dim=0, p=1) * list(weights.shape)[0]



					if list(weights.shape)[0] != list(pred_domain_target_single.shape)[0]:
						weights = tile(weights, 0, args.num_segments)

					loss_adversarial_single_target = ivan_CEL(pred_domain_target_single, target_domain_label.cuda(non_blocking=True), weights.cuda())
					# loss_adversarial += loss_adversarial_single
					loss_adversarial += loss_adversarial_single_source * 0.5
					loss_adversarial += loss_adversarial_single_target * 0.5

			losses_a.update(loss_adversarial.item(), pred_domain.size(0))
			loss += args.adv_param * loss_adversarial

		# (III) other loss
		# 1. entropy loss for target data
		if args.add_loss_DA == 'target_entropy' and args.use_target != 'none':
			loss_entropy = cross_entropy_soft(out_target)
			losses_e.update(loss_entropy.item(), out_target.size(0))
			loss += gamma * loss_entropy

		# 2. discrepancy loss for MCD (CVPR 18)
		if args.ens_DA == 'MCD' and args.use_target != 'none':
			_, _, _, _, _, attn_target, out_target, out_target_2, pred_domain_target, feat_target = model(source_data, target_data, beta, mu, is_train=True, reverse=True)

			# ignore dummy tensors
			_, out_target, out_target_2, _, _ = removeDummy(attn_target, out_target, out_target_2, pred_domain_target, feat_target, batch_target_ori)

			loss_dis = -dis_MCD(out_target, out_target_2)
			losses_s.update(loss_dis.item(), out_target.size(0))
			loss += loss_dis

		# 3. attentive entropy loss
		if args.add_loss_DA == 'attentive_entropy' and args.use_attn != 'none' and args.use_target != 'none':
			loss_entropy = attentive_entropy(torch.cat((out_source, out_target),0), pred_domain_all[1])
			losses_e.update(loss_entropy.item(), out_target.size(0))
			loss += gamma * loss_entropy


		# 5. H loss
		if args.entropy_max_loss == True:
			loss_H = H_loss(out_target)
			losses_h.update(loss_H.item(), out_target.size(0))
			loss += args.lambda_ * loss_H 


		pred = out

		prec1, prec5 = accuracy(pred.data, label, topk=(1, 2))

		losses.update(loss.item())
		top1.update(prec1.item(), out_source.size(0))
		top5.update(prec5.item(), out_source.size(0))

		# compute gradient and do SGD step
		optimizer.zero_grad()

		loss.backward()

		if args.clip_gradient is not None:
			total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
			if total_norm > args.clip_gradient and args.verbose:
				print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0 and args.auto == False:
			line = 'Train: [{0}][{1}/{2}], lr: {lr:.5f}\t' + \
				   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' + \
				   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' + \
				   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' + \
				   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t' + \
				   'Loss {loss.val:.4f} ({loss.avg:.4f})   ' \
				   'loss_c {loss_c.avg:.4f}\t loss_h {loss_h.avg:.4f}\t'

			if args.dis_DA != 'none' and args.use_target != 'none':
				line += 'alpha {alpha:.3f}  loss_d {loss_d.avg:.4f}\t'

			if args.adv_DA != 'none' and args.use_target != 'none':
				line += 'beta {beta[0]:.3f}, {beta[1]:.3f}, {beta[2]:.3f}  loss_a {loss_a.avg:.4f}\t'

			if args.add_loss_DA != 'none' and args.use_target != 'none':
				line += 'gamma {gamma:.6f}  loss_e {loss_e.avg:.4f}\t'

			if args.ens_DA != 'none' and args.use_target != 'none':
				line += 'mu {mu:.6f}  loss_s {loss_s.avg:.4f}\t'

			line = line.format(
				epoch, i, len(source_loader), batch_time=batch_time, data_time=data_time, alpha=alpha, beta=beta, gamma=gamma, mu=mu,
				loss=losses, loss_c=losses_c, loss_h=losses_h, loss_d=losses_d, loss_a=losses_a, loss_e=losses_e, loss_s=losses_s, top1=top1, top5=top5,
				lr=optimizer.param_groups[0]['lr'])

			if i % args.show_freq == 0:
				print(line)

			log.write('%s\n' % line)

		# adjust the learning rate for ech step (e.g. DANN)
		if args.lr_adaptive == 'dann':
			adjust_learning_rate_dann(optimizer, p)

		# save attention values w/ the selected class
		if args.save_attention >= 0:
			attn_source = attn_source[source_label==args.save_attention]
			attn_target = attn_target[target_label==args.save_attention]
			attn_epoch_source = torch.cat((attn_epoch_source, attn_source.cpu()))
			attn_epoch_target = torch.cat((attn_epoch_target, attn_target.cpu()))


	if args.tsne and epoch >=8 and epoch <= 25:
		feat_source = torch.cat(feat_source_list).squeeze()
		feat_target = torch.cat(feat_target_list).squeeze()
		source_label = torch.cat(source_label_list).squeeze()
		target_label = torch.cat(target_label_list).squeeze()
		target_label[target_label >= num_class] = num_class
		save_tsne_path='tsn/TA3N_'+ str(epoch) +'_.png'
		
		visualize_TSNE(feat_source, feat_target, source_label, target_label, save_tsne_path, class_names)
		print('finshed tsne')

	# update the embedding every epoch
	if args.tensorboard:
		n_iter_train = epoch * len(source_loader) # calculate the total iteration
		# embedding
		# see source and target separately
		writer.add_embedding(feat_source_display, metadata=label_source_display.data, global_step=n_iter_train, tag='train_source')
		writer.add_embedding(feat_target_display, metadata=label_target_display.data, global_step=n_iter_train, tag='train_target')

		# mix source and target
		feat_all_display = torch.cat((feat_source_display, feat_target_display), 0)
		label_all_domain_display = torch.cat((label_source_domain_display, label_target_domain_display), 0)
		writer.add_embedding(feat_all_display, metadata=label_all_domain_display.data, global_step=n_iter_train, tag='train_DA')

		# emphazise some classes (1, 3, 11 here)
		label_source_1 = 1 * torch.eq(label_source_display, torch.cuda.LongTensor([1]).repeat(label_source_display.size(0))).long().cuda(non_blocking=True)
		label_source_3 = 2 * torch.eq(label_source_display, torch.cuda.LongTensor([3]).repeat(label_source_display.size(0))).long().cuda(non_blocking=True)
		label_source_11 = 3 * torch.eq(label_source_display, torch.cuda.LongTensor([11]).repeat(label_source_display.size(0))).long().cuda(non_blocking=True)

		label_target_1 = 4 * torch.eq(label_target_display, torch.cuda.LongTensor([1]).repeat(label_target_display.size(0))).long().cuda(non_blocking=True)
		label_target_3 = 5 * torch.eq(label_target_display, torch.cuda.LongTensor([3]).repeat(label_target_display.size(0))).long().cuda(non_blocking=True)
		label_target_11 = 6 * torch.eq(label_target_display, torch.cuda.LongTensor([11]).repeat(label_target_display.size(0))).long().cuda(non_blocking=True)

		label_source_display_new = label_source_1 + label_source_3 + label_source_11
		id_source_show = ~torch.eq(label_source_display_new, 0).cuda(non_blocking=True)
		label_source_display_new = label_source_display_new[id_source_show]
		feat_source_display_new = feat_source_display[id_source_show]

		label_target_display_new = label_target_1 + label_target_3 + label_target_11
		id_target_show = ~torch.eq(label_target_display_new, 0).cuda(non_blocking=True)
		label_target_display_new = label_target_display_new[id_target_show]
		feat_target_display_new = feat_target_display[id_target_show]

		feat_all_display_new = torch.cat((feat_source_display_new, feat_target_display_new), 0)
		label_all_display_new = torch.cat((label_source_display_new, label_target_display_new), 0)
		writer.add_embedding(feat_all_display_new, metadata=label_all_display_new.data, global_step=n_iter_train, tag='train_DA_labels')

	if args.auto == False:
		log_short.write('%s\n' % line)
	return losses_c.avg, attn_epoch_source.mean(0), attn_epoch_target.mean(0)

def validate(val_loader, model, criterion, num_class, epoch, log, meter):

	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	# switch to evaluate mode
	model.eval()

	meter.reset()
	correct1 = 0
	size = 0
	labels = []

	end = time.time()

	target_entropy_all = [] # for global-level EVT

	target_logits_cls, target_entropy_cls = [], [] # for class-level EVT
	for cls in range(num_class):
		target_logits_cls.append([])
		target_entropy_cls.append([])

	romove_unk = False if epoch != 61 else True
	
	if args.open_method != 'cls EVT':
		entropy_cuts = []

	# initialize the embedding
	if args.tensorboard:
		feat_val_display = None
		label_val_display = None

	for i, (val_data, val_record) in enumerate(val_loader):



		val_label = torch.tensor([int(i) for i in val_record[2]])
		val_path = val_record[0]

		val_size_ori = val_data.size()  # original shape
		batch_val_ori = val_size_ori[0]

		# add dummy tensors to keep the same batch size for each epoch (for the last epoch)
		if batch_val_ori < args.batch_size[2]:
			val_data_dummy = torch.zeros(args.batch_size[2] - batch_val_ori, val_size_ori[1], val_size_ori[2])
			val_data = torch.cat((val_data, val_data_dummy))

		# add dummy tensors to make sure batch size can be divided by gpu #
		if val_data.size(0) % gpu_count != 0:
			val_data_dummy = torch.zeros(gpu_count - val_data.size(0) % gpu_count, val_data.size(1), val_data.size(2))
			val_data = torch.cat((val_data, val_data_dummy))

		val_label = val_label.cuda(non_blocking=True)
		with torch.no_grad():

			if args.baseline_type == 'frame':
				val_label_frame = val_label.unsqueeze(1).repeat(1,args.num_segments).view(-1) # expand the size for all the frames

			# compute output
			_, _, _, _, _, attn_val, out_val, out_val_2, pred_domain_val, feat_val = model(val_data, val_data, [0]*len(args.beta), 0, is_train=False, reverse=False)

			# ignore dummy tensors
			attn_val, out_val, out_val_2, pred_domain_val, feat_val = removeDummy(attn_val, out_val, out_val_2, pred_domain_val, feat_val, batch_val_ori)

			# measure accuracy and record loss
			label = val_label_frame if args.baseline_type == 'frame' else val_label

			# store the embedding
			if args.tensorboard:
				feat_val_display = feat_val[1] if i == 0 else torch.cat((feat_val_display, feat_val[1]), 0)
				label_val_display = label if i == 0 else torch.cat((label_val_display, label), 0)

			pred = out_val
			batch_size = label.size(0)


			# OSVM
			# pred1 = pred.data.max(1)[1]
			
			#======================== SPLIT UNK =====================#
			
			if args.open_method == "EVT": # EVT
       
				pred1 = torch.cat((pred1, pred.data.max(1)[1])) if i != 0 else pred.data.max(1)[1]
			
				# loss = criterion(pred, label)
				target_entropy = -(F.softmax(pred, dim=1) * F.log_softmax(pred, dim=1)).sum(dim=1)
				target_entropy_all.extend(target_entropy.tolist())

				# # EVT (batch level)
				# entropy_cut = EVT(target_entropy, 0.5) # EVT TAIL 
				# if romove_unk:
				# 	target_path_all.extend(val_path)
		
				# EVT (domain level)
				label[label > num_class] = num_class # change all extra classes in target to unk
				labels = labels + list(label.detach().cpu().numpy())
				if len(target_entropy_all) == len(val_loader.dataset):
					power = args.EVT_power
					threshold = args.EVT_threshold
					entropy_cut = EVT(target_entropy_all, threshold, power) 
					idx = (np.array(target_entropy_all)).tolist() > entropy_cut
					idx = torch.from_numpy(idx.astype('uint8'))
					pred1[idx] = num_class  # unk class
					prec = pred1.eq(torch.LongTensor(labels).cuda().data)
					correct1 += prec.sum()
					meter.update(np.array(labels), prec.double().cpu().numpy())
			
			elif args.open_method == "cls EVT": # class-level EVT
				
				pred1 = torch.cat((pred1, pred.data.max(1)[1])) if i != 0 else pred.data.max(1)[1]
				target_entropy = -(F.softmax(pred, dim=1) * F.log_softmax(pred, dim=1)).sum(dim=1)
				target_entropy_all.extend(target_entropy.tolist())

				for cls in range(num_class):
					logits_per_cls = pred.data[pred.data.max(1)[1] == cls]
					if logits_per_cls.nelement() != 0:
						entropy_per_cls = -(F.softmax(logits_per_cls, dim=1) * F.log_softmax(logits_per_cls, dim=1)).sum(dim=1)
						target_entropy_cls[cls].extend(entropy_per_cls.tolist())

				label[label > num_class] = num_class # change all extra classes in target to unk
				labels = labels + list(label.detach().cpu().numpy())
				if len(target_entropy_all) == len(val_loader.dataset):
					entropy_cuts = []
					power = args.EVT_power
					threshold = args.EVT_threshold
					for cls in range(num_class):
						if len(target_entropy_cls[cls]) != 0:
							entropy_cut = EVT(target_entropy_cls[cls], threshold, power)
							if entropy_cut == None:
								entropy_cuts.append(100)
							else: 
								entropy_cuts.append(entropy_cut)
						else:
							entropy_cuts.append(100) # no entropy can be greater than 100
					entropy_cuts = torch.from_numpy(np.array(entropy_cuts)).cuda()
					target_entropy_all = torch.tensor(target_entropy_all).cuda()
					idx = target_entropy_all > entropy_cuts[pred1].float()
					pred1[idx] = num_class  # unk class
					prec = pred1.eq(torch.LongTensor(labels).cuda().data)
					correct1 += prec.sum()
					meter.update(np.array(labels), prec.double().cpu().numpy())

			elif args.open_method == "cls OSVM": # class-level OSVM
				pred1 = pred.data.max(1)[1]
				class_threshold = []
				threshold_value = 0.6 
				for cls in range(num_class):
					logits_per_cls = pred.data.max(1)[0][pred.data.max(1)[1] == cls]

					if int(list(logits_per_cls.shape)[0] * threshold_value) != 0:
						threshold = logits_per_cls.topk(int(list(logits_per_cls.shape)[0] * threshold_value))[0].cpu().numpy()[-1]
						class_threshold.append(threshold)
					else:
						class_threshold.append(0)

				class_threshold = torch.from_numpy(np.array(class_threshold)).cuda()
				idx = pred[:, :].max(-1)[0].detach().double() < class_threshold[pred[:, :].max(-1)[1]].double()

				pred1[idx] = num_class  # unk class
				label[label > num_class] = num_class

				labels = labels + list(label.detach().cpu().numpy())
				prec = pred1.eq(label.data)
				correct1 += prec.sum()
				meter.update(label.view(-1).cpu().numpy(), prec.double().cpu().numpy())

			elif args.open_method == "OSVM": # OSVM

				pred1 = pred.data.max(1)[1]
				pred = F.softmax(pred, dim=-1)
				idx = pred[:, :].max(-1)[0].detach() < args.OSVM_threshold
				pred1[idx] = num_class  # unk class
				label[label > num_class] = num_class

				labels = labels + list(label.detach().cpu().numpy())
				prec = pred1.eq(label.data)
				correct1 += prec.sum()
				meter.update(label.view(-1).cpu().numpy(), prec.double().cpu().numpy())

			size += batch_size

			# res = []
			# for k in topk:
			# 	correct_k = correct[:k].view(-1).float().sum(0)
			# 	res.append(correct_k.mul_(100.0 / batch_size))


	# preds = np.array(preds)
	labels = np.array(labels)
	u, counts = np.unique(labels, return_counts=True)
	known_ratio = counts[0] / len(labels)
	unk_ratio = counts[1] / len(labels)
	ac = meter.sum[:-1].sum() / meter.count[:-1].sum()
	ac_hat = meter.avg[-1]
	H_score = 2 * (ac * ac_hat) / (ac + ac_hat)


	if args.auto == False: 

		print('\nTest Results:')
		for id in range(0, num_class):
			print('Class {}: {:.4f}'.format(id, meter.avg[id]))

		print('ALL: {:.4f},  OS: {:.4f}, OS*: {:.4f}, UNK: {:.4f}, '
			'balanced: {:.4f}.'
			.format(meter.sum.sum() / meter.count.sum(), meter.avg.mean(), meter.avg[:-1].mean(), meter.avg[-1], 2*(meter.avg[-1]*meter.avg[:-1].mean()) /(meter.avg[:-1].mean() + meter.avg[-1])))


	# 		if args.baseline_type == 'tsn':
	# 			pred = pred.view(val_label.size(0), -1, num_class).mean(dim=1) # average all the segments (needed when num_segments != val_segments)
	#
	# 		loss = criterion(pred, label)
	# 		prec1, prec5 = accuracy(pred.data, label, topk=(1, 1))
	#
	# 		losses.update(loss.item(), out_val.size(0))
	# 		top1.update(prec1.item(), out_val.size(0))
	# 		top5.update(prec5.item(), out_val.size(0))
	#
	# 		# measure elapsed time
	# 		batch_time.update(time.time() - end)
	# 		end = time.time()
	#
	# 		if i % args.print_freq == 0:
	# 			line = 'Test: [{0}][{1}/{2}]\t' + \
	# 				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' + \
	# 				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t' + \
	# 				  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' + \
	# 				  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
	#
	# 			line = line.format(
	# 				   epoch, i, len(val_loader), batch_time=batch_time, loss=losses,
	# 				   top1=top1, top5=top5)
	#
	# 			if i % args.show_freq == 0:
	# 				print(line)
	#
	# 			log.write('%s\n' % line)
	#
	# if args.tensorboard:  # update the embedding every iteration
	# 	# embedding
	# 	n_iter_val = epoch * len(val_loader)
	#
	# 	writer.add_embedding(feat_val_display, metadata=label_val_display.data, global_step=n_iter_val, tag='validation')
	#
	# print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
	# 	  .format(top1=top1, top5=top5, loss=losses)))
	#


	# print("entropy_cuts for next training epoch:", entropy_cuts)
	return 2*(meter.avg[-1]*meter.avg[:-1].mean()) /(meter.avg[:-1].mean() + meter.avg[-1]), entropy_cuts


def save_checkpoint(state, is_best, path_exp, filename='checkpoint.pth.tar'):

	path_file = path_exp + filename
	torch.save(state, path_file)
	if is_best:
		path_best = path_exp + 'model_best.pth.tar'
		shutil.copyfile(path_file, path_best)

class AverageMeterIvan(object):
    """Computes and stores the average and current value"""

    def __init__(self, num_class):
        self.num_class = num_class

        self.avg = np.zeros(num_class)
        self.sum = np.zeros(num_class)
        self.count = np.zeros(num_class)

    def reset(self):

        self.avg = np.zeros(self.num_class)
        self.sum = np.zeros(self.num_class)
        self.count = np.zeros(self.num_class)

    def update(self, class_idx, val, n=1):
        for i, value in enumerate(class_idx):
            self.sum[value] += val[i] * n
            self.count[value] += n
            self.avg[value] = self.sum[value] / self.count[value]


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, decay):
	"""Sets the learning rate to the initial LR decayed by 10 """
	for param_group in optimizer.param_groups:
		param_group['lr'] /= decay

def adjust_learning_rate_loss(optimizer, decay, stat_current, stat_previous, op):
	ops = {'>': (lambda x, y: x > y), '<': (lambda x, y: x < y), '>=': (lambda x, y: x >= y), '<=': (lambda x, y: x <= y)}
	if ops[op](stat_current, stat_previous):
		for param_group in optimizer.param_groups:
			param_group['lr'] /= decay

def adjust_learning_rate_dann(optimizer, p):
	for param_group in optimizer.param_groups:
		param_group['lr'] = args.lr / (1. + 10 * p) ** 0.75

def loss_adaptive_weight(loss, pred):
	weight = 1 / pred.var().log()
	constant = pred.std().log()
	return loss * weight + constant

def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)

	# print("\n\n", maxk, "\n\n")

	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].contiguous().view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res

# remove dummy tensors
def removeDummy(attn, out_1, out_2, pred_domain, feat, batch_size):
	attn = attn[:batch_size]
	out_1 = out_1[:batch_size]
	out_2 = out_2[:batch_size]
	pred_domain = [pred[:batch_size] for pred in pred_domain]
	feat = [f[:batch_size] for f in feat]

	return attn, out_1, out_2, pred_domain, feat

def EVT(x, th, power=1):
	"""Fit the EVT with the target entropy value, 
	and return value of entropy with probability of GEV's cdf == th"""
	x = np.array(x)**power
	shape, loc, scale = gev.fit(x)
	xx = np.linspace(loc+0.00001-2, loc+0.00001+4, num=1000)
	P = gev.cdf(xx, shape, loc, scale)

	for i,p in enumerate(P):
		if p >= th:
			return i*6/1000 + loc-2

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)


if __name__ == '__main__':

	logger = logging.getLogger()
	logger.setLevel(logging.INFO)  # Setup the root logger.
	optuna.logging.enable_propagation()  # Propagate logs to the root logger.
	optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.


	# logger.addHandler(logging.FileHandler("auto_tune/28-07-auto_osve_random_3.log", mode="w"))

	study = optuna.create_study(direction="maximize",  pruner=optuna.pruners.NopPruner())
	study.optimize(main, n_trials=1) # 1 (no auto tune) | 80 trials / 10 hour
	pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
	complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
	print("Study statistics: ")
	print("  Number of finished trials: ", len(study.trials))
	print("  Number of pruned trials: ", len(pruned_trials))
	print("  Number of complete trials: ", len(complete_trials))
	print("Best trial:")
	trial = study.best_trial

	print("  Value: ", trial.value)
	print("  Params: ")
	for key, value in trial.params.items():
		print("    {}: {}".format(key, value))
	
