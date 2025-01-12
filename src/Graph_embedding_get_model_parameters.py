import argparse
import time
import torch
from dataloader_embedding import get_loader, load_data
from Graph_embedding_model_initial_eye import graph_embedding2
from utils import *
import os
from torch.utils.tensorboard import SummaryWriter
import shutil
import pdb

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def main(args):
	# Create model directory
	# pdb.set_trace()

	exp_prec = []
	for exp_num in range(10):
		# Build the models
		exp_str = "exp_" + str(exp_num)
		para_results_path = args.para_results_path + '/' + exp_str

		if not os.path.exists(para_results_path):
			os.makedirs(para_results_path)


		test_model = graph_embedding2(args.embedding_num, args.embedding_dim, args.hot_num)
		test_model = test_model.to(device)

		test(test_model, para_results_path, exp_num)

		del test_model


def test(test_model, prefix, exp_num):
	# Evaluate the models
	test_model.load_state_dict(torch.load(os.path.join(args.model_path, 'Graph_embedding_model-{}.ckpt'.format(exp_num))))
	test_model.eval()
	
	with torch.no_grad():
		for name, para in test_model.named_parameters():
			if name == 'w_node_embedding.weight':
				print(name)
				print(para.size)
				print(para)
				np.savetxt(prefix + '/' + 'roi_embedding.txt', para.t().cpu().numpy(), fmt='%.2f')
			elif name == 'embedding_combine.weight':
				print(name)
				print(para.size)
				print(para)
				np.savetxt(prefix + '/' + 'combine_embedding.txt', para.t().cpu().numpy(), fmt='%.2f')


	


if __name__ == '__main__':
	seed_everything(830)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device_ids = [0]
    device = torch.device("cuda:{}".format(device_ids[0]))
	parser = argparse.ArgumentParser()
	parser.add_argument('-time', '--time', type=str, default='dHCP31_33', help='path for image data')
	parser.add_argument('--root', type=str, default='/media/qidianzl/zl_1/zl/few_shot/few_shot_Embedding/Data', help='path for image data')
	parser.add_argument('--data_path', type=str, default='/Graph_embedding_data_dHCP31_33', help='path for image data')
	parser.add_argument('--model_path', type=str, default='./dHCP31_33/models/', help='path for saving trained models')
	parser.add_argument('--training_results_path', type=str, default='./dHCP31_33/training_results', help='')
	parser.add_argument('--validation_results_path', type=str, default='./dHCP31_33/validation_results',help='')
	parser.add_argument('--testing_results_path', type=str, default='./dHCP31_33/testing_results', help='')
	parser.add_argument('--runs_path', type=str, default='./dHCP31_33/runs', help='path for runs')
	parser.add_argument('--para_results_path', type=str, default='./dHCP31_33/para_results', help='')

	parser.add_argument('--log_step', type=int, default=1, help='step size for prining log info')
	parser.add_argument('--save_step', type=int, default=1, help='step size for saving trained models')
	parser.add_argument('--num_epochs', type=int, default=500)
	parser.add_argument('--batch_size', type=int, default=256)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--model_lr', type=float, default=1e-3)
	parser.add_argument('--beta1', type=float, default=0.5)
	parser.add_argument('--beta2', type=float, default=0.999)
	parser.add_argument('--weight_decay', type=float, default=0.01)
	parser.add_argument('--patience', type=float, default=10)

	parser.add_argument('--hot_num', type=int, default=3)
	parser.add_argument('--embedding_num', type=int, default=75)
	parser.add_argument('--embedding_dim', type=int, default=128)

	parser.add_argument('-pretrain', '--pretrain', type=str, default='T')
	parser.add_argument('-pretrain_path', '--pretrain_path', type=str, default='/media/qidianzl/zl_1/zl/few_shot/Model/hop_2')
	parser.add_argument('-pretrain_model', '--pretrain_model', type=str, default='2', help='2 or 3 for hop_1')

	parser.add_argument('-gpu_id', '--gpu_id', type=int, default=0, help='path for image data')

	args = parser.parse_args()

	pre_model_list = ['1', '3', '4', '5', '6', '7']

	for pre_model in pre_model_list:
		args.pretrain_model = pre_model
		args.data_path = args.root + '/' + 'Graph_embedding_data_' + args.time
		args.model_path = './' + args.time + '_HCP-pre-' + args.pretrain_model + '/models/'
		args.training_results_path = './' + args.time + '_HCP-pre-' + args.pretrain_model + '/training_results'
		args.validation_results_path = './' + args.time + '_HCP-pre-' + args.pretrain_model + '/validation_results'
		args.testing_results_path = './' + args.time + '_HCP-pre-' + args.pretrain_model + '/testing_results'
		args.runs_path = './' + args.time + '_HCP-pre-' + args.pretrain_model + '/runs'
		args.para_results_path = './' + args.time + '_HCP-pre-' + args.pretrain_model + '/para_results'

		print(args)
		main(args)