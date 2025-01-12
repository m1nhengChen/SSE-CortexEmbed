import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device_ids = [0]
device = torch.device("cuda:{}".format(device_ids[0]))
import argparse
import time
from dataloader_embedding import get_loader, load_data
from Graph_embedding_model_initial_eye import graph_embedding2, kan_embedding
from utils import *
from torch.utils.tensorboard import SummaryWriter
import shutil
import pdb

seed_everything(830)


def main(args):
    # Create model directory
    # pdb.set_trace()

    exp_prec = []
    #  We only need the upper triangular part of the matrix (excluding the diagonal)
    triu_indices = torch.triu_indices(75, 75, offset=1, device=device)
    pcc_mean = torch.zeros([75, 75], device=device)
    for exp_num in range(10):
        # if exp_num>=7 and exp_num<9 :
        #     continue
        # if exp_num==1 or exp_num==3 or exp_num==6 or exp_num==0 or exp_num==9:
        #     continue
        if exp_num==4 or exp_num==5 or exp_num==8 or exp_num==9:
            continue
        # if exp_num!=9:
        #     continue
        # Build the models
        exp_str = "exp_" + str(exp_num)
        para_results_path = args.para_results_path + "/" + exp_str

        if not os.path.exists(para_results_path):
            os.makedirs(para_results_path)

        test_model = kan_embedding(args.embedding_num, args.embedding_dim, args.hot_num)
        # test_model = graph_embedding2(
        #     args.embedding_num, args.embedding_dim, args.hot_num
        # )
        test_model = test_model.to(device)

        pcc_matrix = test(test_model, para_results_path, exp_num)
        pcc_mean += pcc_matrix
        # w_node_embedding.base_weight
        # torch.Size([128, 75])
        # w_node_embedding.spline_weight
        # torch.Size([128, 75, 8])
        # w_node_embedding.spline_scaler
        # torch.Size([128, 75])
        del test_model
    # Extract the PCC values from the upper triangular matrix
    pcc_values = pcc_mean[triu_indices[0], triu_indices[1]]
    # print(pcc_values)
    # print(pcc_values.shape)
    # Get the top 10 pairs of rows with the highest PCC values
    top10_indices = torch.topk(pcc_values, 10).indices
    # print(top10_indices)
    # Get the actual row pairs corresponding to the highest PCC values
    top10_pairs = triu_indices[:, top10_indices]

    # Print the output
    print("Top 10 pairs of rows with highest PCC:")
    print(top10_pairs)


# Calculate Pearson Correlation Coefficient
def pearson_corr(x):
    # Subtract the mean
    x = x - torch.mean(x, dim=1, keepdim=True)
    # Compute the standard deviation
    std_x = torch.std(x, dim=1, keepdim=True)
    # Normalize the input
    x = x / std_x
    # Compute the correlation matrix
    corr_matrix = torch.mm(x, x.T) / x.shape[1]
    return corr_matrix




def test(test_model, prefix, exp_num):
    # Evaluate the models
    # print(args.model_path)
    # print(os.path.join(args.model_path, 'Graph_embedding_model-{}.ckpt'.format(exp_num)))
    test_model.load_state_dict(
        torch.load(
            os.path.join(
                args.model_path, "Graph_embedding_model-{}.ckpt".format(exp_num)
            ),
            weights_only=True
        )
    )
    test_model.eval()
    with torch.no_grad():
        for name, para in test_model.named_parameters():
            # print(name)
            # print(para.size)
            # print(para.shape)
            if name == "w_node_embedding.base_weight":
                # print(name)
                # print(para.size)
                # print(para)
                base_weight = para
                # np.savetxt(prefix + '/' + 'roi_embedding.txt', para.t().cpu().numpy(), fmt='%.2f')
            elif name == "w_node_embedding.spline_weight":
                # print(name)
                # print(para.size)
                # print(para)
                spline_weight = para
                # np.savetxt(prefix + '/' + 'combine_embedding.txt', para.t().cpu().numpy(), fmt='%.2f')
            elif name == "w_node_embedding.spline_scaler":
                # print(name)
                # print(para.size)
                # print(para)
                spline_scaler = para
            elif name == "w_node_embedding.weight":
                base_weight = para
        spline_scale_weight = spline_weight * spline_scaler.unsqueeze(-1)
        # print(spline_scale_weight.shape)
        base_weight = base_weight.transpose(0, 1)
        spline_scale_weight = spline_scale_weight.mean(dim=-1).transpose(0, 1)
        # print(spline_scale_weight.shape)
        # print(base_weight.shape)
        ROI_embedding = base_weight + spline_scale_weight
        # ROI_embedding = base_weight
        # ROI_embedding = spline_scale_weight
        # print(prefix)
        np.savetxt(
            prefix + "/" + "ROI_embedding.txt", ROI_embedding.t().cpu().numpy(), fmt="%.2f"
        )
        # Compute the PCC matrix for rows
        pcc_matrix = pearson_corr(ROI_embedding)
        # print(pcc_matrix)
        # print(pcc_matrix.shape)

        return pcc_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-time", "--time", type=str, default="HCP", help="path for image data"
    )
    parser.add_argument(
        "--root", type=str, default="./HCP1064", help="path for image data"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/media/minheng/hdd_3/cmh/dataset/bsse_node_input_data",
        help="path for image data",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./HCP1064/ours/models",
        help="path for saving trained models",
    )
    parser.add_argument(
        "--training_results_path",
        type=str,
        default="./HCP1064/training_results",
        help="",
    )
    parser.add_argument(
        "--validation_results_path",
        type=str,
        default="./HCP1064/validation_results",
        help="",
    )
    parser.add_argument(
        "--testing_results_path", type=str, default="./HCP1064/testing_results", help=""
    )
    parser.add_argument(
        "--runs_path", type=str, default="./HCP1064/runs", help="path for runs"
    )
    parser.add_argument(
        "--para_results_path",
        type=str,
        default="./HCP1064/ours/runs",
        help="path for runs",
    )
    parser.add_argument(
        "--log_step", type=int, default=1, help="step size for prining log info"
    )
    parser.add_argument(
        "--save_step", type=int, default=1, help="step size for saving trained models"
    )
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--model_lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience", type=float, default=10)
    parser.add_argument(
        "--save_mse_path",
        type=str,
        default="./HCP1064/validation_results",
        help="",
    )

    parser.add_argument(
        "--lamda",
        type=float,
        default=2,
        help="hyperparameter for non-zero element weight",
    )
    parser.add_argument("--hot_num", type=int, default=4)
    parser.add_argument("--embedding_num", type=int, default=75)
    parser.add_argument("--embedding_dim", type=int, default=128)

    parser.add_argument("-pretrain", "--pretrain", type=str, default="T")
    parser.add_argument(
        "-pretrain_path",
        "--pretrain_path",
        type=str,
        default="/media/qidianzl/zl_1/zl/few_shot/Model/hop_2",
    )
    parser.add_argument(
        "-pretrain_model",
        "--pretrain_model",
        type=str,
        default="2",
        help="2 or 3 for hop_1",
    )

    parser.add_argument(
        "-gpu_id", "--gpu_id", type=int, default=0, help="path for image data"
    )

    args = parser.parse_args()

    # pre_model_list = ['1', '3', '4', '5', '6', '7']

    # for pre_model in pre_model_list:
    #     args.pretrain_model = pre_model
    #     # args.data_path = args.root + '/' + 'Graph_embedding_data_' + args.time
    #     args.model_path = './' + args.time + '_HCP-pre-' + args.pretrain_model + '/models/'
    #     args.training_results_path = './' + args.time + '_HCP-pre-' + args.pretrain_model + '/training_results'
    #     args.validation_results_path = './' + args.time + '_HCP-pre-' + args.pretrain_model + '/validation_results'
    #     args.testing_results_path = './' + args.time + '_HCP-pre-' + args.pretrain_model + '/testing_results'
    #     args.runs_path = './' + args.time + '_HCP-pre-' + args.pretrain_model + '/runs'
    # args.model_path = args.root + "/models"
    args.training_results_path = args.root + "/training_results"
    args.validation_results_path = args.root + "/validation_results"
    args.testing_results_path = args.root + "/testing_results"
    args.runs_path = args.root + "/runs"
    print(args)
    main(args)
