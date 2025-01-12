import argparse
import time
import torch
from dataloader_embedding import get_loader, load_data
from Graph_embedding_model_initial_eye import graph_embedding2, kan_embedding
from utils import *
from Graph_embedding_loss import rmse
from torch.utils.tensorboard import SummaryWriter
import shutil
import pdb
import pandas as pd

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device_ids = [0]
device = torch.device("cuda:{}".format(device_ids[0]))
seed_everything(830)
# ours
# name = "ours"
# model_path = "./HCP1064/ours/models/Graph_embedding_model-1.ckpt"
# lamda=3
# name = "lamda3"
# model_path = "./HCP1064/ours_1/models/Graph_embedding_model-1.ckpt"
# lamda=1
# name = "lamda1"
# model_path = "./HCP1064/ours_1/models/Graph_embedding_model-5.ckpt"
# loss
name = "loss"
model_path = "./HCP1064/net/models/Graph_embedding_model-5.ckpt"



def main(args):
    # Create model directory
    # device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    if os.path.exists(args.runs_path):
        shutil.rmtree(args.runs_path)
    os.makedirs(args.runs_path)

    if os.path.exists(args.model_path):
        shutil.rmtree(args.model_path)
    os.makedirs(args.model_path)

    if os.path.exists(args.training_results_path):
        shutil.rmtree(args.training_results_path)
    os.makedirs(args.training_results_path)

    if os.path.exists(args.validation_results_path):
        shutil.rmtree(args.validation_results_path)
    os.makedirs(args.validation_results_path)

    if os.path.exists(args.testing_results_path):
        shutil.rmtree(args.testing_results_path)
    os.makedirs(args.testing_results_path)

    # pdb.set_trace()
    all_data = load_data(args.data_path)

    exp_prec = []
    # Build the models
    exp_str = "exp_" + str(0)
    training_results_path = args.training_results_path + "/" + exp_str
    validation_results_path = args.validation_results_path + "/" + exp_str
    testing_results_path = args.testing_results_path + "/" + exp_str

    if not os.path.exists(training_results_path):
            os.makedirs(training_results_path)
    if not os.path.exists(validation_results_path):
            os.makedirs(validation_results_path)
    if not os.path.exists(testing_results_path):
            os.makedirs(testing_results_path)

    model = kan_embedding(args.embedding_num, args.embedding_dim, args.hot_num)
        # model = graph_embedding2(args.embedding_num, args.embedding_dim, args.hot_num)
    model = model.to(device)

    if args.pretrain == "T":
        model.load_state_dict(torch.load(model_path))

        # for name, param in model.named_parameters():
        #     if name == 'w_node_embedding.weight':
        #         print(name)
        #         print(param)
        #         print(param.shape)

    val_data_loader = get_loader(
            args.data_path,
            all_data,
            False,
            True,
            args.batch_size,
            num_workers=args.num_workers,
            hot_num=args.hot_num,
        )

    val_prec_loss = validate(val_data_loader, model, 0, 0, device)
    print(val_prec_loss)
    del model


def validate(data_loader, model, epoch, exp_num, device):
    # Evaluate the models
    model.eval()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)

    start = time.time()
    loss_list = []
    pcc_list = []
    ssim_list = []

    for i, (DataID, multi_hot_feature) in enumerate(data_loader):
        # pdb.set_trace()
        data_time.update(time.time() - start)
        # hot_num = multi_hot_feature.shape[1]
        multi_hot_feature = multi_hot_feature.to(device).float()

        with torch.no_grad():
            x_decoder, x_embedding, x_de_embedding, _ = model(multi_hot_feature)
            loss = torch.nn.functional.mse_loss(multi_hot_feature, x_decoder)
            pcc = pearson_corrcoef(multi_hot_feature, x_decoder)
            multi_hot_feature=multi_hot_feature.squeeze(0).squeeze(0).unsqueeze(1)
            # print(multi_hot_feature.shape)
            x_decoder= x_decoder.squeeze(0).squeeze(0).unsqueeze(1)
            sim = ssim(multi_hot_feature, x_decoder)
            print("epoch-----:", epoch, "val_loss:", loss)
        loss_list.append(loss.cpu().detach().numpy().squeeze())
        pcc_list.append(pcc.cpu().detach().numpy().squeeze())
        ssim_list.append(sim.cpu().detach().numpy().squeeze())
        losses.update(loss.item())
        batch_time.update(time.time() - start)

        start = time.time()

        # Print log info
        # if i % args.log_step == 0:
    print(len(loss_list))
    print(
        "Val Epoch: [{0}/{1}]\t"
        "Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
        "Loss {loss.val:.4f} ({loss.avg:.4f})".format(
            i, len(data_loader), batch_time=batch_time, loss=losses
        )
    )
    loss_list = pd.DataFrame(loss_list)
    pcc_list = pd.DataFrame(pcc_list)
    ssim_list = pd.DataFrame(ssim_list)
    header = ["loss"]
    print(model_path)
    mse_csv_path = name + "_mse.csv"
    pcc_csv_path = name + "_pcc.csv"
    ssim_csv_path = name + "_ssim.csv"
    loss_list.to_csv(mse_csv_path, header=header, index=False)
    pcc_list.to_csv(pcc_csv_path, header=header, index=False)
    ssim_list.to_csv(ssim_csv_path, header=header, index=False)
    return losses.avg


if __name__ == "__main__":
    seed_everything(830)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device_ids = [0]
    device = torch.device("cuda:{}".format(device_ids[0]))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-time", "--time", type=str, default="HCP", help="path for image data"
    )
    parser.add_argument("--root", type=str, default="./HCP", help="path for image data")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/media/minheng/hdd_3/cmh/dataset/bsse_node_input_data",
        help="path for image data",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./HCP1064/models",
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
    args.model_path = args.root + "/models"
    args.training_results_path = args.root + "/training_results"
    args.validation_results_path = args.root + "/validation_results"
    args.testing_results_path = args.root + "/testing_results"
    args.runs_path = args.root + "/runs"
    print(args)
    main(args)
