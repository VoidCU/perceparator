from adamp import AdamP, SGDP
import numpy as np
import json5
from model.sepformer import Sepformer
from src.trainer import Trainer
from dataset.data import DataLoader, MyDataset
import torch


def main(config):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # # 数据
    # tr_dataset = MyDataset(json_dir=config["train_dataset"]["train_dir"],  # 目录下包含 mix.json, s1.json, s2.json
    #                           batch_size=config["train_dataset"]["batch_size"],
    #                           sample_rate=config["train_dataset"]["sample_rate"],  # 采样率
    #                           segment=config["train_dataset"]["segment"])  # 语音时长

    # cv_dataset = MyDataset(json_dir=config["validation_dataset"]["validation_dir"],
    #                           batch_size=config["validation_dataset"]["batch_size"],
    #                           sample_rate=config["validation_dataset"]["sample_rate"],
    #                           segment=config["validation_dataset"]["segment"],
    #                           cv_max_len=config["validation_dataset"]["cv_max_len"])

    # tr_loader = DataLoader(tr_dataset,
    #                             batch_size=config["train_loader"]["batch_size"],
    #                             shuffle=config["train_loader"]["shuffle"],
    #                             num_workers=config["train_loader"]["num_workers"])

    # cv_loader = DataLoader(cv_dataset,
    #                             batch_size=config["validation_loader"]["batch_size"],
    #                             shuffle=config["validation_loader"]["shuffle"],
    #                             num_workers=config["validation_loader"]["num_workers"])

    # Modified by US(Thapathalians)
    tr_dataset = MyDataset(
        config["train_dataset"]["train_dir"], config["train_dataset"]["batch_size"], config["train_dataset"]["sample_rate"], config["train_dataset"]["segment"])
    cv_dataset = MyDataset(config["validation_dataset"]["validation_dir"], config["validation_dataset"]["batch_size"],
                           config["validation_dataset"]["sample_rate"], config["validation_dataset"]["segment"])

    tr_loader = DataLoader(tr_dataset,
                           batch_size=config["train_loader"]["batch_size"],
                           shuffle=config["train_loader"]["shuffle"],
                           num_workers=config["train_loader"]["num_workers"])

    cv_loader = DataLoader(cv_dataset,
                           batch_size=config["validation_loader"]["batch_size"],
                           shuffle=config["validation_loader"]["shuffle"],
                           num_workers=config["validation_loader"]["num_workers"])

    data = {"tr_loader": tr_loader, "cv_loader": cv_loader}

    # 模型
    if config["model"]["type"] == "sepformer":
        model = Sepformer(N=config["model"]["sepformer"]["N"],
                          C=config["model"]["sepformer"]["C"],
                          L=config["model"]["sepformer"]["L"],
                          H=config["model"]["sepformer"]["H"],
                          K=config["model"]["sepformer"]["K"],
                          Global_B=config["model"]["sepformer"]["Global_B"],
                          Local_B=config["model"]["sepformer"]["Local_B"])
    else:
        print("No loaded model!")

    # if torch.cuda.is_available():
    #     model = torch.nn.DataParallel(model)
    #     model.cuda()

    if config["optimizer"]["type"] == "sgd":
        optimize = torch.optim.SGD(
            params=model.parameters(),
            lr=config["optimizer"]["sgd"]["lr"],
            momentum=config["optimizer"]["sgd"]["momentum"],
            weight_decay=config["optimizer"]["sgd"]["l2"])
    elif config["optimizer"]["type"] == "adam":
        optimize = torch.optim.Adam(
            params=model.parameters(),
            lr=config["optimizer"]["adam"]["lr"],
            betas=(config["optimizer"]["adam"]["beta1"], config["optimizer"]["adam"]["beta2"]))
    elif config["optimizer"]["type"] == "sgdp":
        optimize = SGDP(
            params=model.parameters(),
            lr=config["optimizer"]["sgdp"]["lr"],
            weight_decay=config["optimizer"]["sgdp"]["weight_decay"],
            momentum=config["optimizer"]["sgdp"]["momentum"],
            nesterov=config["optimizer"]["sgdp"]["nesterov"],
        )
    elif config["optimizer"]["type"] == "adamp":
        optimize = AdamP(
            params=model.parameters(),
            lr=config["optimizer"]["adamp"]["lr"],
            betas=(config["optimizer"]["adamp"]["beta1"],
                   config["optimizer"]["adamp"]["beta2"]),
            weight_decay=config["optimizer"]["adamp"]["weight_decay"],
        )
    else:
        print("Not support optimizer")
        return

    trainer = Trainer(data, model, optimize, config)

    trainer.train()


if __name__ == '__main__':
    path = r"config/train/train.json5"
    with open(path) as f:
        configuration = json5.load(f)
    main(configuration)
