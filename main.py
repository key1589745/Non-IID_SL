# -*- coding:utf-8 -*-
import torch
from torch.backends import cudnn
import argparse, os, warnings
from utils import current_time, modify_modules
from network import ResUnet
from dataset import getTrainLoader, getValiLoader, getTestLoader, train_val_split
from train import *
from test import test_model
warnings.filterwarnings("ignore")


def main(configs):
    
        
    # configure the dataset
    centers = ['cA','cB','cC','cD']
    labels = [[2,3],[1,2],[0],[3]] 

    # load dataset
    train_loaders = {}
    vali_loaders = {}
    test_loaders = {}
    for center, label in zip(centers,labels):
        train, validation, test = train_val_split(configs.dataroot+center) 
        train_loaders[center] = getTrainLoader(train,configs,label)
        vali_loaders[center] = getValiLoader(validation,configs)
        test_loaders[center] = getTestLoader(test)

        
    # network param
    net = ResUnet(configs)
    if configs.mode == 'nsl':
        # modify the batchnorm module to store the sample mean and variance
        for name, module in net.named_modules():
            if 'decoder' in name and 'bn3' in name and isinstance(module, nn.BatchNorm2d):
                att_stack = name.split('.')
                net = modify_modules(net,att_stack)         
    # save dir
    cur_path = os.path.abspath(os.curdir)
    SAVE_DIR =  cur_path + '/result/' + configs.mode + '_' + configs.loss + '_vis/'
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # training
    if configs.mode == 'fedavg': # swarm learning is similar to fedavg
        trainer = Trainer_FedAvg(net, train_loaders, vali_loaders, configs, SAVE_DIR, labels)
    elif configs.mode == 'naive': # a baseline method for personalized federated learning
        trainer = Trainer_Naive(net, train_loaders, vali_loaders, configs, SAVE_DIR, labels)
    elif configs.mode == 'nsl': # the proposed Non-IID swarm learning method
        trainer = Trainer_NSL(net, train_loaders, vali_loaders, configs, SAVE_DIR, labels)
    elif configs.mode == 'ditto':
        trainer = Trainer_Ditto(net, train_loaders, vali_loaders, configs, SAVE_DIR, labels)
    elif configs.mode == 'fedbn':
        trainer = Trainer_FedBN(net, train_loaders, vali_loaders, configs, SAVE_DIR, labels)
    elif configs.mode == 'fedprox':
        trainer = Trainer_FedProx(net, train_loaders, vali_loaders, configs, SAVE_DIR, labels)
    elif configs.mode == 'fedcurv':
        trainer = Trainer_FedEWC(net, train_loaders, vali_loaders, configs, SAVE_DIR, labels)
    elif configs.mode == 'feddml':
        trainer = Trainer_FedDML(net, train_loaders, vali_loaders, configs, SAVE_DIR, labels)
    elif configs.mode == 'pseudo':
        trainer = Trainer_Pseudo(net, train_loaders, vali_loaders, configs, SAVE_DIR, labels)
    else:
        raise NotImplementedError
        
    models = trainer.train()

    
    # testing
    np.save('result/{}_{}_old.npy'.format(configs.mode,configs.loss), trainer.vis_old, allow_pickle=True)
    np.save('result/{}_{}_new.npy'.format(configs.mode,configs.loss), trainer.vis_new, allow_pickle=True)
    test_model(models, test_loaders, SAVE_DIR)
    for client, model in models.items():
        torch.save(model,SAVE_DIR+client+'.pth')
            
    print('Train finished: ', current_time())



if __name__ == '__main__':

    # set parameters
    parser = argparse.ArgumentParser()

        # dataset param
    parser.add_argument('--dataroot', type=str, default='MnM/')
    parser.add_argument('--num_data', type=int, default=1000,help='number of data sampled for one epoch')
    parser.add_argument('--num_cls', type=int, default=4)
    parser.add_argument('--size', type=int, default=192)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)

        # network param
            # backbone
    parser.add_argument('--return_interm_layers', type=bool, default=False)
    parser.add_argument('--hidden_dim', type=int, default=256, help='backbone feature')
    parser.add_argument('--name', type=str, default='resnet18')
    parser.add_argument('--skip', type=bool, default=True)

        # train param
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--local_epoch', type=int, default=1)
    parser.add_argument('--mode', type=str, default='nsl')
    parser.add_argument('--loss', type=str, default='UNCE')
    parser.add_argument('--cuda', type=int, default=3)

    CONFIGs = parser.parse_args()
    torch.cuda.set_device(CONFIGs.cuda)
    
    #os.environ["CUDA_VISIBLE_DEVICES"] = CONFIGs.cuda
    cudnn.benchmark = True

    print(CONFIGs)
    main(CONFIGs)