# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from utils import Poly
from evaluation import knowledge_metric, Evaluator
from losses import get_loss, KL_distill_loss, L2_penalty, FeSA_loss
import numpy as np
import copy


class Trainer_FedAvg(object):
    def __init__(self,model,data_loader_train,data_loader_vali,config,save_dir,labels=None):

        # models
        self.models = {client:copy.deepcopy(model).cuda() \
                       for client, data in data_loader_train.items()}

        self.model_cache = copy.deepcopy(model).cuda()

        self.best_models = {}

        self.client_weights = {c: torch.tensor(1/len(data_loader_train)).cuda() \
                               for c, data in data_loader_train.items()}

        # train
        self.lr = config.lr
        self.epochs = config.epochs
        self.local_epoch = config.local_epoch
        self.num_data = config.num_data
        self.iters_per_epoch = self.num_data // config.batch_size
        self.labels = dict(zip(self.models.keys(),labels))
        self.criterion = {}
        if labels is not None:
            for client, label in zip(data_loader_train.keys(), labels):

                self.criterion[client] = get_loss(config.loss,labels=label).cuda()
        else:
            self.criterion = {client:get_loss(config.loss).cuda() for client in data_loader_train.keys()}
        self.train_loaders = data_loader_train
        self.vali_loaders = data_loader_vali
        self.optimizers = {client:torch.optim.Adam([{'params': model.parameters()}],lr = self.lr,weight_decay=1e-4) \
                           for client, model in self.models.items()}

        # evaluate
        self.epochs_per_vali = 10
        self.evaluator = Evaluator(data_loader_vali,config.num_cls)
        # save result
        # visualization of the knowledge forgetting
        self.vis_old = {client:{'mean':np.zeros((300,self.iters_per_epoch)), \
                        'std':np.zeros((300,self.iters_per_epoch))} \
                        for client in self.models.keys()}
        self.vis_new = {client:{'mean':np.zeros((300,self.iters_per_epoch)), \
                        'std':np.zeros((300,self.iters_per_epoch))} \
                        for client in self.models.keys()}
        self.visor = knowledge_metric(data_loader_vali,config.num_cls)
        self.save_dir = save_dir


    def train(self):

        max_dice = 0

        lr_scheduler = {c:Poly(optimizer=op, num_epochs=self.epochs,iters_per_epoch=self.iters_per_epoch) \
                        for c, op in self.optimizers.items()}

        for epoch in range(self.epochs):

            for l_epoch in range(self.local_epoch):

                for client in self.train_loaders.keys():

                    self.train_local(client,epoch)

                    lr_scheduler[client].step(epoch=epoch)

            self.communication()

            # evaluate
            if epoch % self.epochs_per_vali==self.epochs_per_vali-1:

                meanDice = 0.
                for client in self.models.keys():
                    dice_dict = self.evaluator.eval(self.model_cache,client)
                    print(epoch, client, dice_dict)
                    meanDice += np.mean(list(dice_dict.values()))

                if meanDice >= max_dice:
                    max_dice = meanDice
                    for client in self.models.keys():
                        self.best_models[client] = copy.deepcopy(self.model_cache)

        return self.best_models


    def communication(self):

        with torch.no_grad():
                # FedAvg
            for key in self.model_cache.state_dict().keys():
                if 'num_batches_tracked' not in key:
                    temp = torch.zeros_like(self.model_cache.state_dict()[key])
                    # aggregate parameters
                    for client, model in self.models.items():
                        temp += self.client_weights[client] * model.state_dict()[key]
                    self.model_cache.state_dict()[key].data.copy_(temp)
                    for client, model in self.models.items():
                        self.models[client].state_dict()[key].data.copy_(temp)


    def train_local(self,client,epoch):

        self.models[client].train()

        for step in range(self.iters_per_epoch):

            self.visualization(client, epoch, step)
            self.optimizers[client].zero_grad()
            train_batch = next(self.train_loaders[client])
            imgs = torch.from_numpy(train_batch['data']).cuda(non_blocking=True)
            labs = torch.from_numpy(train_batch['seg']).type(torch.LongTensor).cuda(non_blocking=True)
            output = self.models[client](imgs)
            if len(labs.shape) == len(output.shape):
                labs = labs[:, 0]
            loss = self.criterion[client](output, labs)
            loss.backward()
            self.optimizers[client].step()

    def visualization(self, client, epoch, step):
        if epoch>100 and epoch<300 and epoch % 10 == 5:
            un_labels = [i for i in range(1,4) if i not in self.labels[client]]
            meanDice = []
            for c in self.models.keys():
                if c != client:
                    meanDice.append(self.visor.eval(self.models[client],c,un_labels))

            meanDice = np.concatenate(meanDice)
            self.vis_old[client]['mean'][epoch,step] = np.mean(meanDice)
            self.vis_old[client]['std'][epoch,step] = np.std(meanDice)

            dice = self.visor.eval(self.models[client],client,self.labels[client])
            self.vis_new[client]['mean'][epoch,step] = np.mean(dice)
            self.vis_new[client]['std'][epoch,step] = np.std(dice)
            self.models[client].train()


class Trainer_NSL(Trainer_FedAvg):
    # Proposed Non-IID swarm learning method
    def __init__(self,model,data_loader_train,data_loader_vali,config,save_dir,labels=None):
        super().__init__(model,data_loader_train,data_loader_vali,config,save_dir,labels)

        self.weight = config.weight


    def train_local(self,client,epoch):

        old_stats = []
        for name, module in self.model_cache.named_modules():
            if 'decoder' in name and 'bn3' in name and isinstance(module, nn.BatchNorm2d):
                old_stats.append({'mean': module.running_mean, 'var': module.running_var})

        self.models[client].train()
        for step in range(self.iters_per_epoch):
            self.optimizers[client].zero_grad()
            train_batch = next(self.train_loaders[client])
            imgs = torch.from_numpy(train_batch['data']).cuda(non_blocking=True)
            labs = torch.from_numpy(train_batch['seg']).type(torch.LongTensor).cuda(non_blocking=True)
            output = self.models[client](imgs)
            if len(labs.shape) == len(output.shape):
                labs = labs[:, 0]


            new_stats = []
            for name, module in self.models[client].named_modules():
                # constrain the statistics in bn3 of decoders
                if 'decoder' in name and 'bn3' in name and isinstance(module, nn.BatchNorm2d):
                    new_stats.append(module.stats)

            loss = self.criterion[client](output, labs)

            feature_loss = self.weight * FeSA_loss(old_stats,new_stats)

            loss += feature_loss

            loss.backward()
            self.optimizers[client].step()

        for name, module in self.models[client].named_modules():
            if 'decoder' in name and 'bn3' in name and isinstance(module, nn.BatchNorm2d):
                module.stats = None


class Trainer_FedProx(Trainer_FedAvg):

    def __init__(self,model,data_loader_train,data_loader_vali,config,save_dir,labels=None):
        super().__init__(model,data_loader_train,data_loader_vali,config,save_dir,labels)

        self.weight = config.weight
        self.omega = defaultdict(lambda: 1)


    def train_local(self,client):

        self.models[client].train()
        for step in range(self.iters_per_epoch):

            self.optimizers[client].zero_grad()
            train_batch = next(self.train_loaders[client])
            imgs = torch.from_numpy(train_batch['data']).cuda(non_blocking=True)
            labs = torch.from_numpy(train_batch['seg']).type(torch.LongTensor).cuda(non_blocking=True)
            output = self.models[client](imgs)
            if len(labs.shape) == len(output.shape):
                labs = labs[:, 0]

            loss = self.criterion[client](output, labs)

            distill_loss = self.weight * L2_penalty(self.models[client], self.model_cache, self.omega)

            loss += distill_loss

            loss.backward()
            self.optimizers[client].step()


class Trainer_FedEWC(Trainer_FedProx):

    def train(self):

        max_dice = defaultdict(lambda: 0.)

        lr_scheduler = {c:Poly(optimizer=op, num_epochs=self.epochs,iters_per_epoch=self.iters_per_epoch) \
                        for c, op in self.optimizers.items()}

        for epoch in range(self.epochs):

            for l_epoch in range(self.local_epoch):

                for client in self.train_loaders.keys():

                    self.train_local(client)
                    self.diag_fisher(client)
                    lr_scheduler[client].step(epoch=epoch)

            self.communication()


            # evaluate
            if epoch % self.epochs_per_vali==self.epochs_per_vali-1:

                for client, model in self.models.items():
                    dice_dict = self.evaluator.eval(model,client)
                    print(epoch, client, dice_dict)
                    meanDice = np.mean(list(dice_dict.values()))
                    if meanDice >= max_dice[client]:
                        max_dice[client] = meanDice
                        self.best_models[client] = copy.deepcopy(model)

        return self.best_models

    def diag_fisher(self, client):

        precision_matrices = {n: torch.zeros_like(p, dtype=torch.float32).cuda() \
                              for n, p in self.models[client].named_parameters() if p.requires_grad}


        for step in range(self.iters_per_epoch):
            self.model_cache.train()
            self.model_cache.zero_grad()
            train_batch = next(self.train_loaders[client])
            imgs = torch.from_numpy(train_batch['data']).cuda(non_blocking=True)
            labs = torch.from_numpy(train_batch['seg']).type(torch.LongTensor).cuda(non_blocking=True)
            output = self.model_cache(imgs)
            if len(labs.shape) == len(output.shape):
                labs = labs[:, 0]

            loss = self.criterion[client](output, labs)

            loss.backward()

            for n, p in self.models[client].named_parameters():
                if p.grad is not None:
                    precision_matrices[n] += p.grad.data ** 2 / (self.num_data)

        self.omega = {n: p for n, p in precision_matrices.items()}
        self.model_cache.zero_grad()



class Trainer_Pseudo(Trainer_FedAvg):

    def train_local(self,client,epoch):

        self.models[client].train()

        for step in range(self.iters_per_epoch):
            self.optimizers[client].zero_grad()
            train_batch = next(self.train_loaders[client])
            imgs = torch.from_numpy(train_batch['data']).cuda(non_blocking=True)
            labs = torch.from_numpy(train_batch['seg']).type(torch.LongTensor).cuda(non_blocking=True)
            output = self.models[client](imgs)
            if len(labs.shape) == len(output.shape):
                labs = labs[:, 0]
            if epoch > 10:
                labs = self.get_pseudo_labs(imgs,labs,client)
            loss = self.criterion[client](output, labs)
            loss.backward()
            self.optimizers[client].step()

        self.lr_scheduler[client].step(epoch=epoch)


    def get_pseudo_labs(self, imgs, labs, client):
        self.model_cache.eval()
        pseudo_labs = [i for i in range(8) if i not in self.labels[client]]
        if len(pseudo_labs) > 0:
            out = self.model_cache(imgs).detach()
            p, c = torch.max(out, 1, keepdim=False)
            for i in self.labels[client]:
                c[labs==i] = 0

            if 0 in self.labels[client]:
                labs = torch.zeros_like(labs)
            else:
                labs = labs.clone()

            for i in pseudo_labs:
                #labs[c==i] = 0
                labs += (c==i) * i * (p>0.5)

        return labs


#################################### Persoanlized method ############################################
#####################################################################################################


class Trainer_Naive(Trainer_FedAvg):

    def train(self):

        max_dice = defaultdict(lambda: 0.)

        lr_scheduler = {c:Poly(optimizer=op, num_epochs=self.epochs,iters_per_epoch=self.iters_per_epoch) \
                        for c, op in self.optimizers.items()}

        for epoch in range(self.epochs):

            for l_epoch in range(self.local_epoch):

                for client in self.train_loaders.keys():

                    self.train_local(client,epoch)

                    lr_scheduler[client].step(epoch=epoch)

            self.communication()


            # evaluate
            if epoch % self.epochs_per_vali==self.epochs_per_vali-1:

                for client, model in self.models.items():
                    dice_dict = self.evaluator.eval(model,client)
                    print(epoch, client, dice_dict)
                    meanDice = np.mean(list(dice_dict.values()))
                    if meanDice >= max_dice[client]:
                        max_dice[client] = meanDice
                        self.best_models[client] = copy.deepcopy(self.model_cache)

        return self.best_models


class Trainer_FedBN(Trainer_Naive):

    def communication(self):

        with torch.no_grad():
                # FedAvg
            for key in self.model_cache.state_dict().keys():
                if 'bn' not in key and 'downsample.1' not in key:
                    temp = torch.zeros_like(self.model_cache.state_dict()[key])
                    # aggregate parameters
                    for client, model in self.models.items():
                        temp += self.client_weights[client] * model.state_dict()[key]
                    self.model_cache.state_dict()[key].data.copy_(temp)
                    for client, model in self.models.items():
                        self.models[client].state_dict()[key].data.copy_(temp)



class Trainer_Ditto(Trainer_Naive):

    def __init__(self,model,data_loader_train,data_loader_vali,config,save_dir,labels=None):
        super().__init__(model,data_loader_train,data_loader_vali,config,save_dir,labels)

        self.weight = config.weight

        self.local_model = {client:copy.deepcopy(model).cuda() \
                            for client, data in data_loader_train.items()}

        self.local_optimizers = {client:torch.optim.Adam([{'params': model.parameters()}],lr = self.lr,weight_decay=1e-4) \
                           for client, model in self.local_model.items()}


    def train(self):

        max_dice = defaultdict(lambda: 0.)

        lr_scheduler = {c:Poly(optimizer=op, num_epochs=self.epochs,iters_per_epoch=self.iters_per_epoch) \
                        for c, op in self.optimizers.items()}
        local_scheduler = {c:Poly(optimizer=op, num_epochs=self.epochs,iters_per_epoch=self.iters_per_epoch) \
                        for c, op in self.local_optimizers.items()}

        for epoch in range(self.epochs):

            for l_epoch in range(self.local_epoch):

                for client in self.train_loaders.keys():

                    self.train_local(client,epoch)

                    lr_scheduler[client].step(epoch=epoch)

            self.communication()

            for client in self.train_loaders.keys():
                self.local_distillation(client)
                local_scheduler[client].step(epoch=epoch)


            # evaluate
            if epoch % self.epochs_per_vali==self.epochs_per_vali-1:

                for client, model in self.local_model.items():
                    dice_dict = self.evaluator.eval(model,client)
                    print(epoch, client, dice_dict)
                    meanDice = np.mean(list(dice_dict.values()))
                    if meanDice >= max_dice[client]:
                        max_dice[client] = meanDice
                        self.best_models[client] = copy.deepcopy(model)

        return self.best_models


    def local_distillation(self, client):

        self.local_model[client].train()

        for step in range(self.iters_per_epoch):
            self.local_optimizers[client].zero_grad()
            train_batch = next(self.train_loaders[client])
            imgs = torch.from_numpy(train_batch['data']).cuda(non_blocking=True)
            labs = torch.from_numpy(train_batch['seg']).type(torch.LongTensor).cuda(non_blocking=True)
            output = self.local_model[client](imgs)
            if len(labs.shape) == len(output.shape):
                labs = labs[:, 0]
            loss = self.criterion[client](output, labs)
            loss.backward()
            for p_local, p_global in zip(self.local_model[client].parameters(),self.model_cache.parameters()):
                if p_local.grad is not None:
                    diff = p_local.data-p_global.data
                    p_local.grad += self.weight * diff
            self.local_optimizers[client].step()



class Trainer_FedDML(Trainer_Naive):

    def __init__(self,model,data_loader_train,data_loader_vali,config,save_dir,labels=None):
        super().__init__(model,data_loader_train,data_loader_vali,config,save_dir,labels)

        self.weight_m = config.weight
        self.weight_l = config.weight
        self.kd_loss = KL_distill_loss()
        self.local_model = {client:copy.deepcopy(model).cuda() \
                    for client, data in data_loader_train.items()}

        self.local_optimizers = {client:torch.optim.Adam([{'params': model.parameters()}],lr = self.lr,weight_decay=1e-4) \
                           for client, model in self.local_model.items()}



    def train_local(self,client):

        self.local_model[client].train()
        self.models[client].train()
        for step in range(self.iters_per_epoch):

            self.optimizers[client].zero_grad()
            self.local_optimizers[client].zero_grad()
            train_batch = next(self.train_loaders[client])
            imgs = torch.from_numpy(train_batch['data']).cuda(non_blocking=True)
            labs = torch.from_numpy(train_batch['seg']).type(torch.LongTensor).cuda(non_blocking=True)
            output = self.models[client](imgs)
            output_local = self.local_model[client](imgs)
            if len(labs.shape) == len(output.shape):
                labs = labs[:, 0]

            meme_loss = self.criterion[client](output, labs)
            local_loss = self.criterion[client](output_local, labs)

            meme_KL = self.weight_m * self.kd_loss(output,output_local.detach())
            local_KL = self.weight_l * self.kd_loss(output_local,output.detach())

            meme_loss += meme_KL
            local_loss += local_KL

            meme_loss.backward()
            local_loss.backward()
            self.optimizers[client].step()
            self.local_optimizers[client].step()



class test_Trainer(object):
    def __init__(self,model,data_loader_train,data_loader_vali,config,save_dir,labels=None):

        self.models = {client:copy.deepcopy(model).cuda() \
                       for client in data_loader_train.keys()}

        self.model_cache = copy.deepcopy(model).cuda()

        self.client_weights = {c: torch.tensor(1/len(data_loader_train)).cuda() \
                               for c, data in data_loader_train.items()}

        # train
        self.lr = config.lr
        self.epochs = config.epochs
        self.local_epoch = config.local_epoch
        self.num_data = config.num_data
        self.iters_per_epoch = self.num_data // config.batch_size
        self.criterion = {}
        if labels is not None:
            for client, label in zip(data_loader_train.keys(), labels):

                self.criterion[client] = get_loss(config.loss,labels=label).cuda()
        else:
            self.criterion = {client:get_loss(config.loss).cuda() for client in data_loader_train.keys()}

        self.train_loaders = data_loader_train
        self.optimizers = {client:torch.optim.Adam([{'params': model.parameters()}],lr = self.lr,weight_decay=1e-4) \
                           for client, model in self.models.items()}


        # evaluate
        self.epochs_per_vali = 10
        self.evaluator = Evaluator(data_loader_vali)
        # save result
        self.save_dir = save_dir


    def train(self):

        max_dice = defaultdict(lambda: 0.)

        lr_scheduler = {c:Poly(optimizer=op, num_epochs=self.epochs,iters_per_epoch=self.iters_per_epoch) \
                        for c, op in self.optimizers.items()}

        for epoch in range(self.epochs):

            for l_epoch in range(self.local_epoch):

                for client in self.train_loaders.keys():

                    self.train_local(client,epoch)

                    lr_scheduler[client].step(epoch=epoch)


            self.model_aggregation()


            # evaluate
            if epoch % self.epochs_per_vali==self.epochs_per_vali-1:
                for client, model in self.models.items():
                    dice_dict = self.evaluator.eval(model, client)
                    print(epoch, client, dice_dict)
                    meanDice = np.mean(list(dice_dict.values()))
                    if meanDice >= max_dice[client]:
                        max_dice[client] = meanDice
                        self.best_models[client] = copy.deepcopy(model)

        return self.best_model



    def model_aggregation(self):

        with torch.no_grad():
                # FedAvg
            for key in self.model_cache.state_dict().keys():
                if 'num_batches_tracked' not in key:
                #if 'num_batches_tracked' not in key and 'seg_head' not in key:
                #if 'num_batches_tracked' not in key and 'decoder' not in key:
                    temp = torch.zeros_like(self.model_cache.state_dict()[key])
                    # aggregate parameters
                    for client, model in self.models.items():
                        temp += self.client_weights[client] * model.state_dict()[key]
                    self.model_cache.state_dict()[key].data.copy_(temp)
                    for client, model in self.models.items():
                        self.models[client].state_dict()[key].data.copy_(temp)


    def train_local(self,client, epoch):

        self.models[client].train()

        for step in range(self.iters_per_epoch):
            self.optimizers[client].zero_grad()
            train_batch = next(self.train_loaders[client])
            imgs = torch.from_numpy(train_batch['data']).cuda(non_blocking=True)
            labs = torch.from_numpy(train_batch['seg']).type(torch.LongTensor).cuda(non_blocking=True)

            output = self.models[client](imgs)

            if len(labs.shape) == len(output.shape):
                labs = labs[:, 0]


            loss = self.criterion[client](output, labs)

            loss.backward()
            self.optimizers[client].step()
