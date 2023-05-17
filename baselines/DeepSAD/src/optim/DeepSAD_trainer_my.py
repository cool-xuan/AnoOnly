from pickletools import optimize
from baseline.DeepSAD.src.base.base_trainer import BaseTrainer
from baseline.DeepSAD.src.base.base_dataset import BaseADDataset
from baseline.DeepSAD.src.base.base_net import BaseNet
from baseline.DeepSAD.src.datasets.odds import ODDSADDataset
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

import logging
import time
import torch
import torch.optim as optim
import numpy as np

from baseline.DeepSAD.src.asam import SAM, ASAM, ADSAM

class DeepSADTrainer(BaseTrainer):

    def __init__(self, c, eta: float, optimizer_name: str = 'adam', lr: float = 0.001, rho: float =0.05, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        # Deep SAD parameters
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.eta = eta

        # Optimization parameters
        self.eps = 1e-6

        # Results
        self.train_time = None
        self.test_aucroc = None; self.test_aucpr = None
        self.test_time = None
        self.test_scores = None
        self.n_epochs = n_epochs
        self.optimizer_name = optimizer_name
        self.rho = rho
        
        self.roc_best = 0.0
        self.pr_best = 0.0
        

    def train(self, dataset: ODDSADDataset, net: BaseNet, dataset_test: ODDSADDataset=None):
        logger = logging.getLogger()

        # Get train data loader
        train_loader = dataset.loaders(batch_size=256, num_workers=self.n_jobs_dataloader)

        # Set device for network
        net = net.to(self.device)

        # Set optimizer (Adam optimizer for now)
        # base_optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        base_optimizer = optim.SGD(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(base_optimizer, self.n_epochs)
        
        
        #^ SAM, ASAM optimizer
        sam_optimizer = ADSAM(base_optimizer, net, rho=self.rho)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            logger.info('Center c initialized.')

        train_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader, use_sampler=False)
        if dataset_test is None:
            test_loader = None
        else:
            test_loader = dataset_test.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()
        for epoch in range(self.n_epochs):

            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _, semi_targets, _ = data
                inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)

                # transfer the label "1" to "-1" for the inverse loss
                semi_targets[semi_targets==1] = -1
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)

                # Update network parameters via backpropagation: forward + backward + optimize
                #! simple optim
                if self.optimizer_name == 'adam':
                    dist_normal = dist[semi_targets==0]
                    loss_normal = torch.mean(dist_normal)
                    dist_abnormal = dist[semi_targets==-1]
                    loss_abnormal = self.eta * ((dist_abnormal + self.eps) ** -1)
                    loss_abnormal = torch.mean(loss_abnormal)
                    loss = loss_normal*0.5 + loss_abnormal*0.5
                    # loss = torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** -1))
                    # loss = torch.mean(loss)
                    base_optimizer.zero_grad() 
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    base_optimizer.step()
                
                #^ SAM only on abonrmal, consist to DevNet
                elif 'ours' in self.optimizer_name:
                    dist_normal = dist[semi_targets==0]
                    loss = torch.mean(dist_normal)
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    sam_optimizer.first_step()
                    
                    outputs = net(inputs)
                    dist = torch.sum((outputs - self.c) ** 2, dim=1)
                    dist_abnormal = dist[semi_targets==-1]
                    loss_abnormal = self.eta * ((dist_abnormal + self.eps) ** -1)
                    loss_abnormal = torch.mean(loss_abnormal)
                    loss_abnormal.backward()
                    sam_optimizer.second_step()
                    
                    outputs = net(inputs)
                    dist = torch.sum((outputs - self.c) ** 2, dim=1)
                    dist_abnormal = dist[semi_targets==-1]
                    loss_abnormal = self.eta * ((dist_abnormal + self.eps) ** -1)
                    loss_abnormal = torch.mean(loss_abnormal)
                    loss_abnormal.backward()
                    # torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    sam_optimizer.third_step()

                # #^ SAM on abnormal samples
                # dist_normal = dist[semi_targets==0]
                # loss = torch.mean(dist_normal)
                # loss.backward(retain_graph=True)
                
                # if sum(semi_targets==-1) > 0:
                #     dist_abnormal = dist[semi_targets==-1]
                #     loss_abnormal = self.eta * ((dist_abnormal + self.eps) ** semi_targets[semi_targets==-1])
                #     loss_abnormal = torch.mean(loss_abnormal)
                #     loss_abnormal.backward()
                #     sam_optimizer.ascent_step()
                    
                #     outputs = net(inputs)
                #     dist = torch.sum((outputs - self.c) ** 2, dim=1)
                #     dist_abnormal = dist[semi_targets==-1]
                #     loss_abnormal = self.eta * ((dist_abnormal + self.eps) ** semi_targets[semi_targets==-1])
                #     loss_abnormal = torch.mean(loss_abnormal)
                #     loss_abnormal.backward()
                #     sam_optimizer.descent_step()

                # # ^ SAM on abnormal samples
                # #* first step
                # #& only normal
                # # losses = dist[semi_targets==0]
                # #& only abnormal
                # # dist_abnormal = dist[semi_targets==-1]
                # # losses = self.eta * ((dist_abnormal + self.eps) ** semi_targets[semi_targets==-1].float())
                # #& all
                # losses = torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets.float()))
                # loss = torch.mean(losses)
                # loss.backward()
                # sam_optimizer.ascent_step()
                
                # #* second step
                # outputs = net(inputs)
                # dist = torch.sum((outputs - self.c) ** 2, dim=1)
                # #& only normal
                # # losses = dist[semi_targets==0]
                # #& only abnormal
                # dist_abnormal = dist[semi_targets==-1]
                # losses = self.eta * ((dist_abnormal + self.eps) ** semi_targets[semi_targets==-1].float())
                # #& all
                # # losses = torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets.float()))
                # loss = torch.mean(losses)
                # loss.backward()
                # sam_optimizer.descent_step()

                # epoch_loss += loss.item()
                n_batches += 1

            # log epoch statistics
            scheduler.step()
            epoch_train_time = time.time() - epoch_start_time
            logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')

            if test_loader is None:
                continue
            else:
                self.val(test_loader, net.eval())
                net.train()

        self.train_time = time.time() - start_time
        logger.info('Training Time: {:.3f}s'.format(self.train_time))
        logger.info('Finished training.')

        return net

    @torch.no_grad()
    def test(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Get test data loader
        test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set device for network
        net = net.to(self.device)

        # Testing
        logger.info('Starting testing...')
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, semi_targets, idx = data

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                semi_targets = semi_targets.to(self.device)
                idx = idx.to(self.device)

                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                # losses = torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets.float()))
                # loss = torch.mean(losses)
                scores = dist

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.detach().cpu().data.numpy().tolist()))

                # epoch_loss += loss.item()
                n_batches += 1

        self.test_time = time.time() - start_time
        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        # labels = np.array(labels)
        scores = np.array(scores)
        self.test_aucroc = roc_auc_score(labels, scores)
        self.test_aucpr = average_precision_score(labels, scores, pos_label = 1)
        
        self.roc_best = self.test_aucroc if self.test_aucroc > self.roc_best else self.roc_best
        self.pr_best = self.test_aucpr if self.test_aucpr > self.pr_best else self.pr_best
        
        # self.roc_best = self.test_aucroc
        # self.pr_best = self.test_aucpr

        # Log results
        # logger.info('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        # logger.info('Test AUCROC: {:.2f}%'.format(100. * self.test_aucroc))
        # logger.info('Test AUCPR: {:.2f}%'.format(100. * self.test_aucpr))
        logger.info('Test Time: {:.3f}s'.format(self.test_time))
        logger.info('Finished testing.')

        return scores
    
    @torch.no_grad()
    def val(self, test_loader, net: BaseNet):
        # Testing
        idx_label_score = []
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, semi_targets, idx = data

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                semi_targets = semi_targets.to(self.device)
                idx = idx.to(self.device)

                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                scores = dist

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.detach().cpu().data.numpy().tolist()))

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        # labels = np.array(labels)
        scores = np.array(scores)
        test_aucroc = roc_auc_score(labels, scores)
        test_aucpr = average_precision_score(labels, scores, pos_label = 1)
        
        self.roc_best = test_aucroc if test_aucroc > self.roc_best else self.roc_best
        self.pr_best  = test_aucpr  if test_aucpr  > self.pr_best  else self.pr_best

    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _, _, _ = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c
