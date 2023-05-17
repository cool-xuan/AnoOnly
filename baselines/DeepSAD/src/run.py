import torch
import logging
import random
import numpy as np
import pandas as pd
import os
from .utils.config import Config
from .utils.visualization.plot_images_grid import plot_images_grid
from .deepsad import deepsad
from .datasets.main import load_dataset
from myutils import Utils
from sklearn.metrics import roc_auc_score, average_precision_score

class DeepSAD():
    def __init__(self, seed, model_name='DeepSAD', batch_size=128, optim='adam', lr=1e-3, epoch=50, rho=0.05):
        self.utils = Utils()
        self.device = self.utils.get_device(gpu_specific=True)  # get device
        self.seed = seed

        self.net_name = 'dense'
        self.xp_path = None
        self.load_config = None
        self.load_model = None
        self.eta = 1.0 # eta in the loss function
        self.optimizer_name = optim
        self.lr = lr
        self.n_epochs = epoch
        self.lr_milestone = [0]
        self.batch_size = batch_size
        self.weight_decay = 1e-6
        self.pretrain = False # whether to use auto-encoder for pretraining
        self.ae_optimizer_name = 'adam'
        self.ae_lr = 0.001
        self.ae_n_epochs = 100
        self.ae_lr_milestone = [0]
        self.ae_batch_size = 128
        self.ae_weight_decay = 1e-6
        self.num_threads = 0
        self.n_jobs_dataloader = 0
        self.rho = rho
        
        self.roc_best = 0.0
        self.pr_best = 0.0

    def fit(self, X_train, y_train, ratio=None, X_test=None, y_test=None):
        """
        Deep SAD, a method for deep semi-supervised anomaly detection.

        :arg DATASET_NAME: Name of the dataset to load.
        :arg NET_NAME: Name of the neural network to use.
        :arg XP_PATH: Export path for logging the experiment.
        """

        # Set seed (using myutils)
        self.utils.set_seed(self.seed)

        # Set the number of threads used for parallelizing CPU operations
        if self.num_threads > 0:
            torch.set_num_threads(self.num_threads)
        logging.info('Computation device: %s' % self.device)
        logging.info('Number of threads: %d' % self.num_threads)
        logging.info('Number of dataloader workers: %d' % self.n_jobs_dataloader)

        # Load data
        data = {'X_train': X_train, 'y_train': y_train}
        dataset = load_dataset(data=data, train=True)
        if X_test is not None:
            dataset_test = load_dataset({'X_test': X_test, 'y_test': y_test}, train=False)
        else:
            dataset_test = None
        input_size = dataset.train_set.data.size(1) #input size

        # Initialize DeepSAD model and set neural network phi
        self.deepSAD = deepsad(self.eta)
        self.deepSAD.set_network(self.net_name, input_size)

        # If specified, load Deep SAD model (center c, network weights, and possibly autoencoder weights)
        if self.load_model:
            self.deepSAD.load_model(model_path=self.load_model, load_ae=True, map_location=self.device)
            logging.info('Loading model from %s.' % self.load_model)

        logging.info('Pretraining: %s' % self.pretrain)
        if self.pretrain:
            # Pretrain model on dataset (via autoencoder)
            self.deepSAD.pretrain(dataset,
                             input_size,
                             optimizer_name=self.ae_optimizer_name,
                             lr=self.ae_lr,
                             n_epochs=self.ae_n_epochs,
                             lr_milestones=self.ae_lr_milestone,
                             batch_size=self.ae_batch_size,
                             weight_decay=self.ae_weight_decay,
                             device=self.device,
                             n_jobs_dataloader=self.n_jobs_dataloader)

        # Train model on dataset
        self.deepSAD.train(dataset,
                          optimizer_name=self.optimizer_name,
                          lr=self.lr,
                          rho=self.rho,
                          n_epochs=self.n_epochs,
                          lr_milestones=self.lr_milestone,
                          batch_size=self.batch_size,
                          weight_decay=self.weight_decay,
                          device=self.device,
                          n_jobs_dataloader=self.n_jobs_dataloader,
                          dataset_test=dataset_test)

        self.roc_best = self.deepSAD.trainer.roc_best
        self.pr_best = self.deepSAD.trainer.pr_best
        # Save results, model, and configuration
        # deepSAD.save_results(export_json=xp_path + '/results.json')
        # deepSAD.save_model(export_model=xp_path + '/model.tar')
        # cfg.save_config(export_json=xp_path + '/config.json')

        # Plot most anomalous and most normal test samples
        # indices, labels, scores = zip(*deepSAD.results['test_scores'])
        # indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
        # idx_all_sorted = indices[np.argsort(scores)]  # from lowest to highest score
        # idx_normal_sorted = indices[labels == 0][np.argsort(scores[labels == 0])]  # from lowest to highest score

        return self

    @torch.no_grad()
    def eval(self, X_test, y_test):
        score = self.predict_score(X_test)
        roc = roc_auc_score(y_test, score)
        pr = average_precision_score(y_test, score)
        
        self.roc_best = roc if roc > self.roc_best else self.roc_best
        self.pr_best = pr if pr > self.pr_best else self.pr_best
    
    @torch.no_grad()
    def predict_score(self, X):
        # input randomly generated y label for consistence
        dataset = load_dataset(data={'X_test': X, 'y_test': np.random.choice([0, 1], X.shape[0])}, train=False)
        score = self.deepSAD.test(dataset, device=self.device, n_jobs_dataloader=self.n_jobs_dataloader)

        return score