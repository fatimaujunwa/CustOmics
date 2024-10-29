# -*- coding: utf-8 -*-
"""
CustOMICS Module
This module is designed for multi-omics data integration and multi-task learning.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from lifelines import KaplanMeierFitter

from src.datasets.multi_omics_dataset import MultiOmicsDataset
from src.models.autoencoder import AutoEncoder
from src.encoders.encoder import Encoder
from src.decoders.decoder import Decoder
from src.encoders.probabilistic_encoder import ProbabilisticEncoder
from src.decoders.probabilistic_decoder import ProbabilisticDecoder
from src.tasks.classification import MultiClassifier
from src.tasks.survival import SurvivalNet
from src.models.vae import VAE
from src.loss.classification_loss import classification_loss
from src.loss.survival_loss import CoxLoss
from src.tools.utils import get_common_samples


class CustOMICS(nn.Module):
    def __init__(self, source_params, central_params, classif_params, surv_params=None, train_params=None, device='cpu'):
        super(CustOMICS, self).__init__()
        self.n_source = len(source_params)
        self.device = device
        self.phase = 1
        self.switch_epoch = train_params.get('switch', 5) if train_params else 5
        self.lr = train_params.get('lr', 1e-3) if train_params else 1e-3
        self.beta = central_params['beta']
        
        # Set up encoders and decoders for each source
        self.lt_encoders = [Encoder(input_dim=source_params[src]['input_dim'],
                                    hidden_dim=source_params[src]['hidden_dim'],
                                    latent_dim=source_params[src]['latent_dim'],
                                    norm_layer=source_params[src]['norm'],
                                    dropout=source_params[src]['dropout'])
                            for src in source_params]
        self.lt_decoders = [Decoder(latent_dim=source_params[src]['latent_dim'],
                                    hidden_dim=source_params[src]['hidden_dim'],
                                    output_dim=source_params[src]['input_dim'],
                                    norm_layer=source_params[src]['norm'],
                                    dropout=source_params[src]['dropout'])
                            for src in source_params]

        # Set up central encoder and decoder
        self.rep_dim = sum(src['latent_dim'] for src in source_params.values())
        self.central_encoder = ProbabilisticEncoder(input_dim=self.rep_dim,
                                                    hidden_dim=central_params['hidden_dim'],
                                                    latent_dim=central_params['latent_dim'],
                                                    norm_layer=central_params['norm'],
                                                    dropout=central_params['dropout'])
        self.central_decoder = ProbabilisticDecoder(latent_dim=central_params['latent_dim'],
                                                    hidden_dim=central_params['hidden_dim'],
                                                    output_dim=self.rep_dim,
                                                    norm_layer=central_params['norm'],
                                                    dropout=central_params['dropout'])

        # Classifier
        self.num_classes = classif_params['n_class']
        self.lambda_classif = classif_params['lambda']
        self.classifier = MultiClassifier(n_class=self.num_classes,
                                          latent_dim=central_params['latent_dim'],
                                          dropout=classif_params['dropout'],
                                          class_dim=classif_params['hidden_layers']).to(self.device)

        # Optional survival model
        self.lambda_survival = surv_params['lambda'] if surv_params and 'lambda' in surv_params else 0
        if surv_params:
            surv_dims = [central_params['latent_dim']] + surv_params['dims'] + [1]
            surv_param = {'drop': surv_params.get('dropout', 0.5), 'norm': surv_params.get('norm', False),
                          'dims': surv_dims, 'activation': surv_params.get('activation', 'SELU'),
                          'l2_reg': surv_params.get('l2_reg', 1e-2), 'device': self.device}
            self.survival_predictor = SurvivalNet(surv_param)
        else:
            self.survival_predictor = None
        
        # Model components
        self.autoencoders = []
        self._set_autoencoders()
        self._set_central_layer()
        self._relocate()
        self.optimizer = self._get_optimizer(self.lr)
        self.history = []

    def _get_optimizer(self, lr):
        params = [p for ae in self.autoencoders for p in ae.parameters()]
        params += list(self.central_layer.parameters()) + list(self.classifier.parameters())
        if self.survival_predictor:
            params += list(self.survival_predictor.parameters())
        return Adam(params, lr=lr)

    def _set_autoencoders(self):
        for i in range(self.n_source):
            self.autoencoders.append(AutoEncoder(self.lt_encoders[i], self.lt_decoders[i], self.device))

    def _set_central_layer(self):
        self.central_layer = VAE(self.central_encoder, self.central_decoder, self.device)

    def _relocate(self):
        for ae in self.autoencoders:
            ae.to(self.device)
        self.central_layer.to(self.device)

    def _compute_baseline(self, clinical_df, lt_samples, event, surv_time):
        kmf = KaplanMeierFitter()
        kmf.fit(clinical_df.loc[lt_samples, surv_time], clinical_df.loc[lt_samples, event])
        return kmf.survival_function_

    def per_source_forward(self, x):
        return [ae(x[i]) for i, ae in enumerate(self.autoencoders)]

    def forward(self, x):
        lt_rep = self.per_source_forward(x)
        lt_hat = [element[0] for element in lt_rep]
        lt_rep = [element[1] for element in lt_rep]
        central_concat = torch.cat(lt_rep, dim=1)
        mean, logvar = self.central_encoder(central_concat)
        return lt_hat, lt_rep, mean

    def _train_loop(self, x, labels, os_time, os_event):
        for i in range(len(x)):
            x[i] = x[i].to(self.device)
        loss = 0
        self.optimizer.zero_grad()
        if self.phase == 1:
            lt_rep, loss = self._compute_loss(x)
            for z in lt_rep:
                if self.survival_predictor:
                    hazard_pred = self.survival_predictor(z)
                    survival_loss = CoxLoss(survtime=os_time, censor=os_event, hazard_pred=hazard_pred, device=self.device)
                    loss += self.lambda_survival * survival_loss
                y_pred_proba = self.classifier(z)
                classification = classification_loss('CE', y_pred_proba, labels)
                loss += self.lambda_classif * classification
        elif self.phase == 2:
            z, loss = self._compute_loss(x)
            if self.survival_predictor:
                hazard_pred = self.survival_predictor(z)
                survival_loss = CoxLoss(survtime=os_time, censor=os_event, hazard_pred=hazard_pred, device=self.device)
                loss += self.lambda_survival * survival_loss
            y_pred_proba = self.classifier(z)
            classification = classification_loss('CE', y_pred_proba, labels)
            loss += self.lambda_classif * classification
        return loss

    def fit(self, omics_train, clinical_df, label, event=None, surv_time=None, omics_val=None, batch_size=32, n_epochs=30, verbose=False):
        encoded_clinical_df = clinical_df.copy()
        self.label_encoder = LabelEncoder().fit(encoded_clinical_df[label].values)
        encoded_clinical_df[label] = self.label_encoder.transform(encoded_clinical_df[label].values)
        self.one_hot_encoder = OneHotEncoder(sparse_output=False).fit(encoded_clinical_df[label].values.reshape(-1, 1))

        kwargs = {'num_workers': 2, 'pin_memory': True} if self.device.type == "cuda" else {}
        lt_samples_train = get_common_samples([df for df in omics_train.values()] + [clinical_df])

        if self.survival_predictor and event and surv_time:
            self.baseline = self._compute_baseline(clinical_df, lt_samples_train, event, surv_time)

        dataset_train = MultiOmicsDataset(omics_df=omics_train, clinical_df=encoded_clinical_df, lt_samples=lt_samples_train,
                                          label=label, event=event, surv_time=surv_time)
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, **kwargs)
        
        if omics_val:
            lt_samples_val = get_common_samples([df for df in omics_val.values()] + [clinical_df])
            dataset_val = MultiOmicsDataset(omics_df=omics_val, clinical_df=encoded_clinical_df, lt_samples=lt_samples_val,
                                            label=label, event=event, surv_time=surv_time)
            val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, **kwargs)

        self.history = []
        for epoch in range(n_epochs):
            overall_loss = 0
            self._switch_phase(epoch)
            for batch_idx, (x, labels, os_time, os_event) in enumerate(train_loader):
                self.train_all()
                loss_train = self._train_loop(x, labels, os_time, os_event)
                overall_loss += loss_train.item()
                loss_train.backward()
                self.optimizer.step()
            average_loss_train = overall_loss / ((batch_idx + 1) * batch_size)
            self.history.append(average_loss_train)

            if verbose:
                print(f"\tEpoch {epoch + 1} complete! \tAverage Loss Train: {average_loss_train:.4f}")

            if omics_val:
                overall_val_loss = 0
                for batch_idx, (x, labels, os_time, os_event) in enumerate(val_loader):
                    self.eval_all()
                    loss_val = self._train_loop(x, labels, os_time, os_event)
                    overall_val_loss += loss_val.item()
                average_loss_val = overall_val_loss / ((batch_idx + 1) * batch_size)
                self.history[-1] = (average_loss_train, average_loss_val)
                if verbose:
                    print(f"\tAverage Loss Val: {average_loss_val:.4f}")

    def plot_loss(self):
        n_epochs = len(self.history)
        plt.title('Loss evolution with epochs')
        plt.plot(range(n_epochs), [loss[0] if isinstance(loss, tuple) else loss for loss in self.history], label='Train Loss')
        if any(isinstance(loss, tuple) for loss in self.history):
            plt.plot(range(n_epochs), [loss[1] if isinstance(loss, tuple) else None for loss in self.history], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
