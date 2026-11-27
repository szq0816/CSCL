import numpy as np
import torch.optim
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

import utils.loss
from utils.datasets import next_batch
from utils.evaluation import evaluation, get_cluster_sols
from models.baseModels import *
from utils.loss import *
# from torch.nn.functional import normalize
from utils.util import target_l2
from utils.visualization import TSNE_show2D, TSNE_show3D, loss_plot
# from mamba_ssm import Mamba
import scipy.io as scio
import os





class CSCL(nn.Module):
    def __init__(self, config):
        super(CSCL, self).__init__()
        self._config = config
        self._input_dim1 = config['Autoencoder']['gcnEncoder1'][0]
        self._input_dim2 = config['Autoencoder']['gcnEncoder2'][0]
        self._latent_dim = config['Autoencoder']['gcnEncoder1'][-1]
        self._n_clusters = config['n_clusters']

        self.gcnEncoder1 = GraphEncoder(config['Autoencoder']['gcnEncoder1'], 'relu', True)
        self.gcnEncoder2 = GraphEncoder(config['Autoencoder']['gcnEncoder2'], 'relu', True)

        self.instance_projector1 = InstanceProject(self._latent_dim)
        self.instance_projector2 = InstanceProject(self._latent_dim)

        self.cluster = ClusterProject(self._latent_dim, self._n_clusters)
        self.adaptive_attention_fusion = Attention(2 * self._latent_dim, self._latent_dim)
        self.mamba_enhancement= MambaEncoder(2*self._latent_dim)





    def forward(self, x1, x2, adj1, adj2):
        h1 = self.gcnEncoder1(x1, adj1)
        h2 = self.gcnEncoder2(x2, adj2)
        z1 = normalize(self.instance_projector1(h1), dim=1)
        z2 = normalize(self.instance_projector2(h2), dim=1)
        H1, H2 = self.mamba_enhancement( h1 , h2)

        y1, p1 = self.cluster(H1)
        y2, p2 = self.cluster(H2)
        return H1, H2, z1, z2, y1, y2, p1, p2,h1,h2

    def eval_acc(self, z, Y_list, accumulated_metrics, logger):
        """cal acc"""
        z = z.cpu().numpy()
        y_pred, _ = get_cluster_sols(z, ClusterClass=KMeans, n_clusters=self._n_clusters, init_args={'n_init': 10})
        scores = evaluation(y_pred=y_pred, y_true=Y_list[0], accumulated_metrics=accumulated_metrics)
        logger.info("\033[2;29m" + 'trainingset_view1 ' + str(scores) + "\033[0m")

    def run_train(self, x_train, Y_list, adj, optimizer, logger, accumulated_metrics,lambda1,lambda2,missing_rate,data_num,device):
        LOSS = []
        lamb1 = lambda1
        lamb2 = lambda2
        lamb3 = 1
        attention = True
        if lamb2 == 0:
            lamb3 = 0
            attention = False
        epochs = self._config['training']['epoch']
        print_num = self._config['print_num']
        batch_size = self._config['training']['batch_size']
        batch_size = batch_size if x_train[0].shape[0] > batch_size else x_train[0].shape[0]

        criterion_instance = InstanceLoss(batch_size, 1.0, device).to(device)
        criterion_cluster = ClusterLoss(self._n_clusters, 0.5, device).to(device)



        # train the model
        for k in range(epochs):
            H1, H2, z1, z2, y1, y2, p1, p2,h1,h2 = self(x_train[0], x_train[1], adj[0], adj[1])
            if attention:
                h = self.adaptive_attention_fusion(H1, H2)
            else:
                h = 0.5 * (H1 + H2)

            # cluster contrastive loss
            cluster_loss = criterion_cluster(y1, y2)
            loss = lamb1 * cluster_loss




            # instance contrastive loss
            z1, z2 = shuffle(z1, z2)
            for batch_z1, batch_z2, batch_No in next_batch(z1, z2, batch_size):
               instance_loss = criterion_instance(batch_z1, batch_z2)
               loss += lamb2 * instance_loss

            # high confidence loss
            y, _ = self.cluster(h)
            y_max = torch.maximum(y1, y2)
            y_max = torch.maximum(y_max, y)
            y_max = target_l2(y_max)
            y = torch.where(y < EPS, torch.tensor([EPS], device=y.device), y)
            kl_loss = F.kl_div(y.log(), y_max.detach(), reduction='batchmean')
            loss += lamb3 * kl_loss


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            LOSS.append(loss.item())
            if (k + 1) % print_num == 0:  # evaluation  k == 0 or(k + 1) % print_num == 0
                with torch.no_grad():
                    H1, H2, z1, z2, y1, y2, p1, p2,h1,h2  = self(x_train[0], x_train[1], adj[0], adj[1])
                    if attention:
                        h = self.adaptive_attention_fusion(H1, H2)
                    else:
                        h = 0.5 * (H1 + H2)
                    y, _ = self.cluster(h)
                    y = y.data.cpu().numpy().argmax(1)
                    scores = evaluation(y_pred=y, y_true=Y_list[0], accumulated_metrics=accumulated_metrics)
                    # print(str(scores))
                    acc = scores['accuracy']
                    nmi = scores['NMI']
                    ari = scores['ARI']
                    output = (
                        f"Epoch:{k + 1}/{epochs} "
                        f"| loss={loss.item():.4f} "
                        f"| ACC={acc:.4f} "
                        f"| NMI={nmi:.4f} "
                        f"| ARI={ari:.4f}"
                    )
                    logger.info(output)




        return accumulated_metrics['acc'][-1], accumulated_metrics['nmi'][-1], accumulated_metrics['ARI'][-1]
