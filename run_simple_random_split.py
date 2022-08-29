from typing import Callable
from model_handler import ModelHandler
import argparse
from utils import *
# torch.cuda.set_device(2)
import pathlib
import torch.nn as nn
from time import perf_counter
from test import label_classification, label_classification_public_split
import pickle as pkl
from torch_geometric.utils import remove_self_loops
import torch_geometric.nn as pyg_nn
import random

PG_PATH = "PG"

class Encoder(torch.nn.Module):
    def __init__(self, hyperparameters):
        super(Encoder, self).__init__()
        self.base_model = getattr(pyg_nn, hyperparameters['base_model'])  if hyperparameters['base_model'] != 'GINConv' else self.GIN()  # GIN
        num_hidden = hyperparameters['num_hidden']
        self.activation_name = hyperparameters['activation']
        assert hyperparameters['n_layers'] >= 2
        self.k = hyperparameters['n_layers']
        self.last_act = hyperparameters.get('last_act', True)
        
        self.conv = [self.base_model(hyperparameters['num_features'], num_hidden, add_self_loops = hyperparameters['selfloop'])]
        for _ in range(1, self.k - 1):
            self.conv.append(self.base_model(num_hidden, num_hidden, add_self_loops = hyperparameters['selfloop']))
        self.conv.append(self.base_model(num_hidden, num_hidden, add_self_loops = hyperparameters['selfloop']))
        self.conv = nn.ModuleList(self.conv)


        if self.activation_name != 'PReLU':
            self.activation = getattr(F, self.activation_name)
        else:
            self.activation = nn.PReLU(num_hidden)


    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        if self.last_act:
            for i in range(self.k):
                x = self.activation(self.conv[i](x, edge_index))
            return x
        else:
            for i in range(self.k-1):
                x = self.activation(self.conv[i](x, edge_index))
            x = self.conv[self.k-1](x, edge_index)
            return x



class Model(nn.Module):
    def __init__(self, hyperparameters, encoder):
        super(Model, self).__init__()
        self.hyperparameters = hyperparameters
        num_hidden = hyperparameters['num_hidden']
        num_proj_hidden = hyperparameters['num_proj_hidden']
        self.use_origin_graph = hyperparameters['use_origin_graph']
        self.encoder = encoder
        self.tau = hyperparameters['tau']
        self.fc1 = nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = nn.Linear(num_proj_hidden, num_hidden)


    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1, dim = 0)
        z2 = F.normalize(z2, dim = 0)
        return torch.mm(z1.t(), z2)


    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))  
        return -torch.log(
            between_sim.diag()  
            / between_sim.sum(1))



    def loss(self,z, z1, mean = True, batch_size = 0, iter_ = False):
        if not iter_:
            h1 = self.projection(z1)  
            h = self.projection(z)
            if batch_size == 0:


                if self.use_origin_graph:
                    l3 = self.semi_loss(h, h1)
                    l4 = self.semi_loss(h1, h)

            else:

                pass


            ret = (l3 + l4) * 0.5 

            ret = ret.mean() if mean else ret.sum()

            return ret
        else:
            pass

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)
    
    def forward(self,x, edge_index):
        return self.encoder(x, edge_index)


def get_optimizer(hyperparameters, model):
    def adam(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def momentum(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    optimizer_name = hyperparameters['optimizer']
    lr = hyperparameters['learning_rate']
    weight_decay = hyperparameters['weight_decay']
    optimizer: Callable[[nn.Module, float, float], torch.optim.Optimizer] = locals()[optimizer_name]
    return optimizer(model, lr, weight_decay)


def test(data, model,hyperparameters):
    model.eval()
    x = data.x.cuda()
    edge_index = data.edge_index.cuda()
    z = model(x, edge_index)

    total_acc = []
    total_mar = []
    if hyperparameters['dataset'] != 'WikiCS':
        for seed in range(1):
            result = label_classification(z, data.y, seed, ratio=0.1)
            total_acc.append(result['F1Mi'])
            total_mar.append(result['F1Ma'])
    else:
        for mask in range(10):
            train_mask = data.train_mask[:, mask].detach().cpu().numpy()
            val_mask = data.val_mask[:, mask].detach().cpu().numpy()
            test_mask = data.test_mask.detach().cpu().numpy()
            result = label_classification(z, data.y, train_mask,val_mask,test_mask)
            total_acc.append(result['F1Mi'])
            total_mar.append(result['F1Ma'])
    acc_mean = np.mean(total_acc)
    acc_std = np.std(total_acc)

    mar_mean = np.mean(total_mar)
    mar_std = np.std(total_mar)

    print(f'Acc = {acc_mean:.4f}', end='')
    print(f'Macro = {mar_mean:.4f}', end='')
    return result

def run_epoch(hyperparameters, A_knn, S_knn, edge_index, features, optimizer , model):
    model.train()
    optimizer.zero_grad()
    """
    all opteration in a whole epoch
    """
    # mode = "train" if training else ("test" if self.is_test else "val")
    if hyperparameters['dataset'] in ['Cora', 'CiteSeer']:
        edge_index_1 = remove_self_loops(A_knn._indices()).cuda()
        edge_index_2 = remove_self_loops(S_knn._indices()).cuda()
    elif hyperparameters['dataset'] in ['PubMed', 'DBLP', 'acm', 'BlogCatalog','flickr']:
        if hyperparameters['edge_index_inverse']:
            edge_index_1 = A_knn[[1,0]].cuda()
            edge_index_2 = S_knn[[1,0]].cuda()
        else:
            edge_index_1 = A_knn.cuda()
            edge_index_2 = S_knn.cuda()
    edge_index = edge_index.cuda()
    features = features.cuda()
    z = model(features, edge_index)
    z1 = model(features, edge_index_1)
    z2 = model(features, edge_index_2)

    loss = model.loss(z, z1, z2, batch_size=0)
    loss.backward()
    optimizer.step()
    # self._update_metrics(loss.item(), {'nloss': -loss.item(), self.model.metric_name: score}, 1, training=training)
    return loss.item()

def run_epoch_simple(hyperparameters, knn, edge_index, features, optimizer , model):
    model.train()
    optimizer.zero_grad()

    if hyperparameters['dataset'] in ['Cora', 'CiteSeer']:
        edge_index_1 = remove_self_loops(knn._indices())[0].cuda()
    elif hyperparameters['dataset'] in ['PubMed', 'DBLP', 'acm', 'BlogCatalog','flickr','CS', 'WikiCS']:

        edge_index_1 = remove_self_loops(knn)[0].cuda()
    edge_index = edge_index.cuda()
    features = features.cuda()
    z = model(features, edge_index)
    z1 = model(features, edge_index_1)
    # z2 = model(features, edge_index_2)

    loss = model.loss(z, z1, batch_size=0)
    loss.backward()
    optimizer.step()
    return loss.item()

def clip_grad(hyperparameters, model):
    # Clip gradients
    if hyperparameters['grad_clipping']:
        parameters = [p for p in model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(parameters, hyperparameters['grad_clipping'])


def get_knn_graph_pkl(hyperparameters, k, etype = 'feature'):
    save_root = osp.join(PG_PATH, hyperparameters['dataset'])
    if etype == 'feature':
        g = pkl.load(open(osp.join(save_root, etype+'_{}nn.pkl'.format(k)), 'rb'))
        return g.edge_index
    else:
        if etype == 'position':
            g = pkl.load(open(osp.join(save_root, etype+'_{}_{}_{}nn.pkl'.format(hyperparameters['positional_embedding_size'], hyperparameters['hops'],k)),'rb'))
        else:
            g = pkl.load(open(osp.join(save_root, etype+'_{}_{}_{}_{}nn.pkl'.format(hyperparameters['positional_embedding_size'], 30, 10,k)),'rb'))
        return g.edge_index

def main(args):
    config_path = osp.join(args.hyper_file, args.dataset + '.yml')
    config = get_config(config_path)
    hyperparameters = config


    model_helper = ModelHandler(args)

    data = model_helper.data
    features = model_helper.features
    hyperparameters['num_features'] = nfeat = model_helper.nfeat
    args.nfeat = nfeat
    edge_index = model_helper.edge_index
    hyperparameters['num_nodes'] = N = model_helper.N
    hyperparameters['num_class'] = num_class = model_helper.num_classes
    hyperparameters['knn_type'] = args.knn_type

    encoder = Encoder(hyperparameters)
    encoder.cuda()

    model = Model(hyperparameters, encoder)
    model.cuda()

    optimizer = get_optimizer(hyperparameters, model)
    epochs =  hyperparameters['epochs']

    if hyperparameters['dataset'] in ['Cora', 'CiteSeer']:
        features_ = data.x.cpu().numpy()
        if hyperparameters['stru_emb'] == 'position':
            # p_g = generate_position_graphs_helper(hyperparameters, data)
            # s_features = p_g.x.cpu().numpy()
            # TODO
            pass
        elif hyperparameters['stru_emb'] == 'wl' or 'sp':
            struc_file = osp.join(args.struct_graph, args.dataset, '{}_wl_kernel_graph_200_30_10.pkl'.format(hyperparameters['dataset']))
            p_g = pkl.load(open(struc_file, 'rb'))
            s_features = p_g.x.cpu().numpy()
    
        if hyperparameters['input_graph'] == 'knn':
            if hyperparameters['knn_type'] == 'cosine':
                sims_f, indices_argsort_f = pre_generate_knn_graph(features_)
                sims_s, indices_argsort_s = pre_generate_knn_graph_s(s_features)
    elif hyperparameters['dataset'] in ['PubMed', 'DBLP']:
        pass
    
    elif hyperparameters['dataset'] in ['acm', 'BlogCatalog','flickr']:
        pass
    
    start = perf_counter()
    prev = start

    k_f_cache = dict()
    k_s_cache = dict()
    for epoch in range(epochs):
        model.train()
        if hyperparameters['input_graph'] == 'knn':
            
            k_f = np.random.randint(hyperparameters['min_k_f'], hyperparameters['max_k_f'])
            if hyperparameters['same_k']:             
                k_s = k_f
            else:
                k_s = np.random.randint(hyperparameters['min_k_s'], hyperparameters['max_k_s'])
            # print('[ Using KNN-graph as input graph: k_f={},k_s={} ]'.format(k_f, k_s))

            if hyperparameters['dataset'] in ['Cora', 'CiteSeer']:
                sims_f_ = np.copy(sims_f)
                sims_s_ = np.copy(sims_s)
                if k_f not in k_f_cache:
                    A_knn = generate_knn_graph(sims_f_, indices_argsort_f, k_f, selfloop = hyperparameters['selfloop'])
                    k_f_cache[k_f] = A_knn
                else:
                    A_knn = k_f_cache[k_f]
                if k_s not in k_s_cache:
                    S_knn = generate_knn_graph(sims_s_, indices_argsort_s, k_s, selfloop = hyperparameters['selfloop'])
                    k_s_cache[k_s] = S_knn
                else:
                    S_knn = k_s_cache[k_s]


            elif hyperparameters['dataset'] in ['PubMed', 'DBLP', 'acm', 'BlogCatalog','flickr','CS', 'WikiCS']:
                if hyperparameters['knn_type'] == 'cosine':
                    if k_f not in k_f_cache:
                        A_knn = get_knn_graph_pkl(hyperparameters, k_f, etype='feature')
                        k_f_cache[k_f] = A_knn
                    else:
                        A_knn = k_f_cache[k_f]
                    if k_s not in k_s_cache:
                        S_knn = get_knn_graph_pkl(hyperparameters, k_s, etype='wl')
                        k_s_cache[k_s] = S_knn
                    else:
                        S_knn = k_s_cache[k_s]

        if epoch % 2==0:
            loss = run_epoch_simple(hyperparameters, A_knn, data.edge_index, data.x, optimizer, model)
        else:
            loss = run_epoch_simple(hyperparameters, S_knn, data.edge_index, data.x, optimizer, model)
        now = perf_counter()

        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
                f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now
    
    result = test(data, model, hyperparameters)

    return result

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default="Cora", help='dataset name')
    parser.add_argument('--knn_type', type = str, default='cosine')
    parser.add_argument('--direction', type=str, default = "maximize")
    parser.add_argument('--dataset_path', type=str, default=DATASET_PATH)
    parser.add_argument('--struct_graph', type=str, default=STRUCT_PATH)
    parser.add_argument('--config_file', type=str, default= 'dataset_config.ini')
    parser.add_argument('--gpu_id', type = int, default=1)
    parser.add_argument('--hyper_file', type=str, default= 'config')
    args = parser.parse_args()
    total_acc = []
    total_mar = []
    torch.manual_seed(0)
    random.seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    for _ in range(10):
        result = main(args)
        total_acc.append(result['F1Mi'])
        total_mar.append(result['F1Ma'])
    acc_mean = np.mean(total_acc)
    acc_std = np.std(total_acc)

    mar_mean = np.mean(total_mar)
    mar_std = np.std(total_mar)

    print("Result of each trial: \n")
    print(total_acc)
    print(f'Final Average Acc = {acc_mean:.4f}+-{acc_std:.4f}', end='')
