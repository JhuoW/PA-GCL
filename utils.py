import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils.num_nodes import maybe_num_nodes
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import sklearn.preprocessing as preprocessing
from torch_geometric.utils import degree, contains_isolated_nodes, to_undirected, is_undirected
import os
from torch_geometric.data import Data
import pickle as pkl
import os.path as osp
from rw_utils import preprocess_transition_prob, generate_node2vec_walks,node2vec_walk
import pathlib
from torch_geometric.utils import to_networkx
import lshknn
from tqdm import tqdm
import argparse
import configparser
from graph_kernels import wl_kernel, sp_kernel
from nystrom import Nystrom
import networkx as nx
import yaml
import utils_.constants as Constants
from torch_geometric.datasets import Planetoid, CitationFull, PPI, WikiCS, CoraFull, Coauthor, PPI
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T

DATASET_PATH = "datasets"
STRUCT_PATH = "structural_graph"

def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config

def diff(X, Y, Z):
    # cur_node_anchor_adj, pre_node_anchor_adj, cur_node_anchor_adj
    assert X.shape == Y.shape
    diff_ = torch.sum(torch.pow(X - Y, 2))
    norm_ = torch.sum(torch.pow(Z, 2))   
    diff_ = diff_ / torch.clamp(norm_, min=Constants.VERY_SMALL_NUMBER)
    return diff_

def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())

def sym_norm(A):
    """sym norm for tensor A"""
    A_ = A.clone()
    D = torch.sum(A_, dim = 1)
    return D[..., None].pow(-0.5) * A * D.pow(-0.5)

def symm_adj_sample(probs):
    """Sampling function for symmetric Bernoulli matrices"""
    e = bernoulli_hard_sample(probs) 
    return e + e.t()


def bernoulli_hard_sample(probs):
    """Sampling function for Bernoulli  伯努利采样函数  n*n 邻接矩阵 每条边随机从0~1间采样"""
    return torch.floor(torch.rand(size = probs.shape, device=probs.device) + probs)


def preprocess_adj(adj, device, add_selfloop = False):
    # adj_normalizer = fetch_normalization(normalization)
    adj_normalizer = noaug_normalized_adjacency
    r_adj = adj_normalizer(adj if not add_selfloop else adj + sp.eye(adj.shape[0]))
    r_adj = sparse_mx_to_torch_sparse_tensor(r_adj).float()
    r_adj = r_adj.to(device)
    return r_adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def noaug_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1)) # 每行和
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def k_hop_subgraph(node_idx, num_hops, edge_index,relabel_nodes=False,
                   num_nodes=None, flow='source_to_target'):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx] 

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])
    subset, inv = torch.cat(subsets).unique(return_inverse=True)   
    inv = inv[:node_idx.numel()] 

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]
    edge_index = edge_index[:, edge_mask]
    origin_edge_index = edge_index
    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_selection = torch.arange(num_nodes, device = row.device) 
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)  
        node_selection = node_selection[node_idx != -1]
        edge_index = node_idx[edge_index]
    origin_center_node = inv

    return edge_index, origin_center_node, subset, origin_edge_index



def to_scipy_sparse_matrix(edge_index, edge_attr=None, num_nodes=None):
    row, col = edge_index.cpu()

    if edge_attr is None:
        edge_attr = np.ones(row.size(0))
    else:
        edge_attr = edge_attr.view(-1).cpu()
        assert edge_attr.size(0) == row.size(0)

    N = edge_index[0].max() + 1
    out = sp.coo_matrix((edge_attr, (row, col)), (N, N))
    return out

def subgraph(subset, edge_index, node_features, num_nodes, relabel_nodes=True):
    device = edge_index.device

    if isinstance(subset, list) or isinstance(subset, tuple):
        subset = torch.tensor(subset, dtype=torch.long)

    if subset.dtype == torch.bool or subset.dtype == torch.uint8:
        n_mask = subset

        if relabel_nodes:
            n_idx = torch.zeros(n_mask.size(0), dtype=torch.long,
                                device=device)
            n_idx[subset] = torch.arange(subset.sum().item(), device=device)
    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        n_mask = torch.zeros(num_nodes, dtype=torch.bool)
        n_mask[subset] = 1

        if relabel_nodes:
            n_idx = torch.zeros(num_nodes, dtype=torch.long, device=device)
            n_idx[subset] = torch.arange(subset.size(0), device=device)  

    mask = n_mask[edge_index[0]] & n_mask[edge_index[1]]
    edge_index = edge_index[:, mask]

    origin_edge_index = edge_index
    if relabel_nodes:
        relabeled_edge_index = n_idx[edge_index] 
        relabeled_node_features  = node_features[subset]     
        relabeled_edge_index = relabeled_edge_index.T[torch.sort(relabeled_edge_index[0]).indices].T

    return relabeled_edge_index, origin_edge_index, relabeled_node_features
#####
def pre_generate_knn_graph_cora_citeseer(g):
    features = g.x
    features[features!=0] = 1 
    sims = cosine_similarity(features.cpu().detach())
    sims[(np.arange(len(sims)), np.arange(len(sims)))] = 0
    indices_argsort = np.argsort(sims, axis=1)
    return sims, indices_argsort

def pre_generate_knn_graph(features):
    features_sparse = sp.csr_matrix(features)
    features_sparse[features_sparse!=0] = 1
    sims = cosine_similarity(features_sparse)
    sims[(np.arange(len(sims)), np.arange(len(sims)))] = 0
    indices_argsort = np.argsort(sims, axis=1)
    return sims, indices_argsort

def pre_generate_knn_graph_s(features):
    features_sparse = sp.csr_matrix(features)
    # features_sparse[features_sparse!=0] = 1
    sims = cosine_similarity(features_sparse)
    sims[(np.arange(len(sims)), np.arange(len(sims)))] = 0
    indices_argsort = np.argsort(sims, axis=1)
    return sims, indices_argsort

def generate_knn_graph(sims, indices_argsort, k, selfloop = False):
    sims_ = np.copy(sims)
    sims_[np.arange(sims_.shape[0])[:,None], indices_argsort[:,:-k]] = 0
    A_feat = sp.csr_matrix(sims_)
    if selfloop: A_feat+=sp.eye(A_feat.shape[0])

    return sparse_mx_to_torch_sparse_tensor(A_feat)
#####

def get_sims(g):
    features = g.x
    features[features!=0] = 1 
    sims = cosine_similarity(features.cpu().detach())
    sims[(np.arange(len(sims)), np.arange(len(sims)))] = 0
    return sims

def generate_knn_graph2(k, sims, selfloop = False):
    sims_ = np.copy(sims)
    for i in range(len(sims_)):
        indices_argsort = np.argsort(sims_[i])  
        sims_[i, indices_argsort[: -k]] = 0 
      
    A_feat = sp.csr_matrix(sims_)
    if selfloop: A_feat+=sp.eye(A_feat.shape[0])

    return sparse_mx_to_torch_sparse_tensor(A_feat)

def eigen_decomposision(n, k, laplacian, hidden_size, retry):

    if k <= 0:
        return torch.zeros(n, hidden_size)
    laplacian = laplacian.astype("float64")  # graph laplacian
    ncv = min(n, max(2 * k + 1, 20))
    # follows https://stackoverflow.com/questions/52386942/scipy-sparse-linalg-eigsh-with-fixed-seed
    v0 = np.random.rand(n).astype("float64")  # 随机n维向量
    for i in range(retry):
        try:
            s, u = sp.linalg.eigsh(laplacian, k=k, which="LA", ncv=ncv, v0=v0)
        except sp.linalg.eigen.arpack.ArpackError:
            # print("arpack error, retry=", i)
            ncv = min(ncv * 2, n)
            if i + 1 == retry:
                sp.save_npz("arpack_error_sparse_matrix.npz", laplacian)
                u = torch.zeros(n, k)
        else:
            break
    x = preprocessing.normalize(u, norm="l2") 
    x = torch.from_numpy(x.astype("float32"))
    x = F.pad(x, (0, hidden_size - k), "constant", 0)  # 补0 到hidden_size维
    return x


def add_undirected_graph_positional_embedding(g, hidden_size, retry=10):

    g.edge_index = to_undirected(g.edge_index)
    n = int(g.edge_index[0].max() + 1)
    adj = to_scipy_sparse_matrix(g.edge_index).astype(float)
    norm = sp.diags(
        degree(g.edge_index[0]).cpu().detach().numpy() ** -0.5, dtype=float
    )
    laplacian = norm * adj * norm
    k = min(n - 2, hidden_size)
    x = eigen_decomposision(n, k, laplacian, hidden_size, retry)  # 得到每个节点的position embedding
    g.x = x.float()
    return g

def generate_position_graphs(args, data, stru_type = 'position'):

    if not osp.exists(f'/data1/_/CL/saved/{args.dataset}/'): 
        os.mkdir('/data1/_/CL/saved/{}/'.format(args.dataset))
    if not osp.exists(f'/data1/_/CL/saved/{args.dataset}/{args.dataset}_{stru_type}_position_graph_{args.hops}_{args.positional_embedding_size}.pkl'):
        position_graphs = []
        p_g = data.clone()
        p_x = torch.empty(size = (data.num_nodes, args.positional_embedding_size),dtype=float)
        for node in tqdm(range(data.num_nodes)):
            relabeled_edge_index, root_pos, node_sets, origin_edge_index  = k_hop_subgraph(node, args.hops, data.edge_index, relabel_nodes=True)
            # transform pyg graph
            g = Data(edge_index=relabeled_edge_index, root = root_pos[0])
            if node_sets.shape[0] > 1:
                g = add_undirected_graph_positional_embedding(g, args.positional_embedding_size)
            else:
                g.x = torch.zeros((1, args.positional_embedding_size))
            position_graphs.append(g)
            if stru_type == 'position':
                p_x[node] = g.x[root_pos]
            elif stru_type == 'position_mean':
                p_x[node] = g.x.mean(dim = 0)
            elif stru_type == 'position_add':
                p_x[node] = g.x.sum(dim = 0)
        p_g.x = p_x
        pkl.dump(p_g, open(f'/data1/_/CL/saved/{args.dataset}/{args.dataset}_{stru_type}_position_graph_{args.hops}_{args.positional_embedding_size}.pkl', 'wb'))
    else:
        print(f'loading /data1/_/CL/saved/{args.dataset}/{args.dataset}_{stru_type}_all_position_subgraphs_{args.hops}_{args.positional_embedding_size}.pkl')
        print(f'loading /data1/_/CL/saved/{args.dataset}/{args.dataset}_{stru_type}_position_graph_{args.hops}_{args.positional_embedding_size}.pkl')
        p_g = pkl.load(open(f'/data1/_/CL/saved/{args.dataset}/{args.dataset}_{stru_type}_position_graph_{args.hops}_{args.positional_embedding_size}.pkl', 'rb'))
    return p_g



# @np_cache(maxsize=256)
def generate_lsh_knn_graph(features, k, threshold = 0.3, m = 100):
    c = lshknn.Lshknn(data=features, k = k, threshold=threshold, m = m)
    knn_data = c()[0].data
    # data = np.ones(knn_data.size)
    row = np.repeat(np.arange(features.shape[1]), k)
    col = knn_data.flatten()

    indices = torch.from_numpy(
        np.vstack((row, col)).astype(np.int64))

    values = torch.from_numpy(np.ones(row.shape[0]))
    shape = torch.Size((features.shape[1], features.shape[1]))
    return torch.sparse.FloatTensor(indices, values, shape)

    # return torch.from_numpy(np.stack((row, col),axis = 0).astype(np.int64))


def generate_lsh_knn_graph2(features, k, threshold = 0.3, m = 100):
    c = lshknn.Lshknn(data=features, k = k, threshold=threshold, m = m)
    knn_data = c()[0].data
    # data = np.ones(knn_data.size)
    row = np.repeat(np.arange(features.shape[1]), k)
    col = knn_data.flatten()

    indices = torch.from_numpy(
        np.vstack((row, col)).astype(np.int64))
    return indices


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx



def load_graph(args, num_nodes):

    struct_edges = np.genfromtxt(osp.join(args.dataset_path, args.dataset, args.dataset+'.edge'), dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(num_nodes,  num_nodes), dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    nsadj = normalize(sadj+sp.eye(sadj.shape[0]))

    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)

    return nsadj



def load_data(args):
    feature_path = osp.join(args.dataset_path, args.dataset, args.dataset + '.feature')
    label_path = osp.join(args.dataset_path, args.dataset, args.dataset + '.label')

    f = np.loadtxt(feature_path, dtype = float) # (3327, 3703)
    l = np.loadtxt(label_path, dtype = int)   # (3327,)
    features = sp.csr_matrix(f, dtype=np.float32)       
    features = sparse_mx_to_torch_sparse_tensor(features)
    label = torch.LongTensor(np.array(l))

    return features, label



def generate_position_graphs_kernel(args, data, kernel):
    struc_path = osp.join(args.struct_graph, args.dataset)
    if not osp.exists(struc_path):
        os.mkdir(struc_path)
    filepath = osp.join(struc_path, f'{args.dataset}_{kernel.__name__}_graph_{args.positional_embedding_size}_{args.num_path}_{args.walk_length}.pkl')
    if not osp.exists(filepath):
        dataset_path = osp.join(args.dataset_path, args.dataset)
        if not osp.exists(osp.join(dataset_path, args.dataset + '.nxg')):
            g = to_networkx(data).to_undirected()
            g.remove_edges_from(nx.selfloop_edges(g))
            nx.write_gpickle(g, open(osp.join(dataset_path, args.dataset + '.nxg'), 'wb'))
        else:
            print("Load nx graph")
            g = nx.read_gpickle(open(osp.join(dataset_path, args.dataset + '.nxg'), 'rb'))
        alias_nodes, alias_edges = preprocess_transition_prob(args, g)
        local_graphs = []  # 每个节点的局部子图
        for node in tqdm(range(args.num_nodes)):
            rw_walks = []
            for _ in range(args.num_path):  # 每个节点num_path条路径
                walk = node2vec_walk(args, g, node, alias_nodes, alias_edges)
                rw_walks.append(walk)
            node_set = np.unique(np.array(rw_walks).flatten())
            local_graphs.append(g.subgraph(node_set))

        model = Nystrom(kernel, n_components=args.positional_embedding_size)
        model.fit(local_graphs)
        Q_t = model.transform(local_graphs)  # 每个节点子图互相之间的相似度  nd array
        
        x = torch.from_numpy(Q_t.astype(np.float))
        
        p_g = data.clone()
        p_g.x = x

        pkl.dump(p_g, open(filepath, 'wb'))
    else:
        print(f'loading {filepath}')
        p_g = pkl.load(open(filepath, 'rb'))
    return p_g



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="Cora", help='dataset name')
    parser.add_argument('--positional_embedding_size', type = int, default=64)
    parser.add_argument('--hops', type = int, default=2)
    parser.add_argument('--dataset_path', type=str, default=DATASET_PATH)
    parser.add_argument('--struct_graph', type=str, default=STRUCT_PATH)
    parser.add_argument('--config_file', type = str, default = 'dataset_config.ini')
    parser.add_argument('--sub_type', type = str, default = 'WL')  # sp
    parser.add_argument('--num_path', type = int, default= 30)  # 每个节点随机游走10次
    parser.add_argument('--walk_length', type = int, default= 10)   # 每个路径100个节点
    parser.add_argument('--p', type=float, default=0.01,help="return parameter for node2vec walk")  # 返回概率较高
    parser.add_argument("--q", type=float, default=8, help="out parameter for node2vec walk")
    args = parser.parse_args()

    conf = configparser.ConfigParser()
    try:
        conf.read(args.config_file)
    except:
        print("loading config: %s failed" % (args.config))

    if args.dataset not in ['PPI']:
        num_nodes = conf.getint(args.dataset + "_Data_Setting", "num_nodes")
        feature_dims = conf.getint(args.dataset + "_Data_Setting", "feature_dims")
        num_classes = conf.getint(args.dataset + "_Data_Setting", "num_classes")
    
    
    if args.dataset in ['acm', 'BlogCatalog' , 'flickr']:
        dataset_path = osp.join(DATASET_PATH, args.dataset)
        pathlib.Path(dataset_path).mkdir(parents=True, exist_ok=True)
        # Dataset from AM-GCN
        if not osp.exists(osp.join(dataset_path, args.dataset + '.nadj')):
            adj= load_graph(args, num_nodes)  # 返回原图adj 
            pkl.dump(adj, open(osp.join(dataset_path, args.dataset + '.nadj'), 'wb'))
        else:
            adj = pkl.load(open(osp.join(dataset_path, args.dataset + '.nadj'), 'rb'))

        if not osp.exists(osp.join(dataset_path, args.dataset + '.features')):
            features, labels = load_data(args)
            pkl.dump(features, open(osp.join(dataset_path, args.dataset + '.features'), 'wb'))
            pkl.dump(labels, open(osp.join(dataset_path, args.dataset + '.labels'), 'wb'))
        else:
            features = pkl.load(open(osp.join(dataset_path, args.dataset + '.features'), 'rb'))
            labels =  pkl.load(open(osp.join(dataset_path, args.dataset + '.labels'), 'rb'))
        G = Data(edge_index=adj._indices()) 
        args.num_nodes = num_nodes
        G.num_nodes = args.num_nodes
        avg_degrees = int(torch.ceil(degree(G.edge_index[0]).mean()))  
        print("Average Degrees: ", avg_degrees)
        if args.sub_type == 'positional':
            p_g = generate_position_graphs(args, G)
        elif args.sub_type == 'WL' or args.sub_type == 'SP':
            args.positional_embedding_size = 200
            p_g = generate_position_graphs_kernel(args, G, wl_kernel if args.sub_type == 'WL' else sp_kernel)
    
    elif args.dataset in ['Cora', 'CiteSeer', 'PubMed', 'DBLP']:
        dataset_path = osp.join(DATASET_PATH, args.dataset)
        pathlib.Path(dataset_path).mkdir(parents=True, exist_ok=True)
        def get_dataset(path,name):
            assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP']
            name = 'dblp' if name == 'DBLP' else name   # Cora
            return (CitationFull if name == 'dblp' else Planetoid)(path, name, transform = T.NormalizeFeatures())
        path = osp.join(DATASET_PATH, args.dataset)
        dataset = get_dataset(path, args.dataset)
        data = dataset[0]
        args.num_nodes = num_nodes
        data.num_nodes = args.num_nodes
        if args.sub_type == 'WL' or args.sub_type == 'SP':
            args.positional_embedding_size = 200
            p_g = generate_position_graphs_kernel(args, data,  wl_kernel if args.sub_type == 'WL' else sp_kernel)
    
    elif args.dataset in ['WikiCS']:
        args.mask = 0
        dataset_path = osp.join(DATASET_PATH, args.dataset)
        pathlib.Path(dataset_path).mkdir(parents=True, exist_ok=True)
        dataset = WikiCS(osp.join(dataset_path, args.dataset),transform = T.NormalizeFeatures())

        data = dataset[0]
        args.num_nodes = data.num_nodes
        print(data.num_edges)
        if args.sub_type == 'WL' or args.sub_type == 'SP':
            args.positional_embedding_size = 200
            p_g = generate_position_graphs_kernel(args, data,  wl_kernel if args.sub_type == 'WL' else sp_kernel)
    elif args.dataset in ["CS", "Physics"]:
        dataset_path = osp.join(DATASET_PATH, args.dataset)
        pathlib.Path(dataset_path).mkdir(parents=True, exist_ok=True)
        dataset = Coauthor(dataset_path, args.dataset, transform = T.NormalizeFeatures())
        data = dataset[0]
        args.num_nodes = num_nodes
        if args.sub_type == 'WL' or args.sub_type == 'SP':
            args.positional_embedding_size = 200
            p_g = generate_position_graphs_kernel(args, data,  wl_kernel if args.sub_type == 'WL' else sp_kernel)
    elif args.dataset in ['PPI']:
        dataset_path = osp.join(DATASET_PATH, args.dataset)
        train_dataset = PPI(dataset_path, split='train')  # 20
        val_dataset = PPI(dataset_path, split='test')    # 2
        test_dataset = PPI(dataset_path, split='test')   # 2
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

        print(train_dataset[0])
        