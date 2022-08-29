
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, CitationFull, WebKB,WikiCS, Actor
import os.path as osp
from torch_geometric.utils import to_dense_adj

from utils import *
torch.cuda.set_device(0)
from torch_geometric.utils import degree, from_networkx

import networkx as nx
from torch_geometric.datasets import Coauthor

class ModelHandler(object):
    def __init__(self, args):
        self.args = args

        path = osp.join(args.dataset_path, args.dataset)
        if args.dataset in ['Cora', 'CiteSeer', 'PubMed', 'DBLP']:
            dataset = self.get_dataset(path, args.dataset)
            data = dataset[0]
            self.config_data(data, dataset)
        elif args.dataset in ['acm', 'BlogCatalog','flickr']:
            data = nx.read_gpickle(open(osp.join(path, args.dataset + '.nxg'), 'rb'))
            self.data = from_networkx(data)
            conf = configparser.ConfigParser()
            try:
                conf.read(args.config_file)
            except:
                print("loading config: %s failed" % (args.config))
            self.data.x = self.features = pkl.load(open(osp.join(path,  args.dataset + '.features'), 'rb'))
            self.data.y = pkl.load(open(osp.join(path,  args.dataset + '.labels'), 'rb'))
            self.N = num_nodes = conf.getint(args.dataset + "_Data_Setting", "num_nodes")
            self.nfeat = conf.getint(args.dataset + "_Data_Setting", "feature_dims")
            self.num_classes = conf.getint(args.dataset + "_Data_Setting", "num_classes")
            self.edge_index = self.data.edge_index
        elif args.dataset in ['CS']:
            dataset = Coauthor(path, args.dataset, transform = T.NormalizeFeatures())
            data = dataset[0]
            print(data)
            # try:
            #     conf.read(args.config_file)
            # except:
            #     print("loading config: %s failed" % (args.config))
            self.config_data(data, dataset)
        # self.logger = DummyLogger(config, dirname=config['out_dir'], pretrained=None)
        elif args.dataset in ['Texas','cornell','wisconsin']:
            dataset = WebKB(root = path, name = args.dataset, transform=T.NormalizeFeatures())
            data = dataset[0]
            self.config_data(data, dataset)
            
        elif args.dataset in ['Actor']:
             dataset = Actor(root = path, transform=T.NormalizeFeatures())
             data = dataset[0]
             self.config_data(data, dataset)
        elif args.dataset in ['WikiCS']:
            dataset = WikiCS(osp.join(path, args.dataset),transform = T.NormalizeFeatures())
            data = dataset[0]
            self.config_data(data, dataset)



    def get_dataset(self, path,name):
        assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP']
        name = 'dblp' if name == 'DBLP' else name   # Cora
        return (CitationFull if name == 'dblp' else Planetoid)(path, name, transform = T.NormalizeFeatures())
    
    def config_data(self, data, dataset):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.data = data
        self.adj = to_dense_adj(data.edge_index)
        self.features = data.x
        self.nfeat = dataset.num_features
        self.edge_index = data.edge_index
        self.N = data.num_nodes
        self.num_classes = dataset.num_classes   
        self.avg_degrees = int(torch.ceil(degree(data.edge_index[0]).mean()))
    

    def get_knn_graph_pkl(self, hyperparameters, k, etype = 'feature'):
        save_root = osp.join(SAVE_PATH, hyperparameters['dataset'])
        if etype == 'feature':
            g = pkl.load(open(osp.join(save_root, etype+'_{}nn.pkl'.format(k)), 'rb'))
            return g.edge_index
        else:
            g = pkl.load(open(osp.join(save_root, etype+'_{}_{}_{}nn.pkl'.format(hyperparameters['positional_embedding_size'], hyperparameters['hops'],k)),'rb'))
            return g.edge_index
        

