from logging import root
from numpy.lib.function_base import rot90
import torch
import networkx as nx
import numpy as np
import random
import copy
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import os.path as osp
from torch_geometric.utils import to_networkx, is_undirected
from tqdm import tqdm

def get_neighbor(g, node):
    neighbor = list(g.neighbors(node))
    return neighbor 


def generate_node2vec_walks(args, g, num_nodes, alias_nodes, alias_edges):
    random_walks = []  
    nodes = list(range(num_nodes))
    for _ in tqdm(range(args.num_path)):   
        random.shuffle(nodes)
        for node in tqdm(nodes, leave=False):
            walk = node2vec_walk(args, g, node, alias_nodes, alias_edges) 
            random_walks.append(walk)

    node_walks = [[] for i in range(num_nodes)]   

    for w in random_walks:
        node_walks[w[0]].append(w)
    return node_walks, random_walks


def node2vec_walk(args, g, begin_node, alias_nodes, alias_edges):
    walk = [begin_node]

    while(len(walk) < args.walk_length):   
        cur = walk[-1]
        cur_neighbors = get_neighbor(g, cur) 
        cur_neighbors = sorted(cur_neighbors)  
        if len(cur_neighbors):
            if len(walk) == 1: 

                abc = alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])  
                walk.append(cur_neighbors[abc])
            else: 
                prev = walk[-2] 
                nextnode = cur_neighbors[alias_draw(alias_edges[(prev, cur)][0], 
                    alias_edges[(prev, cur)][1])]  
                walk.append(nextnode)
        else:
            break

    return walk


def preprocess_transition_prob(args, g):
    degree_seq_dict = dict(g.degree)   
    degree_seq = [degree_seq_dict[i] for i in range(args.num_nodes)]  
    alias_nodes = {}

    for node in g.nodes():
        normalized_probs = [1/degree_seq[node] for i in range(degree_seq[node])]   
        alias_nodes[node] = alias_setup(normalized_probs)  
    
    alias_edges = {}
    
    for edge in g.edges():   
        alias_edges[edge] = get_alias_edge(args, g, edge[0], edge[1]) 
        alias_edges[(edge[1], edge[0])] = get_alias_edge(args, g, edge[1], edge[0]) 
    
    alias_nodes = alias_nodes
    alias_edges = alias_edges
    
    return alias_nodes, alias_edges

def alias_setup(probs):

    K = len(probs)
    q = np.zeros(K) 
    J = np.zeros(K).astype(int)  
    smaller = []
    larger = []

    for kk, prob in enumerate(probs):   
        q[kk] = K * prob          # [1,1,1]
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)   # [0,1,2]  

    while len(smaller) >0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()
        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q  # J = [0,0,0] q = [0,0,0]

def alias_draw(J, q):
    K = len(J) 
    kk = int(np.floor(np.random.rand()*K)) 
    if np.random.rand()<q[kk]:  
        return kk
    else:
        return J[kk]

def get_alias_edge(args, g, src, dst):

    unnormalized_probs = []  
    for dst_nbr in sorted(g.neighbors(dst)):   
        if dst_nbr == src:  
            unnormalized_probs.append(1/args.p)  
        elif g.has_edge(dst_nbr, src):    
            # one hop neighbor
            unnormalized_probs.append(1)    
        else:
            unnormalized_probs.append(1/args.q)   
    normalize_const = np.sum(unnormalized_probs) 
    normalized_probs = [prob/normalize_const for prob in unnormalized_probs] 
    return alias_setup(normalized_probs)

if __name__ == '__main__':
    path = osp.join('datasets', 'PubMed')
    dataset = Planetoid(root=path, name='PubMed', transform = T.NormalizeFeatures())
    data = dataset[0]
    N = data.num_nodes
    num_features = dataset.num_features
    print("Number of Nodes: ", N)
    print("Number of Features: ", num_features)
    g = to_networkx(data)

    g = g.to_undirected()
    alias_nodes, alias_edges = preprocess_transition_prob(g,N)
    
    
    node_walks, random_walks = generate_node2vec_walks(g, N, alias_nodes, alias_edges)
    
    pass