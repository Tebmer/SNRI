from torch.utils.data import Dataset
import timeit
import os
import logging
import lmdb
import numpy as np
import json
import pickle
import dgl
from utils.graph_utils import ssp_multigraph_to_dgl, incidence_matrix
from utils.data_utils import process_files, save_to_file, plot_rel_dist
from .graph_sampler import *
import pdb


def generate_subgraph_datasets(params, splits=['train', 'valid'], saved_relation2id=None, max_label_value=None, is_ent2rels=None):

    testing = 'test' in splits
    adj_list, triplets, entity2id, relation2id, id2entity, id2relation, h2r, m_h2r, t2r, m_t2r = process_files(params.file_paths, saved_relation2id, sort_data=params.sort_data)

    # plot_rel_dist(adj_list, os.path.join(params.main_dir, f'data/{params.dataset}/rel_dist.png'))

    data_path = os.path.join(params.main_dir, f'data/{params.dataset}/relation2id.json')
    if not os.path.isdir(data_path) and not testing:
        with open(data_path, 'w') as f:
            json.dump(relation2id, f)

    graphs = {}

    for split_name in splits:
        graphs[split_name] = {'triplets': triplets[split_name], 'max_size': params.max_links}

    # Sample train and valid/test links
    for split_name, split in graphs.items():
        logging.info(f"Sampling negative links for {split_name}")
        split['pos'], split['neg'] = sample_neg(adj_list, split['triplets'], params.num_neg_samples_per_link, max_size=split['max_size'], constrained_neg_prob=params.constrained_neg_prob)

    if testing:
        directory = os.path.join(params.main_dir, 'data/{}/'.format(params.dataset))
        save_to_file(directory, f'neg_{params.test_file}_{params.constrained_neg_prob}.txt', graphs['test']['neg'], id2entity, id2relation)

    links2subgraphs(adj_list, graphs, params, max_label_value)
    

def get_kge_embeddings(dataset, kge_model):

    path = './experiments/kge_baselines/{}_{}'.format(kge_model, dataset)
    node_features = np.load(os.path.join(path, 'entity_embedding.npy'))
    with open(os.path.join(path, 'id2entity.json')) as json_file:
        kge_id2entity = json.load(json_file)
        kge_entity2id = {v: int(k) for k, v in kge_id2entity.items()}

    return node_features, kge_entity2id


class SubgraphDataset(Dataset):
    """Extracted, labeled, subgraph dataset -- DGL Only"""

    def __init__(self, db_path, db_name_pos, db_name_neg, raw_data_paths, included_relations=None, add_traspose_rels=False, num_neg_samples_per_link=1, use_kge_embeddings=False, dataset='', kge_model='', file_name='', is_ret_nodes_num=False):

        self.main_env = lmdb.open(db_path, readonly=True, max_dbs=3, lock=False)
        self.db_pos = self.main_env.open_db(db_name_pos.encode())
        self.db_neg = self.main_env.open_db(db_name_neg.encode())
        self.node_features, self.kge_entity2id = get_kge_embeddings(dataset, kge_model) if use_kge_embeddings else (None, None)
        self.num_neg_samples_per_link = num_neg_samples_per_link
        self.file_name = file_name
        self.is_ret_nodes_num = is_ret_nodes_num

        ssp_graph, __, __, __, id2entity, id2relation, h2r, m_h2r, t2r, m_t2r = process_files(raw_data_paths, included_relations, add_traspose_rels)
        self.num_rels = len(ssp_graph)

        # Add transpose matrices to handle both directions of relations.
        if add_traspose_rels:
            ssp_graph_t = [adj.T for adj in ssp_graph]
            ssp_graph += ssp_graph_t

        # the effective number of relations after adding symmetric adjacency matrices and/or self connections
        self.aug_num_rels = len(ssp_graph)
        self.graph = ssp_multigraph_to_dgl(ssp_graph)
        self.ssp_graph = ssp_graph
        self.id2entity = id2entity
        self.id2relation = id2relation
        self.m_h2r = m_h2r
        self.m_t2r = m_t2r

        self.max_n_label = np.array([0, 0])
        with self.main_env.begin() as txn:
            self.max_n_label[0] = int.from_bytes(txn.get('max_n_label_sub'.encode()), byteorder='little')
            self.max_n_label[1] = int.from_bytes(txn.get('max_n_label_obj'.encode()), byteorder='little')

            self.avg_subgraph_size = struct.unpack('f', txn.get('avg_subgraph_size'.encode()))
            self.min_subgraph_size = struct.unpack('f', txn.get('min_subgraph_size'.encode()))
            self.max_subgraph_size = struct.unpack('f', txn.get('max_subgraph_size'.encode()))
            self.std_subgraph_size = struct.unpack('f', txn.get('std_subgraph_size'.encode()))

            self.avg_enc_ratio = struct.unpack('f', txn.get('avg_enc_ratio'.encode()))
            self.min_enc_ratio = struct.unpack('f', txn.get('min_enc_ratio'.encode()))
            self.max_enc_ratio = struct.unpack('f', txn.get('max_enc_ratio'.encode()))
            self.std_enc_ratio = struct.unpack('f', txn.get('std_enc_ratio'.encode()))

            self.avg_num_pruned_nodes = struct.unpack('f', txn.get('avg_num_pruned_nodes'.encode()))
            self.min_num_pruned_nodes = struct.unpack('f', txn.get('min_num_pruned_nodes'.encode()))
            self.max_num_pruned_nodes = struct.unpack('f', txn.get('max_num_pruned_nodes'.encode()))
            self.std_num_pruned_nodes = struct.unpack('f', txn.get('std_num_pruned_nodes'.encode()))

        logging.info(f"Max distance from sub : {self.max_n_label[0]}, Max distance from obj : {self.max_n_label[1]}")

        # logging.info('=====================')
        # logging.info(f"Subgraph size stats: \n Avg size {self.avg_subgraph_size}, \n Min size {self.min_subgraph_size}, \n Max size {self.max_subgraph_size}, \n Std {self.std_subgraph_size}")

        # logging.info('=====================')
        # logging.info(f"Enclosed nodes ratio stats: \n Avg size {self.avg_enc_ratio}, \n Min size {self.min_enc_ratio}, \n Max size {self.max_enc_ratio}, \n Std {self.std_enc_ratio}")

        # logging.info('=====================')
        # logging.info(f"# of pruned nodes stats: \n Avg size {self.avg_num_pruned_nodes}, \n Min size {self.min_num_pruned_nodes}, \n Max size {self.max_num_pruned_nodes}, \n Std {self.std_num_pruned_nodes}")

        with self.main_env.begin(db=self.db_pos) as txn:
            self.num_graphs_pos = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')
        with self.main_env.begin(db=self.db_neg) as txn:
            self.num_graphs_neg = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')

        self.__getitem__(0)

    def __getitem__(self, index):
        with self.main_env.begin(db=self.db_pos) as txn:
            str_id = '{:08}'.format(index).encode('ascii')
            nodes_pos, r_label_pos, g_label_pos, n_labels_pos = deserialize(txn.get(str_id)).values()
            subgraph_pos = self._prepare_subgraphs(nodes_pos, r_label_pos, n_labels_pos)
            
            # Get the neighbor relations of target head and tails
            # nei_rels_pos = [self.ent2rels[nodes_pos[0]], self.ent2rels[nodes_pos[1]]]
            nei_rels_pos = [[0, 1], [0, 1]]
            
        subgraphs_neg = []
        r_labels_neg = []
        g_labels_neg = []
        nei_rels_negs = []

        with self.main_env.begin(db=self.db_neg) as txn:
            for i in range(self.num_neg_samples_per_link):
                str_id = '{:08}'.format(index + i * (self.num_graphs_pos)).encode('ascii')
                nodes_neg, r_label_neg, g_label_neg, n_labels_neg = deserialize(txn.get(str_id)).values()
                subgraphs_neg.append(self._prepare_subgraphs(nodes_neg, r_label_neg, n_labels_neg))
                # Get the neighbor relations of target head and tails
                # nei_rels_neg = [self.ent2rels[nodes_neg[0]], self.ent2rels[nodes_neg[1]]]
                nei_rels_neg = [[0, 1], [0, 1]]
                nei_rels_negs.append(nei_rels_neg)
                r_labels_neg.append(r_label_neg)
                g_labels_neg.append(g_label_neg)
        
        # print("Nodes of subgraph: ", len(subgraph_pos.nodes()))
        return subgraph_pos, g_label_pos, r_label_pos, subgraphs_neg, g_labels_neg, r_labels_neg, 

    def __len__(self):
        return self.num_graphs_pos

    def _prepare_subgraphs(self, nodes, r_label, n_labels):
        subgraph: dgl.DGLGraph = self.graph.subgraph(nodes)
        subgraph.edata['type'] = self.graph.edata['type'][subgraph.edata[dgl.EID]]
        subgraph.edata['label'] = torch.tensor(r_label * np.ones(subgraph.edata['type'].shape), dtype=torch.long)
        
        # Check if the target relation is in the subgraph
        has_rel = subgraph.has_edges_between(0, 1)
        if has_rel:
            edges_btw_roots = subgraph.edge_ids(0, 1)
            # rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == r_label)   
            rel_link = subgraph.edata['type'][edges_btw_roots].item() == r_label # If the target relation is not in the subgraph, add a self-loop to the subgraph
        # if rel_link.squeeze().nelement() == 0:
        if not has_rel or not rel_link : # If there is no relation between the roots, or the target relation is not in the subgraph (these two cases may only occur for neg sample, because for neg sample, the target relation may not really exist, so we have to add this edge manually)
            # subgraph.add_edge(0, 1)
            subgraph.add_edges([0], [1])
            subgraph.edata['type'][-1] = torch.tensor(r_label).type(torch.LongTensor)
            subgraph.edata['label'][-1] = torch.tensor(r_label).type(torch.LongTensor)

        # map the id read by GraIL to the entity IDs as registered by the KGE embeddings
        kge_nodes = [self.kge_entity2id[self.id2entity[n]] for n in nodes] if self.kge_entity2id else None
        n_feats = self.node_features[kge_nodes] if self.node_features is not None else None
        subgraph = self._prepare_features_new(subgraph, n_labels, r_label, n_feats)

        # Add the original node id feature
        subgraph.ndata['parent_id'] = self.graph.subgraph(nodes).ndata[dgl.NID]
        # Add the neighbor relations 
        subgraph.ndata['out_nei_rels'] = torch.LongTensor(self.m_h2r[subgraph.ndata['parent_id']])
        subgraph.ndata['in_nei_rels'] = torch.LongTensor(self.m_t2r[subgraph.ndata['parent_id']])

        return subgraph

    def _prepare_features(self, subgraph, n_labels, n_feats=None):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1))
        label_feats[np.arange(n_nodes), n_labels] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)
        self.n_feat_dim = n_feats.shape[1]  # Find cleaner way to do this -- i.e. set the n_feat_dim
        return subgraph

    def _prepare_features_new(self, subgraph, n_labels, r_label, n_feats=None):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        # label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        # label_feats[np.arange(n_nodes), 0] = 1
        # label_feats[np.arange(n_nodes), self.max_n_label[0] + 1] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)

        head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
        tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = 1  # head
        n_ids[tail_id] = 2  # tail
        
        # 'id' is used to represent the target head and targe tail nodes
        subgraph.ndata['id'] = torch.FloatTensor(n_ids)
        
        # 'r_label' is used to represent the relation label of this subgraph
        subgraph.ndata['r_label'] = torch.LongTensor(np.ones(n_nodes) * r_label)
        self.n_feat_dim = n_feats.shape[1]  # Find cleaner way to do this -- i.e. set the n_feat_dim

        
        return subgraph
