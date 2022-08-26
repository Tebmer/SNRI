import os
import random
import argparse
import logging
import json
import time

import multiprocessing as mp
import scipy.sparse as ssp
from tqdm import tqdm
import networkx as nx
import torch
import numpy as np
import dgl


def process_files(files, saved_relation2id, add_traspose_rels):
    '''
    files: Dictionary map of file paths to read the triplets from.
    saved_relation2id: Saved relation2id (mostly passed from a trained model) which can be used to map relations to pre-defined indices and filter out the unknown ones.
    '''
    entity2id = {}
    relation2id = saved_relation2id

    triplets = {}

    ent = 0
    rel = 0

    for file_type, file_path in files.items():

        data = []
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split('\n')[:-1]]

        for triplet in file_data:
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = ent
                ent += 1
            if triplet[2] not in entity2id:
                entity2id[triplet[2]] = ent
                ent += 1

            # Save the triplets corresponding to only the known relations
            if triplet[1] in saved_relation2id:
                data.append([entity2id[triplet[0]], entity2id[triplet[2]], saved_relation2id[triplet[1]]])

        triplets[file_type] = np.array(data)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Construct the the neighbor relations of each entity
    num_rels = len(id2relation)
    num_ents = len(entity2id)
    h2r = {}
    h2r_len = {}
    t2r = {}
    t2r_len = {}
    for triplet in triplets['graph']:
        h, t, r = triplet
        if h not in h2r:
            h2r_len[h] = 1
            h2r[h] = [r]
        else:
            h2r_len[h] += 1
            h2r[h].append(r)
        
        if t not in t2r:
            t2r[t] = [r]
            t2r_len[t]  = 1
        else:
            t2r[t].append(r)
            t2r_len[t] += 1
        
    # ent2rels[-1] = [num_rels]
    # Construct the list of adjacency matrix each corresponding to eeach relation. Note that this is constructed only from the train data.

    # Construct the matrix of ent2rels
    # rels_len = triplets['train'].shape(0) // num_ents
    h_nei_rels_len = int(np.percentile(list(h2r_len.values()), 75))
    t_nei_rels_len = int(np.percentile(list(t2r_len.values()), 75))
    print("Average number of relations each node: ", "head: ", h_nei_rels_len, 'tail: ', t_nei_rels_len)
    

    # The index "num_rels" of relation is considered as "padding" relation.
    # Use padding relation to initialize matrix of ent2rels.
    m_h2r = np.ones([num_ents, h_nei_rels_len]) * num_rels
    for ent, rels in h2r.items():
        if len(rels) > h_nei_rels_len:
            rels = np.array(rels)[np.random.choice(np.arange(len(rels)), h_nei_rels_len)]
            m_h2r[ent] = rels
        else:
            rels = np.array(rels)
            m_h2r[ent][: rels.shape[0]] = rels      
    
    m_t2r = np.ones([num_ents, t_nei_rels_len]) * num_rels
    for ent, rels in t2r.items():
        if len(rels) > t_nei_rels_len:
            rels = np.array(rels)[np.random.choice(np.arange(len(rels)), t_nei_rels_len)]
            m_t2r[ent] = rels
        else:
            rels = np.array(rels)
            m_t2r[ent][: rels.shape[0]] = rels      


    # for ent, rels in ent2rels.items():
    #     if len(rels) > rels_len:
    #         rels = np.array(rels)[np.random.choice(np.arange(len(rels)), rels_len)]
    #         m_ent2rels[ent] = rels
    #     else:
    #         rels = np.array(rels)
    #         m_ent2rels[ent][: rels.shape[0]] = rels      

    adj_list = []
    for i in range(len(saved_relation2id)):
        idx = np.argwhere(triplets['graph'][:, 2] == i)
        adj_list.append(ssp.csc_matrix((np.ones(len(idx), dtype=np.uint8), (triplets['graph'][:, 0][idx].squeeze(1), triplets['graph'][:, 1][idx].squeeze(1))), shape=(len(entity2id), len(entity2id))))

    # Add transpose matrices to handle both directions of relations.
    adj_list_aug = adj_list
    if add_traspose_rels:
        adj_list_t = [adj.T for adj in adj_list]
        adj_list_aug = adj_list + adj_list_t

    dgl_adj_list = ssp_multigraph_to_dgl(adj_list_aug)

    return adj_list, dgl_adj_list, triplets, entity2id, relation2id, id2entity, id2relation, m_h2r, m_t2r


def intialize_worker(model, adj_list, dgl_adj_list, id2entity, params, node_features, kge_entity2id, m_h2r, m_t2r):
    global model_, adj_list_, dgl_adj_list_, id2entity_, params_, node_features_, kge_entity2id_, m_h2r_, m_t2r_
    model_, adj_list_, dgl_adj_list_, id2entity_, params_, node_features_, kge_entity2id_, m_h2r_, m_t2r_ = model, adj_list, dgl_adj_list, id2entity, params, node_features, kge_entity2id, m_h2r, m_t2r


def get_neg_samples_replacing_head_tail(test_links, adj_list, num_samples=50):

    n, r = adj_list[0].shape[0], len(adj_list)
    heads, tails, rels = test_links[:, 0], test_links[:, 1], test_links[:, 2]

    neg_triplets = []
    for i, (head, tail, rel) in enumerate(zip(heads, tails, rels)):
        neg_triplet = {'head': [[], 0], 'tail': [[], 0]}
        neg_triplet['head'][0].append([head, tail, rel])
        while len(neg_triplet['head'][0]) < num_samples:
            neg_head = head
            neg_tail = np.random.choice(n)

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['head'][0].append([neg_head, neg_tail, rel])

        neg_triplet['tail'][0].append([head, tail, rel])
        while len(neg_triplet['tail'][0]) < num_samples:
            neg_head = np.random.choice(n)
            neg_tail = tail
            # neg_head, neg_tail, rel = np.random.choice(n), np.random.choice(n), np.random.choice(r)

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['tail'][0].append([neg_head, neg_tail, rel])

        neg_triplet['head'][0] = np.array(neg_triplet['head'][0])
        neg_triplet['tail'][0] = np.array(neg_triplet['tail'][0])

        neg_triplets.append(neg_triplet)

    return neg_triplets


def get_neg_samples_replacing_head_tail_all(test_links, adj_list):

    n, r = adj_list[0].shape[0], len(adj_list)
    heads, tails, rels = test_links[:, 0], test_links[:, 1], test_links[:, 2]

    neg_triplets = []
    print('sampling negative triplets...')
    for i, (head, tail, rel) in tqdm(enumerate(zip(heads, tails, rels)), total=len(heads)):
        neg_triplet = {'head': [[], 0], 'tail': [[], 0]}
        neg_triplet['head'][0].append([head, tail, rel])
        for neg_tail in range(n):
            neg_head = head

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['head'][0].append([neg_head, neg_tail, rel])

        neg_triplet['tail'][0].append([head, tail, rel])
        for neg_head in range(n):
            neg_tail = tail

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['tail'][0].append([neg_head, neg_tail, rel])

        neg_triplet['head'][0] = np.array(neg_triplet['head'][0])
        neg_triplet['tail'][0] = np.array(neg_triplet['tail'][0])

        neg_triplets.append(neg_triplet)

    return neg_triplets


def get_neg_samples_replacing_head_tail_from_ruleN(ruleN_pred_path, entity2id, saved_relation2id):
    with open(ruleN_pred_path) as f:
        pred_data = [line.split() for line in f.read().split('\n')[:-1]]

    neg_triplets = []
    for i in range(len(pred_data) // 3):
        neg_triplet = {'head': [[], 10000], 'tail': [[], 10000]}
        if pred_data[3 * i][1] in saved_relation2id:
            head, rel, tail = entity2id[pred_data[3 * i][0]], saved_relation2id[pred_data[3 * i][1]], entity2id[pred_data[3 * i][2]]
            for j, new_head in enumerate(pred_data[3 * i + 1][1::2]):
                neg_triplet['head'][0].append([entity2id[new_head], tail, rel])
                if entity2id[new_head] == head:
                    neg_triplet['head'][1] = j
            for j, new_tail in enumerate(pred_data[3 * i + 2][1::2]):
                neg_triplet['tail'][0].append([head, entity2id[new_tail], rel])
                if entity2id[new_tail] == tail:
                    neg_triplet['tail'][1] = j

            neg_triplet['head'][0] = np.array(neg_triplet['head'][0])
            neg_triplet['tail'][0] = np.array(neg_triplet['tail'][0])

            neg_triplets.append(neg_triplet)

    return neg_triplets


def incidence_matrix(adj_list):
    '''
    adj_list: List of sparse adjacency matrices
    '''

    rows, cols, dats = [], [], []
    dim = adj_list[0].shape
    for adj in adj_list:
        adjcoo = adj.tocoo()
        rows += adjcoo.row.tolist()
        cols += adjcoo.col.tolist()
        dats += adjcoo.data.tolist()
    row = np.array(rows)
    col = np.array(cols)
    data = np.array(dats)
    return ssp.csc_matrix((data, (row, col)), shape=dim)


def _bfs_relational(adj, roots, max_nodes_per_hop=None):
    """
    BFS for graphs with multiple edge types. Returns list of level sets.
    Each entry in list corresponds to relation specified by adj_list.
    Modified from dgl.contrib.data.knowledge_graph to node accomodate sampling
    """
    visited = set()
    current_lvl = set(roots)

    next_lvl = set()

    while current_lvl:

        for v in current_lvl:
            visited.add(v)

        next_lvl = _get_neighbors(adj, current_lvl)
        next_lvl -= visited  # set difference

        if max_nodes_per_hop and max_nodes_per_hop < len(next_lvl):
            next_lvl = set(random.sample(next_lvl, max_nodes_per_hop))

        yield next_lvl

        current_lvl = set.union(next_lvl)


def _get_neighbors(adj, nodes):
    """Takes a set of nodes and a graph adjacency matrix and returns a set of neighbors.
    Directly copied from dgl.contrib.data.knowledge_graph"""
    sp_nodes = _sp_row_vec_from_idx_list(list(nodes), adj.shape[1])
    sp_neighbors = sp_nodes.dot(adj)
    neighbors = set(ssp.find(sp_neighbors)[1])  # convert to set of indices
    return neighbors


def _sp_row_vec_from_idx_list(idx_list, dim):
    """Create sparse vector of dimensionality dim from a list of indices."""
    shape = (1, dim)
    data = np.ones(len(idx_list))
    row_ind = np.zeros(len(idx_list))
    col_ind = list(idx_list)
    return ssp.csr_matrix((data, (row_ind, col_ind)), shape=shape)


def get_neighbor_nodes(roots, adj, h=1, max_nodes_per_hop=None):
    bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(h):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            pass
    return set().union(*lvls)


def subgraph_extraction_labeling(ind, rel, A_list, h=1, enclosing_sub_graph=False, max_nodes_per_hop=None, node_information=None, max_node_label_value=None):
    # extract the h-hop enclosing subgraphs around link 'ind'
    A_incidence = incidence_matrix(A_list)
    A_incidence += A_incidence.T

    # could pack these two into a function
    root1_nei = get_neighbor_nodes(set([ind[0]]), A_incidence, h, max_nodes_per_hop)
    root2_nei = get_neighbor_nodes(set([ind[1]]), A_incidence, h, max_nodes_per_hop)

    subgraph_nei_nodes_int = root1_nei.intersection(root2_nei)
    subgraph_nei_nodes_un = root1_nei.union(root2_nei)

    # Extract subgraph | Roots being in the front is essential for labelling and the model to work properly.
    if enclosing_sub_graph:
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_int)
    else:
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_un)

    subgraph = [adj[subgraph_nodes, :][:, subgraph_nodes] for adj in A_list]

    labels, enclosing_subgraph_nodes = node_label_new(incidence_matrix(subgraph), max_distance=h)

    pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes].tolist()
    pruned_labels = labels[enclosing_subgraph_nodes]

    if max_node_label_value is not None:
        pruned_labels = np.array([np.minimum(label, max_node_label_value).tolist() for label in pruned_labels])

    return pruned_subgraph_nodes, pruned_labels


def remove_nodes(A_incidence, nodes):
    idxs_wo_nodes = list(set(range(A_incidence.shape[1])) - set(nodes))
    return A_incidence[idxs_wo_nodes, :][:, idxs_wo_nodes]


def node_label_new(subgraph, max_distance=1):
    # an implementation of the proposed double-radius node labeling (DRNd   L)
    roots = [0, 1]
    sgs_single_root = [remove_nodes(subgraph, [root]) for root in roots]
    dist_to_roots = [np.clip(ssp.csgraph.dijkstra(sg, indices=[0], directed=False, unweighted=True, limit=1e6)[:, 1:], 0, 1e7) for r, sg in enumerate(sgs_single_root)]
    dist_to_roots = np.array(list(zip(dist_to_roots[0][0], dist_to_roots[1][0])), dtype=int)

    # dist_to_roots[np.abs(dist_to_roots) > 1e6] = 0
    # dist_to_roots = dist_to_roots + 1
    target_node_labels = np.array([[0, 1], [1, 0]])
    labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels

    enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]
    # print(len(enclosing_subgraph_nodes))
    return labels, enclosing_subgraph_nodes


def ssp_multigraph_to_dgl(graph, n_feats=None):
    """
    Converting ssp multigraph (i.e. list of adjs) to dgl multigraph.
    """

    g_nx = nx.MultiDiGraph()
    g_nx.add_nodes_from(list(range(graph[0].shape[0])))
    # Add edges
    for rel, adj in enumerate(graph):
        # Convert adjacency matrix to tuples for nx0
        nx_triplets = []
        for src, dst in list(zip(adj.tocoo().row, adj.tocoo().col)):
            nx_triplets.append((src, dst, {'type': rel}))
        g_nx.add_edges_from(nx_triplets)

    # make dgl graph
    g_dgl = dgl.DGLGraph(multigraph=True)
    g_dgl.from_networkx(g_nx, edge_attrs=['type'])
    # add node features
    if n_feats is not None:
        g_dgl.ndata['feat'] = torch.tensor(n_feats)

    return g_dgl


def prepare_features(subgraph, n_labels, max_n_label, n_feats=None):
    # One hot encode the node label feature and concat to n_featsure
    n_nodes = subgraph.number_of_nodes()
    label_feats = np.zeros((n_nodes, max_n_label[0] + 1 + max_n_label[1] + 1))
    label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
    label_feats[np.arange(n_nodes), max_n_label[0] + 1 + n_labels[:, 1]] = 1
    n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
    subgraph.ndata['feat'] = torch.FloatTensor(n_feats)

    head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
    tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
    n_ids = np.zeros(n_nodes)
    n_ids[head_id] = 1  # head
    n_ids[tail_id] = 2  # tail
    subgraph.ndata['id'] = torch.FloatTensor(n_ids)

    return subgraph


def get_subgraphs(all_links, adj_list, dgl_adj_list, max_node_label_value, id2entity, m_h2r, m_t2r, node_features=None, kge_entity2id=None):
    # dgl_adj_list = ssp_multigraph_to_dgl(adj_list)

    subgraphs = []
    r_labels = []
    nodes_num = []
    ht_nei_rels = []
    for link in all_links:
        head, tail, rel = link[0], link[1], link[2]
        nodes, node_labels = subgraph_extraction_labeling((head, tail), rel, adj_list, h=params_.hop, enclosing_sub_graph=params.enclosing_sub_graph, max_node_label_value=max_node_label_value)

        subgraph = dgl.DGLGraph(dgl_adj_list.subgraph(nodes))
        subgraph.edata['type'] = dgl_adj_list.edata['type'][dgl_adj_list.subgraph(nodes).parent_eid]
        subgraph.edata['label'] = torch.tensor(rel * np.ones(subgraph.edata['type'].shape), dtype=torch.long)

        edges_btw_roots = subgraph.edge_id(0, 1)
        rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == rel)

        if rel_link.squeeze().nelement() == 0:
            # subgraph.add_edge(0, 1, {'type': torch.tensor([rel]), 'label': torch.tensor([rel])})
            subgraph.add_edge(0, 1)
            subgraph.edata['type'][-1] = torch.tensor(rel).type(torch.LongTensor)
            subgraph.edata['label'][-1] = torch.tensor(rel).type(torch.LongTensor)

        nodes_num.append(subgraph.nodes().shape[0])
        
        kge_nodes = [kge_entity2id[id2entity[n]] for n in nodes] if kge_entity2id else None
        n_feats = node_features[kge_nodes] if node_features is not None else None
        subgraph = prepare_features(subgraph, node_labels, max_node_label_value, n_feats)
        
        subgraph.ndata['parent_id'] = dgl_adj_list.subgraph(nodes).parent_nid
        subgraph.ndata['out_nei_rels'] = torch.LongTensor(m_h2r[subgraph.ndata['parent_id']])
        subgraph.ndata['in_nei_rels'] = torch.LongTensor(m_t2r[subgraph.ndata['parent_id']])
        subgraph.ndata['r_label'] = torch.LongTensor(np.ones(subgraph.number_of_nodes()) * rel)
        ht_nei_rels.append((subgraph.ndata['out_nei_rels'][0].tolist(), subgraph.ndata['out_nei_rels'][1].tolist()))
        # print(ht_nei_rels)
        subgraphs.append(subgraph)
        r_labels.append(rel)

    batched_graph = dgl.batch(subgraphs)

    r_labels = torch.LongTensor(r_labels)

    return (batched_graph, r_labels), np.array(nodes_num), ht_nei_rels


def get_rank(neg_links):
    head_neg_links = neg_links['head'][0]
    head_target_id = neg_links['head'][1]

    if head_target_id != 10000:
        data, head_nodes_num, head_nei_rels = get_subgraphs(head_neg_links, adj_list_, dgl_adj_list_, model_.gnn.max_label_value, id2entity_, m_h2r_, m_t2r_, node_features_, kge_entity2id_)
        head_scores = model_(data).squeeze(1).detach().numpy()
        head_rank = np.argwhere(np.argsort(head_scores)[::-1] == head_target_id) + 1
    else:
        head_scores = np.array([])
        head_rank = 10000

    tail_neg_links = neg_links['tail'][0]
    tail_target_id = neg_links['tail'][1]

    if tail_target_id != 10000:
        data, tail_nodes_num, tail_nei_rels = get_subgraphs(tail_neg_links, adj_list_, dgl_adj_list_, model_.gnn.max_label_value, id2entity_, m_h2r_, m_t2r_, node_features_, kge_entity2id_)
        tail_scores = model_(data).squeeze(1).detach().numpy()
        tail_rank = np.argwhere(np.argsort(tail_scores)[::-1] == tail_target_id) + 1
    else:
        tail_scores = np.array([])
        tail_rank = 10000

    return head_scores, head_rank, tail_scores, tail_rank, head_nodes_num, tail_nodes_num, head_nei_rels, tail_nei_rels


def save_to_file(neg_triplets, id2entity, id2relation):

    with open(os.path.join('./data', params.dataset, 'ranking_head.txt'), "w") as f:
        for neg_triplet in neg_triplets:
            for s, o, r in neg_triplet['head'][0]:
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')

    with open(os.path.join('./data', params.dataset, 'ranking_tail.txt'), "w") as f:
        for neg_triplet in neg_triplets:
            for s, o, r in neg_triplet['tail'][0]:
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')


def save_score_to_file(neg_triplets, all_head_scores, all_tail_scores, id2entity, id2relation, all_h_nodes_num, all_t_nodes_num, all_h_nei_rels, all_t_nei_rels):

    with open(os.path.join('./data', params.dataset, 'grail_ranking_head_predictions.txt'), "w") as f:
        for i, neg_triplet in enumerate(neg_triplets):
            for [s, o, r], head_score, nodes_num, nei_rels in zip(neg_triplet['head'][0], all_head_scores[50 * i:50 * (i + 1)], all_h_nodes_num[50 * i:50 * (i + 1)], all_h_nei_rels[50 * i:50 * (i + 1)]):
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o], str(head_score), str(nodes_num)] + [' head nei rels: '] + [str(rel) for rel in nei_rels[0]] + [' tail nei rels: '] + [str(rel) for rel in nei_rels[1]]) + '\n')

    with open(os.path.join('./data', params.dataset, 'grail_ranking_tail_predictions.txt'), "w") as f:
        for i, neg_triplet in enumerate(neg_triplets):
            for [s, o, r], tail_score, nodes_num, nei_rels in zip(neg_triplet['tail'][0], all_tail_scores[50 * i:50 * (i + 1)], all_t_nodes_num[50 * i:50 * (i + 1)], all_t_nei_rels[50 * i:50 * (i + 1)]):
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o], str(tail_score), str(nodes_num)] + [' head nei rels: '] + [str(rel) for rel in nei_rels[0]] + [' tail nei rels: '] + [str(rel) for rel in nei_rels[1]]) + '\n')


def save_score_to_file_from_ruleN(neg_triplets, all_head_scores, all_tail_scores, id2entity, id2relation):

    with open(os.path.join('./data', params.dataset, 'grail_ruleN_ranking_head_predictions.txt'), "w") as f:
        for i, neg_triplet in enumerate(neg_triplets):
            for [s, o, r], head_score in zip(neg_triplet['head'][0], all_head_scores[50 * i:50 * (i + 1)]):
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o], str(head_score)]) + '\n')

    with open(os.path.join('./data', params.dataset, 'grail_ruleN_ranking_tail_predictions.txt'), "w") as f:
        for i, neg_triplet in enumerate(neg_triplets):
            for [s, o, r], tail_score in zip(neg_triplet['tail'][0], all_tail_scores[50 * i:50 * (i + 1)]):
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o], str(tail_score)]) + '\n')


def get_kge_embeddings(dataset, kge_model):

    path = './experiments/kge_baselines/{}_{}'.format(kge_model, dataset)
    node_features = np.load(os.path.join(path, 'entity_embedding.npy'))
    with open(os.path.join(path, 'id2entity.json')) as json_file:
        kge_id2entity = json.load(json_file)
        kge_entity2id = {v: int(k) for k, v in kge_id2entity.items()}

    return node_features, kge_entity2id


def main(params):
    
    s_t = time.time()
    model = torch.load(params.model_path, map_location='cpu')

    adj_list, dgl_adj_list, triplets, entity2id, relation2id, id2entity, id2relation, m_h2r, m_t2r = process_files(params.file_paths, model.relation2id, params.add_traspose_rels)

    node_features, kge_entity2id = get_kge_embeddings(params.dataset, params.kge_model) if params.use_kge_embeddings else (None, None)

    if params.mode == 'sample':
        neg_triplets = get_neg_samples_replacing_head_tail(triplets['links'], adj_list)
        save_to_file(neg_triplets, id2entity, id2relation)
    elif params.mode == 'all':
        neg_triplets = get_neg_samples_replacing_head_tail_all(triplets['links'], adj_list)
    elif params.mode == 'ruleN':
        neg_triplets = get_neg_samples_replacing_head_tail_from_ruleN(params.ruleN_pred_path, entity2id, relation2id)

    ranks = []
    all_head_scores = []
    all_tail_scores = []
    nodes_num = []
    all_h_nei_rels = []
    all_t_nei_rels = []
    all_h_nodes_num = []
    all_t_nodes_num = []
    # For one time test the feasibility of code
    # global model_, adj_list_, dgl_adj_list_, id2entity_, params_, node_features_, kge_entity2id_, m_h2r_, m_t2r_
    # model_, adj_list_, dgl_adj_list_, id2entity_, params_, node_features_, kge_entity2id_, m_h2r_, m_t2r_ = model, adj_list, dgl_adj_list, id2entity, params, node_features, kge_entity2id, m_h2r, m_t2r
    # print("result: ", get_rank(neg_triplets[0]))

    # p = mp.Pool(processes=None, initializer=intialize_worker, initargs=(model, adj_list, dgl_adj_list, id2entity, params, node_features, kge_entity2id))
    with mp.Pool(processes=None, initializer=intialize_worker, initargs=(model, adj_list, dgl_adj_list, id2entity, params, node_features, kge_entity2id, m_h2r, m_t2r)) as p:
        for head_scores, head_rank, tail_scores, tail_rank, h_nodes_num, t_nodes_num, h_nei_rels, t_nei_rels in tqdm(p.imap(get_rank, neg_triplets), total=len(neg_triplets)):
            ranks.append(head_rank)
            ranks.append(tail_rank)

            all_head_scores += head_scores.tolist()
            all_tail_scores += tail_scores.tolist()

            nodes_num.append(h_nodes_num.tolist()[0])
            nodes_num.append(t_nodes_num.tolist()[0])

            all_h_nodes_num += h_nodes_num.tolist()
            all_t_nodes_num += t_nodes_num.tolist()

            
            all_h_nei_rels += h_nei_rels
            all_t_nei_rels += t_nei_rels

    # intialize_worker(model, adj_list, dgl_adj_list, id2entity, params, node_features, kge_entity2id)
    # for link in tqdm(neg_triplets, total=len(neg_triplets)):
    #     head_scores, head_rank, tail_scores, tail_rank = get_rank(link)
    #     ranks.append(head_rank)
    #     ranks.append(tail_rank)

    #     all_head_scores += head_scores.tolist()
    #     all_tail_scores += tail_scores.tolist()

    if params.mode == 'ruleN':
        save_score_to_file_from_ruleN(neg_triplets, all_head_scores, all_tail_scores, id2entity, id2relation)
    else:
        save_score_to_file(neg_triplets, all_head_scores, all_tail_scores, id2entity, id2relation, all_h_nodes_num, all_t_nodes_num, all_h_nei_rels, all_t_nei_rels)

    # isHit1List = [x for x in ranks if x <= 1]
    # isHit5List = [x for x in ranks if x <= 5]
    # isHit10List = [x for x in ranks if x <= 10]
    # hits_1 = len(isHit1List) / len(ranks)
    # hits_5 = len(isHit5List) / len(ranks)
    # hits_10 = len(isHit10List) / len(ranks)

    # mrr = np.mean(1 / np.array(ranks))

    # logger.info(f'MRR | Hits@1 | Hits@5 | Hits@10 : {mrr} | {hits_1} | {hits_5} | {hits_10}')
    # 区分subgraph nodes num为2以及大于2的结果
    isHit1List_e2 = []
    isHit1List_g2 = []
    isHit5List_e2 = []
    isHit5List_g2 = []
    isHit10List_e2 = []
    isHit10List_g2 = []
    ranks_e2 = []
    ranks_g2 = []
    for x, n in zip(ranks, nodes_num):
        # nodes num 为2
        if n == 2:
            if x <= 1: isHit1List_e2.append(x)
            if x <= 5: isHit5List_e2.append(x)
            if x <= 10: isHit10List_e2.append(x)
            ranks_e2.append(x)
        else:
            if x <= 1: isHit1List_g2.append(x)
            if x <= 5: isHit5List_g2.append(x)
            if x <= 10: isHit10List_g2.append(x)
            ranks_g2.append(x)

    hits_1_e2 = len(isHit1List_e2) / len(ranks_e2)
    hits_5_e2 = len(isHit5List_e2) / len(ranks_e2)
    hits_10_e2 = len(isHit10List_e2) / len(ranks_e2)
    hits_1_g2 = len(isHit1List_g2) / len(ranks_g2)
    hits_5_g2 = len(isHit5List_g2) / len(ranks_g2)
    hits_10_g2 = len(isHit10List_g2) / len(ranks_g2)
    mrr_e2 = np.mean(1 / np.array(ranks_e2))
    mrr_g2 = np.mean(1 / np.array(ranks_g2))
    logger.info(f'Nodes num equal to 2: Nums | MRR | Hits@1 | Hits@5 | Hits@10 : {len(ranks_e2)} | {mrr_e2} | {hits_1_e2} | {hits_5_e2} | {hits_10_e2}')
    logger.info(f'Nodes num greater than 2: Nums | MRR | Hits@1 | Hits@5 | Hits@10 : {len(ranks_g2)} | {mrr_g2} | {hits_1_g2} | {hits_5_g2} | {hits_10_g2}')
    
    
    isHit1List = [x for x in ranks if x <= 1]
    isHit5List = [x for x in ranks if x <= 5]
    isHit10List = [x for x in ranks if x <= 10]
    hits_1 = len(isHit1List) / len(ranks)
    hits_5 = len(isHit5List) / len(ranks)
    hits_10 = len(isHit10List) / len(ranks)
    mrr = np.mean(1 / np.array(ranks))
    

    logger.info(f'Total: MRR | Hits@1 | Hits@5 | Hits@10 : {mrr} | {hits_1} | {hits_5} | {hits_10}')
    logger.info(f"Time used: {time.time() - s_t}")

    # 保存变量
    result_npy = os.path.join('./experiments', params.experiment_name, 'result-' + time.strftime('%Y%m%d%H',time.localtime(time.time()))  )
    # if not os.path.isfile(result_npy):
    result = {'ranks': ranks, 'nodes_num': nodes_num, "head_scores": head_scores, 'tail_scores': tail_scores}
    np.save(result_npy, result)

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Testing script for hits@10')

    # Experiment setup params
    parser.add_argument("--experiment_name", "-e", type=str, default="fb_v2_margin_loss",
                        help="Experiment name. Log file with this name will be created")
    parser.add_argument("--dataset", "-d", type=str, default="FB237_v2",
                        help="Path to dataset")
    parser.add_argument("--mode", "-m", type=str, default="sample", choices=["sample", "all", "ruleN"],
                        help="Negative sampling mode")
    parser.add_argument("--use_kge_embeddings", "-kge", type=bool, default=False,
                        help='whether to use pretrained KGE embeddings')
    parser.add_argument("--kge_model", type=str, default="TransE",
                        help="Which KGE model to load entity embeddings from")
    parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=True,
                        help='whether to only consider enclosing subgraph')
    parser.add_argument("--hop", type=int, default=3,
                        help="How many hops to go while eextracting subgraphs?")
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=False,
                        help='Whether to append adj matrix list with symmetric relations?')
    
    params = parser.parse_args()

    params.device = 'cpu'

    params.file_paths = {
        'graph': os.path.join('./data', params.dataset, 'train.txt'),
        'links': os.path.join('./data', params.dataset, 'test.txt')
    }

    params.ruleN_pred_path = os.path.join('./data', params.dataset, 'pos_predictions.txt')
    params.model_path = os.path.join('experiments', params.experiment_name, 'best_graph_classifier.pth')

    file_handler = logging.FileHandler(os.path.join('experiments', params.experiment_name, f'log_rank_test_{time.time()}.txt'))
    logger = logging.getLogger()
    logger.addHandler(file_handler)

    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v
                          in sorted(dict(vars(params)).items())))
    logger.info('============================================')

    main(params)
