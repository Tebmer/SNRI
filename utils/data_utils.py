import os
import pdb
import logging
import numpy as np
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt


def plot_rel_dist(adj_list, filename):
    rel_count = []
    for adj in adj_list:
        rel_count.append(adj.count_nonzero())

    fig = plt.figure(figsize=(12, 8))
    plt.plot(rel_count)
    fig.savefig(filename, dpi=fig.dpi)


def process_files(files, saved_relation2id=None, add_traspose_rels=False, sort_data=False):
    '''
    files: Dictionary map of file paths to read the triplets from.
    saved_relation2id: Saved relation2id (mostly passed from a trained model) which can be used to map relations to pre-defined indices and filter out the unknown ones.
    '''
    entity2id = {}
    relation2id = {} if saved_relation2id is None else saved_relation2id

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
            if not saved_relation2id and triplet[1] not in relation2id:
                relation2id[triplet[1]] = rel
                rel += 1

            # Save the triplets corresponding to only the known relations
            if triplet[1] in relation2id:
                data.append([entity2id[triplet[0]], entity2id[triplet[2]], relation2id[triplet[1]]])

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
    
    for triplet in triplets['train']:
        h, t, r = triplet
        if h not in h2r:
            h2r_len[h] = 1
            h2r[h] = [r]
        else:
            h2r_len[h] += 1
            h2r[h].append(r)
        
        if add_traspose_rels:
            # Consider the reverse relation, the id of reverse relation is (relation + #relations)
            if t not in t2r:
                t2r[t] = [r + num_rels]
            else:
                t2r[t].append(r + num_rels)
        if t not in t2r:
            t2r[t] = [r]
            t2r_len[t]  = 1
        else:
            t2r[t].append(r)
            t2r_len[t] += 1
    
    # Consider nodes with no neighbors as index '-1' and their relation index: num_rels.
    # ent2rels[-1] = [num_rels]

    # Construct the matrix of ent2rels
    # rels_len = triplets['train'].shape(0) // num_ents
    h_nei_rels_len = int(np.percentile(list(h2r_len.values()), 75))
    t_nei_rels_len = int(np.percentile(list(t2r_len.values()), 75))
    logging.info("Average number of relations each node: ", "head: ", h_nei_rels_len, 'tail: ', t_nei_rels_len)
    
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
    
    print("Construct matrix of ent2rels done!")
    
    # Sort the data according to relation id 
    if sort_data:
        triplets['train'] = triplets['train'][np.argsort(triplets['train'][:,2])]
        
    adj_list = []
    for i in range(len(relation2id)):
        idx = np.argwhere(triplets['train'][:, 2] == i)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8), (triplets['train'][:, 0][idx].squeeze(1), triplets['train'][:, 1][idx].squeeze(1))), shape=(len(entity2id), len(entity2id))))

    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation, h2r, m_h2r, t2r, m_t2r


def save_to_file(directory, file_name, triplets, id2entity, id2relation):
    file_path = os.path.join(directory, file_name)
    with open(file_path, "w") as f:
        for s, o, r in triplets:
            f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')
