from .rgcn_model import RGCN
from dgl import mean_nodes
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
import numpy as np
from .discriminator import Discriminator
from .batch_gru import BatchGRU
"""
File based off of dgl tutorial on RGCN
Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""


class GraphClassifier(nn.Module):
    def __init__(self, params, relation2id, ent2rels):  # in_dim, h_dim, rel_emb_dim, out_dim, num_rels, num_bases):
        super().__init__()

        self.params = params
        self.relation2id = relation2id
        self.ent2rels = ent2rels
        self.gnn = RGCN(params)  # in_dim, h_dim, h_dim, num_rels, num_bases)

        # num_rels + 1 instead of nums_rels, in order to add a "padding" relation.
        self.rel_emb = nn.Embedding(self.params.num_rels + 1, self.params.inp_dim, sparse=False, padding_idx=self.params.num_rels)
        
        self.ent_padding = nn.Parameter(torch.FloatTensor(1, self.params.sem_dim).uniform_(-1, 1))
        if self.params.init_nei_rels == 'both':
            self.w_rel2ent = nn.Linear(2 * self.params.inp_dim, self.params.sem_dim)
        elif self.params.init_nei_rels == 'out' or 'in':
            self.w_rel2ent = nn.Linear(self.params.inp_dim, self.params.sem_dim)

        self.sigmoid = nn.Sigmoid()
        self.nei_rels_dropout = nn.Dropout(self.params.nei_rels_dropout)
        self.dropout = nn.Dropout(self.params.dropout)
        self.softmax = nn.Softmax(dim=1)

        if self.params.add_ht_emb:    
            # self.fc_layer = nn.Linear(3 * self.params.num_gcn_layers * self.params.emb_dim + self.params.rel_emb_dim, 1)
            self.fc_layer = nn.Linear(3 * self.params.num_gcn_layers * self.params.emb_dim + self.params.emb_dim, 1)
        else:
            self.fc_layer = nn.Linear(self.params.num_gcn_layers * self.params.emb_dim + self.params.rel_emb_dim, 1)

        if self.params.comp_hrt:
            self.fc_layer = nn.Linear(2 * self.params.num_gcn_layers * self.params.emb_dim, 1)
        
        if self.params.nei_rel_path:
            self.fc_layer = nn.Linear(3 * self.params.num_gcn_layers * self.params.emb_dim + 2 * self.params.emb_dim, 1)

        if self.params.comp_ht == 'mlp':
            self.fc_comp = nn.Linear(2 * self.params.emb_dim, self.params.emb_dim)

        if self.params.nei_rel_path:
            self.disc = Discriminator(self.params.num_gcn_layers * self.params.emb_dim + self.params.emb_dim, self.params.num_gcn_layers * self.params.emb_dim + self.params.emb_dim)
        else:
            self.disc = Discriminator(self.params.num_gcn_layers * self.params.emb_dim , self.params.num_gcn_layers * self.params.emb_dim)

        self.rnn = torch.nn.GRU(self.params.emb_dim, self.params.emb_dim, batch_first=True)

        self.batch_gru = BatchGRU(self.params.num_gcn_layers * self.params.emb_dim )

        self.W_o = nn.Linear(self.params.num_gcn_layers * self.params.emb_dim * 2, self.params.num_gcn_layers * self.params.emb_dim)

    def init_ent_emb_matrix(self, g):
        """ Initialize feature of entities by matrix form """
        out_nei_rels = g.ndata['out_nei_rels']
        in_nei_rels = g.ndata['in_nei_rels']
        
        target_rels = g.ndata['r_label']
        out_nei_rels_emb = self.rel_emb(out_nei_rels)
        in_nei_rels_emb = self.rel_emb(in_nei_rels)
        target_rels_emb = self.rel_emb(target_rels).unsqueeze(2)

        out_atts = self.softmax(self.nei_rels_dropout(torch.matmul(out_nei_rels_emb, target_rels_emb).squeeze(2)))
        in_atts = self.softmax(self.nei_rels_dropout(torch.matmul(in_nei_rels_emb, target_rels_emb).squeeze(2)))
        out_sem_feats = torch.matmul(out_atts.unsqueeze(1), out_nei_rels_emb).squeeze(1)
        in_sem_feats = torch.matmul(in_atts.unsqueeze(1), in_nei_rels_emb).squeeze(1)
        
        if self.params.init_nei_rels == 'both':
            ent_sem_feats = self.sigmoid(self.w_rel2ent(torch.cat([out_sem_feats, in_sem_feats], dim=1)))
        elif self.params.init_nei_rels == 'out':
            ent_sem_feats = self.sigmoid(self.w_rel2ent(out_sem_feats))
        elif self.params.init_nei_rels == 'in':
            ent_sem_feats = self.sigmoid(self.w_rel2ent(in_sem_feats))

        g.ndata['init'] = torch.cat([g.ndata['feat'], ent_sem_feats], dim=1)  # [B, self.inp_dim]

    def comp_ht_emb(self, head_embs, tail_embs):
        if self.params.comp_ht == 'mult':
            ht_embs = head_embs * tail_embs
        elif self.params.comp_ht == 'mlp':
            ht_embs = self.fc_comp(torch.cat([head_embs, tail_embs], dim=1))
        elif self.params.comp_ht == 'sum':
            ht_embs = head_embs + tail_embs
        else:
            raise KeyError(f'composition operator of head and relation embedding {self.comp_ht} not recognized.')

        return ht_embs

    def comp_hrt_emb(self, head_embs, tail_embs, rel_embs):
        rel_embs = rel_embs.repeat(1, self.params.num_gcn_layers)
        if self.params.comp_hrt == 'TransE':
            hrt_embs = head_embs + rel_embs - tail_embs
        elif self.params.comp_hrt == 'DistMult':
            hrt_embs = head_embs * rel_embs * tail_embs
        else: raise KeyError(f'composition operator of (h, r, t) embedding {self.comp_hrt} not recognized.')
        
        return hrt_embs

    def nei_rel_path(self, g, rel_labels, r_emb_out):
        """ Neighboring relational path module """
        # Only consider in-degree relations first.
        nei_rels = g.ndata['in_nei_rels']
        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        heads_rels = nei_rels[head_ids]
        tails_rels = nei_rels[tail_ids]

        # Extract neighboring relational paths
        batch_paths = []
        for (head_rels, r_t, tail_rels) in zip(heads_rels, rel_labels, tails_rels):
            paths = []
            for h_r in head_rels:
                for t_r in tail_rels:
                    path = [h_r, r_t, t_r]
                    paths.append(path)
            batch_paths.append(paths)       # [B, n_paths, 3] , n_paths = n_head_rels * n_tail_rels
        
        batch_paths = torch.LongTensor(batch_paths).to(rel_labels.device)# [B, n_paths, 3], n_paths = n_head_rels * n_tail_rels
        batch_size = batch_paths.shape[0]
        batch_paths = batch_paths.view(batch_size * len(paths), -1) # [B * n_paths, 3]

        batch_paths_embs = F.embedding(batch_paths, r_emb_out, padding_idx=-1) # [B * n_paths, 3, inp_dim]

        # Input RNN 
        _, last_state = self.rnn(batch_paths_embs) # last_state: [1, B * n_paths, inp_dim]
        last_state = last_state.squeeze(0) # squeeze the dim 0 
        last_state = last_state.view(batch_size, len(paths), self.params.emb_dim) # [B, n_paths, inp_dim]
        # Aggregate paths by attention
        if self.params.path_agg == 'mean':
            output = torch.mean(last_state, 1) # [B, inp_dim]
        
        if self.params.path_agg == 'att':
            r_label_embs = F.embedding(rel_labels, r_emb_out, padding_idx=-1) .unsqueeze(2) # [B, inp_dim, 1]
            atts = torch.matmul(last_state, r_label_embs).squeeze(2) # [B, n_paths]
            atts = F.softmax(atts, dim=1).unsqueeze(1) # [B, 1, n_paths]
            output = torch.matmul(atts, last_state).squeeze(1) # [B, 1, n_paths] * [B, n_paths, inp_dim] -> [B, 1, inp_dim] -> [B, inp_dim]
        else:
            raise ValueError('unknown path_agg')
        
        return output # [B, inp_dim]

    def get_logits(self, s_G, s_g_pos, s_g_cor): 
        ret = self.disc(s_G, s_g_pos, s_g_cor)
        return ret
    
    def forward(self, data, is_return_emb=False, cor_graph=False):
        # Initialize the embedding of entities
        g, rel_labels = data
        
        # Neighboring Relational Feature Module
        ## Initialize the embedding of nodes by neighbor relations
        if self.params.init_nei_rels == 'no':
            g.ndata['init'] = g.ndata['feat'].clone()
        else:
            self.init_ent_emb_matrix(g)
        
        # Corrupt the node feature
        if cor_graph:
            g.ndata['init'] = g.ndata['init'][torch.randperm(g.ndata['feat'].shape[0])]  
        
        # r: Embedding of relation
        r = self.rel_emb.weight.clone()
        
        # Input graph into GNN to get embeddings.
        g.ndata['h'], r_emb_out = self.gnn(g, r)
        
        # GRU layer for nodes
        graph_sizes = g.batch_num_nodes()
        out_dim = self.params.num_gcn_layers * self.params.emb_dim
        g.ndata['repr'] = F.relu(self.batch_gru(g.ndata['repr'].view(-1, out_dim), graph_sizes))
        node_hiddens = F.relu(self.W_o(g.ndata['repr']))  # num_nodes x hidden 
        g.ndata['repr'] = self.dropout(node_hiddens)  # num_nodes x hidden
        g_out = mean_nodes(g, 'repr').view(-1, out_dim)

        # Get embedding of target nodes (i.e. head and tail nodes)
        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = g.ndata['repr'][head_ids]
        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata['repr'][tail_ids]
        
        if self.params.add_ht_emb:
            g_rep = torch.cat([g_out,
                               head_embs.view(-1, out_dim),
                               tail_embs.view(-1, out_dim),
                               F.embedding(rel_labels, r_emb_out, padding_idx=-1)], dim=1)
        else:
            g_rep = torch.cat([g_out, self.rel_emb(rel_labels)], dim=1)
        
        # Represent subgraph by composing (h,r,t) in some way. (Not use in paper)
        if self.params.comp_hrt:
            edge_embs = self.comp_hrt_emb(head_embs.view(-1, out_dim), tail_embs.view(-1, out_dim), F.embedding(rel_labels, r_emb_out, padding_idx=-1))
            g_rep = torch.cat([g_out, edge_embs], dim=1)

        # Model neighboring relational paths 
        if self.params.nei_rel_path:
            # Model neighboring relational path
            g_p = self.nei_rel_path(g, rel_labels, r_emb_out)
            g_rep = torch.cat([g_rep, g_p], dim=1)
            s_g = torch.cat([g_out, g_p], dim=1)
        else:
            s_g = g_out
        output = self.fc_layer(g_rep)

        self.r_emb_out = r_emb_out
        
        if not is_return_emb:
            return output
        else:
            # Get the subgraph-level embedding
            s_G = s_g.mean(0)
            return output, s_G, s_g



