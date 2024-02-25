import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import logging
import torch
from scipy.sparse import SparseEfficiencyWarning

from subgraph_extraction.datasets import SubgraphDataset, generate_subgraph_datasets
from utils.initialization_utils import initialize_experiment, initialize_model
from utils.graph_utils import collate_dgl, collate_dgl_train, move_batch_to_device_dgl, move_batch_to_device_dgl_train

from model.dgl.graph_classifier import GraphClassifier as dgl_model

from managers.evaluator import Evaluator
from managers.trainer import Trainer

from warnings import simplefilter


def main(params):
    simplefilter(action='ignore', category=UserWarning)
    simplefilter(action='ignore', category=SparseEfficiencyWarning)

    params.db_path = os.path.join(params.main_dir, f'./data/{params.dataset}/subgraphs_en_{params.enclosing_sub_graph}_neg_{params.num_neg_samples_per_link}_hop_{params.hop}')

    if not os.path.isdir(params.db_path):
        generate_subgraph_datasets(params)

    train = SubgraphDataset(params.db_path, 'train_pos', 'train_neg', params.file_paths,
                            add_traspose_rels=params.add_traspose_rels,
                            num_neg_samples_per_link=params.num_neg_samples_per_link,
                            use_kge_embeddings=params.use_kge_embeddings, dataset=params.dataset,
                            kge_model=params.kge_model, file_name=params.train_file)
    valid = SubgraphDataset(params.db_path, 'valid_pos', 'valid_neg', params.file_paths,
                            add_traspose_rels=params.add_traspose_rels,
                            num_neg_samples_per_link=params.num_neg_samples_per_link,
                            use_kge_embeddings=params.use_kge_embeddings, dataset=params.dataset,
                            kge_model=params.kge_model, file_name=params.valid_file)

    # Avoid that m_h2r and m_t2r of train and valid are different
    valid.m_h2r = train.m_h2r
    valid.m_t2r = train.m_t2r

    params.num_rels = train.num_rels
    params.aug_num_rels = train.aug_num_rels

    # Set the embedding dimension of relation and node
    if params.init_nei_rels == 'no':
        params.inp_dim = train.n_feat_dim
    else:
        params.inp_dim = train.n_feat_dim + params.sem_dim

    # Log the max label value to save it in the model. This will be used to cap the labels generated on test set.
    params.max_label_value = train.max_n_label

    graph_classifier = initialize_model(params, dgl_model, params.load_model)

    logging.info(f"Device: {params.device}")
    logging.info(f"Input dim : {params.inp_dim}, # Relations : {params.num_rels}, # Augmented relations : {params.aug_num_rels}")

    valid_evaluator = Evaluator(params, graph_classifier, valid)

    trainer = Trainer(params, graph_classifier, train, valid_evaluator)

    logging.info('Starting training with full batch...')

    trainer.train()


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='TransE model')

    # Experiment setup params
    parser.add_argument("--experiment_name", "-e", type=str, default="default",
                        help="A folder with this name would be created to dump saved models and log files")
    parser.add_argument("--dataset", "-d", type=str,
                        help="Dataset string")
    parser.add_argument("--gpu", type=int, default=0,
                        help="Which GPU to use?")
    parser.add_argument('--disable_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--load_model', action='store_true',
                        help='Load existing model?')
    parser.add_argument("--train_file", "-tf", type=str, default="train",
                        help="Name of file containing training triplets")
    parser.add_argument("--valid_file", "-vf", type=str, default="valid",
                        help="Name of file containing validation triplets")

    # Training regime params
    parser.add_argument("--num_epochs", "-ne", type=int, default=30,
                        help="Learning rate of the optimizer")
    parser.add_argument("--eval_every", type=int, default=3,
                        help="Interval of epochs to evaluate the model?")
    parser.add_argument("--eval_every_iter", type=int, default=455,
                        help="Interval of iterations to evaluate the model?")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Interval of epochs to save a checkpoint of the model?")
    parser.add_argument("--early_stop", type=int, default=100,
                        help="Early stopping patience")
    parser.add_argument("--optimizer", type=str, default="Adam",
                        help="Which optimizer to use?")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate of the optimizer")
    parser.add_argument("--clip", type=int, default=1000,
                        help="Maximum gradient norm allowed")
    parser.add_argument("--l2", type=float, default=5e-4,
                        help="Regularization constant for GNN weights")
    parser.add_argument("--margin", type=float, default=10,
                        help="The margin between positive and negative samples in the max-margin loss")

    # Data processing pipeline params
    parser.add_argument("--max_links", type=int, default=1000000,
                        help="Set maximum number of train links (to fit into memory)")
    parser.add_argument("--hop", type=int, default=3,
                        help="Enclosing subgraph hop number")
    parser.add_argument("--max_nodes_per_hop", "-max_h", type=int, default=None,
                        help="if > 0, upper bound the # nodes per hop by subsampling")
    parser.add_argument("--use_kge_embeddings", "-kge", type=bool, default=False,
                        help='whether to use pretrained KGE embeddings')
    parser.add_argument("--kge_model", type=str, default="TransE",
                        help="Which KGE model to load entity embeddings from")
    parser.add_argument('--model_type', '-m', type=str, choices=['ssp', 'dgl'], default='dgl',
                        help='what format to store subgraphs in for model')
    parser.add_argument('--constrained_neg_prob', '-cn', type=float, default=0.0,
                        help='with what probability to sample constrained heads/tails while neg sampling')
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--num_neg_samples_per_link", '-neg', type=int, default=1,
                        help="Number of negative examples to sample per positive link")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of dataloading processes")
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=False,
                        help='whether to append adj matrix list with symmetric relations')
    parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=True,
                        help='whether to only consider enclosing subgraph')

    # Model params
    parser.add_argument("--rel_emb_dim", "-r_dim", type=int, default=32,
                        help="Relation embedding size")
    parser.add_argument("--attn_rel_emb_dim", "-ar_dim", type=int, default=32,
                        help="Relation embedding size for attention")
    parser.add_argument("--emb_dim", "-dim", type=int, default=32,
                        help="Entity embedding size")
    parser.add_argument("--num_gcn_layers", "-l", type=int, default=3,
                        help="Number of GCN layers")
    parser.add_argument("--num_bases", "-b", type=int, default=4,
                        help="Number of basis functions to use for GCN weights")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout rate in GNN layers")
    parser.add_argument("--edge_dropout", type=float, default=0.5,
                        help="Dropout rate in edges of the subgraphs")
    parser.add_argument('--gnn_agg_type', '-a', type=str, choices=['sum', 'mlp', 'gru'], default='sum',
                        help='what type of aggregation to do in gnn msg passing')
    parser.add_argument('--add_ht_emb', '-ht', type=bool, default=True,
                        help='whether to concatenate head/tail embedding with pooled graph representation')
    parser.add_argument('--has_attn', '-attn', type=bool, default=True,
                        help='whether to have attn in model or not')
    parser.add_argument('--sem_dim', type=int, default=24,
                        help='the dimension of sematic part of node embedding')
    parser.add_argument('--max_nei_rels', type=int, default=10, help='the maximum num of neighbor relations of each node when initialzing the node embedding.')
    parser.add_argument('--nei_rels_dropout', type=float, default=0.4, help='Dropout rate in aggregating relation embeddings.')
    parser.add_argument('--is_comp', type=str, default='mult', choices=['mult', 'sub'], help='The composition manner of node and relation')
    parser.add_argument('--comp_ht', type=str, choices=['mult, mlp, sum'], default='sum', help='The composition operator of head and tail embedding')
    parser.add_argument('--comp_hrt', type=str, choices=['TransE, DistMult'], default=None, help='The composition operator of (h, r, t)embedding')
    parser.add_argument('--coef_dgi_loss', type=float, default=5, help='Coefficient of MI loss')
    parser.add_argument('--init_nei_rels', type=str, choices=['no', 'out', 'in', 'both'], default='in', help='the manner of utilizing relatioins when initializing entity embedding')
    parser.add_argument('--sort_data', type=bool, default=True,
                        help='whether to training data according to relation id ')
    parser.add_argument('--nei_rel_path', action='store_false',
                        help='whether to consider neighboring relational paths')
    parser.add_argument('--path_agg', type=str, choices=['mean', 'att'], default='att', help='the manner of aggreating neighboring relational paths.')

    params = parser.parse_args()
    initialize_experiment(params, __file__)

    params.file_paths = {
        'train': os.path.join(params.main_dir, 'data/{}/{}.txt'.format(params.dataset, params.train_file)),
        'valid': os.path.join(params.main_dir, 'data/{}/{}.txt'.format(params.dataset, params.valid_file))
    }

    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
    else:
        params.device = torch.device('cpu')

    params.collate_fn = collate_dgl_train
    params.move_batch_to_device = move_batch_to_device_dgl_train
    main(params)
