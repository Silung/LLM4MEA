from datasets import DATASETS
from config import STATE_DICT_KEY, EPOCH_STATE_DICT_KEY, ACC_ITER_STATE_DICT_KEY
import argparse
import torch
from model import *
from dataloader import *
from trainer import *
from config import *
from utils import *

try:
    import torch_directml
except:
    print("Direct-ml not available")


def distill(args, bb_model_root=None, export_root=None, resume=False):
    args.lr = 0.001
    args.enable_lr_warmup = False
    fix_random_seed_as(args.model_init_seed)
    _, _, test_loader = dataloader_factory(args)

    if args.model_code == 'bert':
        model = BERT(args)
    elif args.model_code == 'sas':
        model = SASRec(args)
    elif args.model_code == 'narm':
        model = NARM(args)
    elif args.model_code == 'gru':
        model = GRU4REC(args)
    
    model_codes = {'b': 'bert', 's':'sas', 'n':'narm'}
    if args.bb_model_code is None:
        args.bb_model_code = model_codes[input('Input black box model code, b for BERT, s for SASRec and n for NARM: ')]
    if args.num_generated_seqs is None:
        args.num_generated_seqs = int(input('Input integer number of seqs budget: '))

    if args.bb_model_code == 'bert':
        bb_model = BERT(args)
    elif args.bb_model_code == 'sas':
        bb_model = SASRec(args)
    elif args.bb_model_code == 'narm':
        bb_model = NARM(args)
    elif args.bb_model_code == 'gru':
        bb_model = GRU4REC(args)
    
    if bb_model_root == None:
        bb_model_root = 'experiments/' + args.bb_model_code + '/' + args.dataset_code
    if export_root == None:
        folder_name = args.bb_model_code + '2' + args.model_code + '_' + args.generated_sampler + str(args.k) + args.loss + str(args.num_generated_seqs)
        export_root = 'experiments/distillation_rank/' + folder_name + '/' + args.dataset_code

    bb_model.load_state_dict(torch.load(os.path.join(bb_model_root, 'models', 'best_acc_model.pth'), map_location='cpu').get(STATE_DICT_KEY))
        
    last_epoch = 0
    last_accum_iter = 0
    if resume:
        try:
            # model.load_state_dict(torch.load(os.path.join(export_root, 'models', 'best_acc_model.pth'), map_location='cpu').get(STATE_DICT_KEY))
            state_dict = torch.load(os.path.join(export_root, 'models', f'checkpoint-recent{args.id}.pth'), map_location='cpu')
            model.load_state_dict(state_dict.get(STATE_DICT_KEY))
            last_epoch = state_dict.get(EPOCH_STATE_DICT_KEY)
            last_accum_iter = state_dict.get(ACC_ITER_STATE_DICT_KEY)
        except FileNotFoundError:
            print('Failed to load old model, continue training new model...')
    trainer = NoDataRankDistillationTrainer(args, args.model_code, model, bb_model, test_loader, export_root, args.loss, last_epoch, last_accum_iter)

    trainer.train()


if __name__ == "__main__":
    set_template(args)

    # when use k-core beauty and k is not 5 (beauty-dense)
    # args.min_uc = k
    # args.min_sc = k
    if args.device =='dml' and torch_directml.is_available():
        args.device = torch_directml.device(torch_directml.default_device())
    distill(args=args, resume=args.resume)

