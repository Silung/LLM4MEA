from datasets import DATASETS
from config import STATE_DICT_KEY
from config import EPOCH_STATE_DICT_KEY
from config import ACC_ITER_STATE_DICT_KEY
import argparse
import torch
from model import *
from dataloader import *
from trainer import *
from utils import *

def test(args, model_root):
    fix_random_seed_as(args.model_init_seed)
    _, _, test_loader = dataloader_factory(args)

    if args.model_code == 'bert':
        model = BERT(args)
    elif args.model_code == 'sas':
        model = SASRec(args)
    elif args.model_code == 'narm':
        model = NARM(args)
    
    start_epoch = 0
    last_accum_iter = 0
    state_dict = torch.load(os.path.join(model_root, 'models', 'best_acc_model.pth'), map_location='cpu')
    model.load_state_dict(state_dict.get(STATE_DICT_KEY))
    start_epoch = state_dict.get(EPOCH_STATE_DICT_KEY)
    last_accum_iter = state_dict.get(ACC_ITER_STATE_DICT_KEY)

    if args.model_code == 'bert':
        trainer = BERTTrainer(args, model, None, None, test_loader, model_root, start_epoch, last_accum_iter)
    if args.model_code == 'sas':
        trainer = SASTrainer(args, model, None, None, test_loader, model_root, start_epoch, last_accum_iter)
    elif args.model_code == 'narm':
        trainer = RNNTrainer(args, model, None, None, test_loader, model_root, start_epoch, last_accum_iter)

    trainer.test()

if __name__ == "__main__":
    set_template(args, 'test')
    model_root = input("Input model's fold path\n")

    test(args, model_root)