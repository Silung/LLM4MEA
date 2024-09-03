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

try:
    import torch_directml
except:
    print("Direct-ml not available")


def train(args, export_root=None, resume=False):
    args.lr = 0.001
    fix_random_seed_as(args.model_init_seed)
    train_loader, val_loader, test_loader = dataloader_factory(args)

    if args.model_code == 'bert':
        model = BERT(args)
    elif args.model_code == 'sas':
        model = SASRec(args)
    elif args.model_code == 'narm':
        model = NARM(args)
    elif args.model_code == 'gru':
        model = GRU4REC(args)

    if export_root == None:
        export_root = 'experiments/' + args.model_code + '/' + args.dataset_code
    
    start_epoch = 0
    last_accum_iter = 0
    if resume:
        try: 
            # state_dict = torch.load(os.path.join(export_root, 'models', 'best_acc_model.pth'), map_location='cpu')
            state_dict = torch.load(os.path.join(export_root, 'models', 'checkpoint-recent.pth'), map_location='cpu')
            model.load_state_dict(state_dict.get(STATE_DICT_KEY))
            start_epoch = state_dict.get(EPOCH_STATE_DICT_KEY)
            last_accum_iter = state_dict.get(ACC_ITER_STATE_DICT_KEY)
        except FileNotFoundError:
            print('Failed to load old model, continue training new model...')

    if args.model_code == 'bert':
        trainer = BERTTrainer(args, model, train_loader, val_loader, test_loader, export_root, start_epoch, last_accum_iter)
    elif args.model_code == 'sas':
        trainer = SASTrainer(args, model, train_loader, val_loader, test_loader, export_root, start_epoch, last_accum_iter)
    elif args.model_code == 'narm':
        args.num_epochs = 100
        trainer = RNNTrainer(args, model, train_loader, val_loader, test_loader, export_root, start_epoch, last_accum_iter)
    elif args.model_code == 'gru':
        trainer = GRUTrainer(args, model, train_loader, val_loader, test_loader, export_root, start_epoch, last_accum_iter)
    trainer.train()
    trainer.test()


if __name__ == "__main__":
    set_template(args)

    # when use k-core beauty and k is not 5 (beauty-dense)
    # args.min_uc = k
    # args.min_sc = k
    if args.device =='dml' and torch_directml.is_available():
        args.device = torch_directml.device(torch_directml.default_device())
    train(args, resume=True)
