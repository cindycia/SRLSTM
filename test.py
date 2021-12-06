'''
Test script
Author: Pu Zhang
Date: 2019/7/1
'''
import argparse
from Processor import *


def get_parser():
    parser = argparse.ArgumentParser(
        description='States Refinement LSTM')
    parser.add_argument(
        '--using_cuda', default=True)  # must be True, We did not test on cpu
    # You may change these arguments (model selection and dirs)
    parser.add_argument(
        '--test_set', default=0,
        help='Set this value to 0~4 for ETH-univ, ETH-hotel, UCY-zara01, UCY-zara02, UCY-univ')
    parser.add_argument(
        '--gpu', default=0,
        help='gpu id')
    parser.add_argument(
        '--base_dir', default='.',
        help='Base directory including these scrits.')
    parser.add_argument(
        '--save_base_dir', default='./savedata/',
        help='Directory for saving caches and models.')
    parser.add_argument(
        '--phase', default='test',
        help='Set this value to \'train\' or \'test\'')
    parser.add_argument(
        '--train_model', default='testmodel',
        help='Your model name')
    parser.add_argument(
        '--load_model', default=237,  # 0:237 1:150 2:288 3:247
        help="load model weights from this index before training or testing")
    parser.add_argument(
        '--pretrain_model', default='',
        help='Your pretrained model name. Used in training second states refienemnt layer.')
    parser.add_argument(
        '--pretrain_load', default=0,
        help="load pretrained model from this index. Used in training second states refienemnt layer.")
    parser.add_argument(
        '--model', default='models.SRLSTM',
        help='Set model type in \'models.LSTM\', \'models.SRLSTM\', \'models.SocialLSTM\'')
    ######################################

    parser.add_argument(
        '--dataset', default='eth5')
    parser.add_argument(
        '--save_dir')
    parser.add_argument(
        '--model_dir')
    parser.add_argument(
        '--config')

    parser.add_argument(
        '--ifvalid', default=True,
        help="=False,use all train set to train,"
             "=True,use train set to train and valid")
    parser.add_argument(
        '--val_fraction', default=0.2)

    # Model parameters

    # LSTM
    parser.add_argument(
        '--output_size', default=2)
    parser.add_argument(
        '--input_embed_size', default=32)
    parser.add_argument(
        '--rnn_size', default=64)
    parser.add_argument(
        '--hidden_dot_size', default=32)
    parser.add_argument(
        '--ifdropout', default=True)
    parser.add_argument(
        '--dropratio', default=0.1)
    parser.add_argument(
        '--std_in', default=0.2)
    parser.add_argument(
        '--std_out', default=0.1)

    # States Refinement
    parser.add_argument(
        '--ifbias_gate', default=True)
    parser.add_argument(
        '--WAr_ac', default='')
    parser.add_argument(
        '--ifbias_WAr', default=False)
    parser.add_argument(
        '--input_size', default=2)

    parser.add_argument(
        '--rela_embed_size', default=32)
    parser.add_argument(
        '--rela_hidden_size', default=16)
    parser.add_argument(
        '--rela_layers', default=1)
    parser.add_argument(
        '--rela_input', default=2)
    parser.add_argument(
        '--rela_drop', default=0.1)
    parser.add_argument(
        '--rela_ac', default='relu')
    parser.add_argument(
        '--ifbias_rel', default=True)

    parser.add_argument(
        '--nei_hidden_size', default=64)
    parser.add_argument(
        '--nei_layers', default=1)
    parser.add_argument(
        '--nei_drop', default=0)
    parser.add_argument(
        '--nei_ac', default='')

    parser.add_argument(
        '--ifbias_nei', default=False)
    parser.add_argument(
        '--mp_ac', default='')
    parser.add_argument(
        '--nei_std', default=0.01)
    parser.add_argument(
        '--rela_std', default=0.3)
    parser.add_argument(
        '--WAq_std', default=0.05)
    parser.add_argument(
        '--passing_time', default=1,
        help='States refinement layers.')

    # Social LSTM
    parser.add_argument(
        '--grid_size', default=4)
    parser.add_argument(
        '--nei_thred_slstm', default=2)

    # Perprocess
    parser.add_argument(
        '--seq_length', default=20)
    parser.add_argument(
        '--obs_length', default=8)
    parser.add_argument(
        '--pred_length', default=12)
    parser.add_argument(
        '--batch_around_ped', default=128)
    parser.add_argument(
        '--batch_size', default=8)
    parser.add_argument(
        '--val_batch_size', default=8)
    parser.add_argument(
        '--test_batch_size', default=4)
    parser.add_argument(
        '--show_step', default=40)
    parser.add_argument(
        '--start_test', default=100)
    parser.add_argument(
        '--num_epochs', default=250)
    parser.add_argument(
        '--ifshow_detail', default=True)
    parser.add_argument(
        '--ifdebug', default=False)
    parser.add_argument(
        '--ifsave_results', default=False)
    parser.add_argument(
        '--randomRotate', default=True,
        help="=True:random rotation of each trajectory fragment")
    parser.add_argument(
        '--neighbor_thred', default=10)
    parser.add_argument(
        '--learning_rate', default=0.0015)
    parser.add_argument(
        '--clip', default=1)
    return parser


def load_arg(p):
    # save arg
    if os.path.exists(p.config):
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                try:
                    assert (k in key)
                except:
                    s = 1
        parser.set_defaults(**default_arg)
        return parser.parse_args()
    else:
        return False


def save_arg(args):
    # save arg
    arg_dict = vars(args)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    with open(args.config, 'w') as f:
        yaml.dump(arg_dict, f)


if __name__ == '__main__':
    parser = get_parser()
    p = parser.parse_args()
    p.save_dir = p.save_base_dir + str(p.test_set) + '/'
    p.model_dir = p.save_base_dir + str(p.test_set) + '/' + p.train_model + '/'
    p.config = p.model_dir + '/config_' + p.phase + '.yaml'

    if not load_arg(p):
        save_arg(p)
    args = load_arg(p)
    torch.cuda.set_device(args.gpu)
    processor = Processor(args)
    if args.phase == 'test':
        processor.playtest()
    else:
        processor.playtrain()
