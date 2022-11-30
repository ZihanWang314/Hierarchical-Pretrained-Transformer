
import torch
import os
import argparse
from utils import input_to_batch, Tokenizer, init_logger
from model.modelling_hpt import HPTModel
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
parser = argparse.ArgumentParser()
parser.add_argument('--logdir', default='defaultlog.txt', type=str)
parser.add_argument('--accumulation_step', default=2, type=int)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
parser.add_argument('--mode', type=str)
parser.add_argument('--epoch', type=int)
parser.add_argument('--data_root', type=str)
parser.add_argument('--model_root', type=str)
parser.add_argument('--tqdm', action='store_true')

#model argument
parser.add_argument('--train_condition', action='store_true')
parser.add_argument('--contrastive_learning', action='store_true')
parser.add_argument('--warmup_epoch_num', type=int, default=10)
parser.add_argument('--total_epoch_num', type=int, default=100)
parser.add_argument('--contrastive_mode', type=str, default='hpt')
parser.add_argument('--model_hidden_size', type=int, default=768)
parser.add_argument('--ha_num_heads', type=int, default=12)
parser.add_argument('--ha_hidden_size', type=int, default=64)
parser.add_argument('--output_file', type=str, default='outputs/output')
parser.add_argument('--inference_data', type=str, default='dev_data')
parser.add_argument('--repeat', type=int, default=10)


args = parser.parse_args()

assert args.contrastive_mode in ['hpt', 'simcse'], 'contrastive mode definition assertion'
if args.contrastive_learning:
    print(f'using {args.contrastive_mode} for contrastive learning')

logger = init_logger(args.logdir)


def main():
    # train_inputs = convert_examples_to_inputs(train_examples)
    # dev_inputs = convert_examples_to_inputs(dev_examples)
    # torch.save(train_inputs, 'data/train_inputs')
    # torch.save(dev_inputs, 'data/dev_inputs')
    start = args.epoch
    if args.mode == 'train':
        train_inputs = torch.load(os.path.join(args.data_root, 'train_inputs'))
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        train_inputs = input_to_batch(train_inputs, batch_size = args.batch_size, distributed = True)
        config = args
        model = HPTModel(config)
        if start > 0:
            model.load_state_dict(torch.load(os.path.join(args.model_root, 'model_current.pt'), map_location='cpu'))
        else:
            print('initializing model from longformer-base-4096')
            torch.save(model.state_dict(), os.path.join(args.model_root, 'model_current.pt'))
        model.train(train_inputs)

    elif args.mode == 'inference':
        if args.inference_data == 'train_data':
            dev_inputs = torch.load(os.path.join(args.data_root, 'train_inputs'))
        elif args.inference_data == 'dev_data':
            dev_inputs = torch.load(os.path.join(args.data_root, 'dev_inputs'))

        dev_inputs = [[j.cuda() for j in i] for i in dev_inputs] 
        dev_inputs = input_to_batch(dev_inputs, batch_size = 16, distributed = False) 
        model = HPTModel(args)
        model.load_state_dict(torch.load(os.path.join(args.model_root, 'model_current.pt'), map_location='cpu'))

        logger.log(f'epoch_{start + 1}')
        metric = model.answering_questions(dev_inputs, 2)
        try:
            with open(os.path.join(args.model_root, 'result.txt'), 'r') as file:
                best_performance = float(file.readlines()[0])
        except:
            best_performance = 0
        if metric > best_performance:
            model.activate_normal_mode()
            torch.save(model.state_dict(), os.path.join(args.model_root, 'model_best.pt'))
            logger.log('achieving best result, now saving to model_best.pt')
            with open(os.path.join(args.model_root, 'result.txt'), 'w') as file:
                file.write(str(metric))


if __name__ == '__main__':
    main()
