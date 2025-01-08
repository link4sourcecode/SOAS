import numpy as np
import pandas as pd
import torch
from util.config_for_all import parse_args
import transformers
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from functools import partial
import torch.nn.functional as F
import evaluate

args = parse_args()


def get_model(model_index=args.model):
    from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights, load_checkpoint_in_model
    cuda_list = '0, 1'.split(',')
    memory = '12.5GiB'
    cuda_memory = {int(cuda): memory for cuda in cuda_list}
    if model_index == 'gpt-j-6b':
        config = transformers.AutoConfig.from_pretrained('./HuggingFace/gpt-j-6b', output_hidden_states=True)
        # with init_empty_weights():
        #     model = transformers.AutoModelForCausalLM.from_pretrained('./HuggingFace/gpt-j-6b', config=config)
        # no_split_module_classes = ["GPTJBlock"]
        # device_map = infer_auto_device_map(model, max_memory=cuda_memory,
        #                                    no_split_module_classes=no_split_module_classes)
        # print(device_map)
        # load_checkpoint_in_model(model, './HuggingFace/gpt-j-6b', device_map=device_map)
        # model = dispatch_model(model, device_map=device_map)
        model = transformers.AutoModelForCausalLM.from_pretrained('./HuggingFace/gpt-j-6b', config=config, device_map='auto')
        tokenizer = transformers.GPT2Tokenizer.from_pretrained('./HuggingFace/gpt-j-6b')
    elif model_index == 'gpt-neo-1.3b':
        config = transformers.AutoConfig.from_pretrained('./HuggingFace/gpt-neo-1.3b', output_hidden_states=True)
        model = transformers.AutoModelForCausalLM.from_pretrained('./HuggingFace/gpt-neo-1.3b', config=config)
        tokenizer = transformers.GPT2Tokenizer.from_pretrained('./HuggingFace/gpt-neo-1.3b')
    elif model_index == 'gpt-neo-2.7b':
        config = transformers.AutoConfig.from_pretrained('./HuggingFace/gpt-neo-2.7b', output_hidden_states=True)
        model = transformers.AutoModelForCausalLM.from_pretrained('./HuggingFace/gpt-neo-2.7b', config=config)
        tokenizer = transformers.GPT2Tokenizer.from_pretrained('./HuggingFace/gpt-neo-2.7b')
    else:
        config = transformers.AutoConfig.from_pretrained('./HuggingFace/gpt-neo-125m', output_hidden_states=True)
        model = transformers.AutoModelForCausalLM.from_pretrained('./HuggingFace/gpt-neo-125m', config=config)
        # model = transformers.AutoModelForCausalLM.from_pretrained('./HuggingFace/gpt-neo-125m', config=config, load_in_8bit=True)
        tokenizer = transformers.GPT2Tokenizer.from_pretrained('./HuggingFace/gpt-neo-125m')
    if model_index != 'gpt-j-6b':
        model = model.to(args.device)
        model = model.half().eval()
        # model = model.eval()
    else:
        model = model.half().eval()
        model = model.eval()
    num_of_layers = len(model.transformer.h)
    emb_dim = model.transformer.embed_dim
    projection_matrix = model.lm_head.weight.half().detach()
    return model, tokenizer, num_of_layers, emb_dim, projection_matrix


def create_dataloaders(args):
    train_prefix = np.load(args.prefix_root)
    train_suffix = np.load(args.suffix_root)

    if args.chunk:
        train_prefix = train_prefix[args.checkpoint:args.checkpoint + args.chunk]
        train_suffix = train_suffix[args.checkpoint:args.checkpoint + args.chunk]

    train_dataset = GenDataset(train_prefix, train_suffix)

    # dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)
    #
    # train_sampler = SequentialSampler(train_dataset)
    # train_dataloader = dataloader_class(train_dataset,
    #                                     batch_size=args.bs,
    #                                     sampler=train_sampler,
    #                                     drop_last=False)
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=False, drop_last=False)
    return train_dataloader


def create_dataloaders_steering(args):
    train_prefix = np.load(args.prefix_root)
    train_suffix = np.load(args.suffix_root)

    if args.chunk:
        train_prefix = train_prefix[args.checkpoint:args.checkpoint + args.chunk]
        train_suffix = train_suffix[args.checkpoint:args.checkpoint + args.chunk]

    train_dataset = GenDataset(train_prefix, train_suffix)

    dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = dataloader_class(train_dataset,
                                        batch_size=1,
                                        sampler=train_sampler,
                                        drop_last=False)
    return train_dataloader


def create_dataloaders_ppo(args):
    pre_pre_suf = pd.read_csv(args.new_root)

    if args.chunk:
        pre_pre_suf = pre_pre_suf[args.checkpoint:args.checkpoint + args.chunk]
    pre_prefix = pre_pre_suf['pre_and_prefix']
    prefix = pre_pre_suf['prefix']
    suffix = pre_pre_suf['suffix']
    pre_prefix_list, prefix_list, suffix_list = [], [], []
    for i in range(len(prefix)):
        pre_prefix_list.append(pre_prefix[i + args.checkpoint])
        prefix_list.append(prefix[i + args.checkpoint])
        suffix_list.append(suffix[i + args.checkpoint])

    train_dataset = GenDataset_ppo(pre_prefix_list, prefix_list, suffix_list)

    dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = dataloader_class(train_dataset,
                                        batch_size=args.bs,
                                        sampler=train_sampler,
                                        drop_last=False)
    return train_dataloader


class GenDataset(Dataset):

    def __init__(self,
                 prefixes,
                 suffixes,
                 ):
        self.prefixes = prefixes.astype(np.int64)
        self.suffixes = suffixes.astype(np.int64)

    def __len__(self) -> int:
        return len(self.prefixes)

    def __getitem__(self, idx: int) -> dict:
        prefix_ids = self.prefixes[idx]
        suffix_ids = self.suffixes[idx]
        input_ids = np.concatenate([prefix_ids, suffix_ids], axis=0)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        data = dict(
            input_ids=input_ids,
        )

        return data


class GenDataset_ppo(Dataset):

    def __init__(self,
                 pre_prefixes,
                 prefixes,
                 suffixes,
                 ):
        if args.model == 'gpt-j-6b':
            print('-----select gpt-j tokenizer-----')
            tokenizer = transformers.GPT2Tokenizer.from_pretrained('./HuggingFace/gpt-j-6b', padding_side='left')
        else:
            print('-----select gpt-neo tokenizer-----')
            tokenizer = transformers.GPT2Tokenizer.from_pretrained('./HuggingFace/gpt-neo-125m', padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        pre_prefixes = tokenizer(pre_prefixes, return_tensors="pt", padding=True)
        prefixes = tokenizer(prefixes, return_tensors="pt", padding=True)
        suffixes = tokenizer(suffixes, return_tensors="pt", padding=True)
        self.pre_prefixes = pre_prefixes["input_ids"]
        self.prefixes = prefixes["input_ids"]
        self.suffixes = suffixes["input_ids"]

    def __len__(self) -> int:
        return len(self.prefixes)

    def __getitem__(self, idx: int) -> dict:
        pre_prefix_ids = self.pre_prefixes[idx]
        prefix_ids = self.prefixes[idx]
        suffix_ids = self.suffixes[idx]

        # prefix_ids = torch.tensor(prefix_ids, dtype=torch.long)
        # suffix_ids = torch.tensor(suffix_ids, dtype=torch.long)
        data = dict(
            pre_query=pre_prefix_ids,
            query=prefix_ids,
            response=suffix_ids,
        )

        return data


def cal_js_div(p_output=None, q_output=None):
    """
    Function that measures JS divergence between target and output logits:
    """
    if not torch.is_tensor(p_output):
        p_output = torch.tensor(p_output, dtype=torch.float)
    if not torch.is_tensor(q_output):
        q_output = torch.tensor(q_output, dtype=torch.float)
    p_output_smx = F.softmax(p_output, dim=-1)
    q_output_smx = F.softmax(q_output, dim=-1)
    log_smx_p = F.log_softmax(p_output, dim=-1)
    log_smx_q = F.log_softmax(q_output, dim=-1)
    m = 0.5 * (p_output_smx + q_output_smx)
    kl1 = F.kl_div(log_smx_p, m, reduction="none").mean(-1)
    kl2 = F.kl_div(log_smx_q, m, reduction="none").mean(-1)
    js_divs = 0.5 * (kl1 + kl2).mean(-1)
    return js_divs


def get_metric(pred_pre, pred_post, gt):
    # be sure to check your envs to satisfy the application of Bert Score and N-Sacre BLEU
    bleu_root = r'C:\Users\User\.conda\envs\your_envs_name\Lib\site-packages\evaluate\metrics\sacrebleu'
    bert_root = r'C:\Users\User\.conda\envs\your_envs_name\Lib\site-packages\evaluate\metrics\bertscore'
    sacre_bleu = evaluate.load(bleu_root, 'sacrebleu')
    bert_score = evaluate.load(bert_root, 'bertscore')
    bleu_pre = sacre_bleu.compute(predictions=[pred_pre], references=[[gt]])['score']
    bleu_post = sacre_bleu.compute(predictions=[pred_post], references=[[gt]])['score']
    bert_pre = bert_score.compute(
        predictions=[pred_pre], references=[[gt]], model_type='distilbert-base-uncased')['f1'][0]
    bert_post = bert_score.compute(
        predictions=[pred_post], references=[[gt]], model_type='distilbert-base-uncased')['f1'][0]

    return (100 - bleu_pre), (100 - bleu_post), bert_pre, bert_post


def orthogonal_decom(commonsense, steering):
    steering = steering.to(commonsense.device)
    dot_product = torch.dot(commonsense, steering)

    # Compute the projection of b onto a
    projection = dot_product / torch.dot(commonsense, commonsense) * commonsense

    # Compute the component of b orthogonal to a
    orthogonal_component = steering - projection
    return projection, orthogonal_component


def get_steering_vector(args):
    steering_vector = torch.load('./output/steering_vector_small.pt')
    root = ''
    if args.model == 'gpt-neo-125m':
        steering_vector= torch.load(f'./output/steering_vector_small.pt')
        # steering_vector = torch.load(f'./candidate/steering_vector_decom_small.pt')
        root = f'./output/steering_vector_small.pt'
    elif args.model == 'gpt-neo-1.3b':
        steering_vector= torch.load(f'./output/steering_vector_decom_medium_150.pt')
        # steering_vector = torch.load(f'./candidate/steering_vector_decom_medium.pt')
        root = f'./output/steering_vector_medium.pt'
    elif args.model == 'gpt-neo-2.7b':
        steering_vector = torch.load(f'./output/steering_vector_decom_large_150.pt')
        # steering_vector = torch.load(f'./candidate/steering_vector_decom_large.pt')
        root = f'./output/steering_vector_large.pt'
    elif args.model == 'gpt-j-6b':
        steering_vector = torch.load(f'./output/steering_vector_decom_super_150.pt')
        # steering_vector = torch.load(f'./candidate/steering_vector_decom_super.pt')
        root = f'./output/steering_vector_super.pt'
    else:
        print('\n steering vector loading error!')

    return steering_vector, root
