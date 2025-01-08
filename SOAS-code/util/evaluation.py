import torch
from util.config_for_all import parse_args
from util.my_util import get_model, get_metric
from tqdm import tqdm
import warnings
import numpy as np
warnings.filterwarnings('ignore')

def defense_evaluation(args, model, tokenizer):
    tokenizer.pad_token = tokenizer.eos_token
    seq_gt = torch.load(
        f'./output/seq_gt_{args.checkpoint}_{args.checkpoint + args.chunk}.pt').to(torch.long)
    seq_pre = torch.load(
        f'./output/seq_post_{args.checkpoint}_{args.checkpoint + args.chunk}.pt').to(torch.long)
    seq_post = torch.load(
        f'./output/seq_post_{args.checkpoint}_{args.checkpoint + args.chunk}.pt').to(torch.long)
    bleu_pre, bleu_post, bert_pre, bert_post = torch.zeros(
        args.chunk, args.trail), torch.zeros(args.chunk, args.trail), torch.zeros(
        args.chunk, args.trail), torch.zeros(args.chunk, args.trail)
    ppl_gt, ppl_pre, ppl_post = torch.zeros(
        args.chunk), torch.zeros(args.chunk, args.trail), torch.zeros(args.chunk, args.trail)
    # print(seq_gt[0][:10])
    # print(seq_pre[0][:10])
    # print(seq_post[0][:10])
    count = 0
    for i in tqdm(range(args.chunk), desc='SacreBLEU Evaluation'):
        gt = seq_gt[i]
        pre = seq_pre[i * args.trail:(i + 1) * args.trail]
        post = seq_post[i * args.trail:(i + 1) * args.trail]
        # if torch.equal(post[0], gt) is True and torch.equal(post[1], gt) is True and torch.equal(post[2], gt) is True:
        if torch.equal(post[0], gt) is True and torch.equal(post[1], gt) is True:
            count += 1
        gt_copy, pre_copy, post_copy = torch.clone(gt), torch.clone(pre), torch.clone(post)
        gt_copy[:args.prefix_len] = -100
        pre_copy[:, :args.prefix_len] = -100
        post_copy[:, :args.prefix_len] = -100
        with torch.no_grad():
            output_gt = model(
                gt.unsqueeze(0).to(args.device), labels=gt_copy.unsqueeze(0).to(args.device))
        de_gt = tokenizer.decode(gt[args.prefix_len:])
        de_pre = tokenizer.batch_decode(pre[:, args.prefix_len:])
        de_post = tokenizer.batch_decode(post[:, args.prefix_len:])
        for idx in range(args.trail):
            with torch.no_grad():
                output_pre = model(
                    pre[idx].unsqueeze(0).to(args.device), labels=pre_copy[idx].unsqueeze(0).to(args.device))
                output_post = model(
                    post[idx].unsqueeze(0).to(args.device), labels=post_copy[idx].unsqueeze(0).to(args.device))
            bleu_pre[i, idx], bleu_post[i, idx], bert_pre[i, idx], bert_post[i, idx] = get_metric(
                de_pre[idx], de_post[idx], de_gt)
            ppl_gt[i] = torch.exp(output_gt[:1][0])
            ppl_pre[i, idx], ppl_post[i, idx] = torch.exp(output_pre[:1][0]), torch.exp(output_post[:1][0])
            with open(f'./eval/result_{i + args.checkpoint}.txt', 'a', encoding='utf-8') as f:
                if idx == 0:
                    f.truncate(0)
                f.write(f'\nseq num.{i} and test time num.{idx}.\n')
                f.write(f'{de_gt}\r\n')
                f.write(f'{de_pre[0]}\r\n')
                f.write(f'{de_post[0]}\r\n')
                f.write(f'the bleu of gt v.s. pre is: {bleu_pre[i, idx]}\n')
                f.write(f'the bleu of gt v.s. post is: {bleu_post[i, idx]}\n')
                f.write(f'the bert of gt v.s. pre is: {bert_pre[i, idx]}\n')
                f.write(f'the bert of gt v.s. pre is: {bert_post[i, idx]}\n')
                f.write(f'the ppl of gt is: {ppl_gt[i]}\n')
                f.write(f'the ppl of pre is: {ppl_pre[i, idx]}\n')
                f.write(f'the ppl of post is: {ppl_post[i, idx]}\n')
                f.write('----------------------------------------------------------------')

    best_ppl_post = []
    best_ppl_pre = []
    best_bleu_post, pre_sort = torch.sort(bleu_post, descending=True, dim=1)
    best_bert_post, post_sort = torch.sort(bert_post, descending=False, dim=1)
    for idx, x in enumerate(post_sort):
        best_ppl_post.append(ppl_post[idx, x[0].item()])
    for idy, y in enumerate(pre_sort):
        best_ppl_pre.append(ppl_pre[idy, y[0].item()])
    print(f'best post bert is: {best_bert_post[:, 0].mean(dim=0)}')
    print(f'best post bleu is: {best_bleu_post[:, 0].mean(dim=0)}')
    print(f'best pre bert is: {torch.mean(bert_pre).item()}')
    print(f'best pre bleu is: {torch.mean(bleu_pre).item()}')
    print(f'average gt ppl is: {torch.mean(ppl_gt).item()}')
    print(f'average best pre ppl is: {np.mean(best_ppl_pre).item()}')
    print(f'average best post ppl is: {np.mean(best_ppl_post).item()}')
    print(f'average pre ppl is: {torch.mean(ppl_pre).item()}')
    print(f'average post ppl is: {torch.mean(ppl_post).item()}')
    print(count)
    print('evaluation finished.')
