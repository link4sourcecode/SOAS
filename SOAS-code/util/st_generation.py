import torch
from tqdm import tqdm
import warnings
import os
from util.config_for_all import parse_args
from util.my_util import create_dataloaders_ppo, get_model, orthogonal_decom, get_steering_vector

warnings.filterwarnings('ignore')
args = parse_args()
model, tokenizer, num_of_layers, emb_dim, _ = get_model()

def steering_vector_generation(args, model, tokenizer, num_of_layers, emb_dim):
    # loading args
    seq_len = args.prefix_len + args.suffix_len
    num_of_trials = args.trail
    dataloader = create_dataloaders_ppo(args)

    # prepare empty storage
    gt = torch.zeros(args.chunk, seq_len)
    sequence_pre = torch.zeros(num_of_trials * args.chunk, seq_len)
    steering_direction = torch.empty(0)

    # loading model
    tokenizer.pad_token = tokenizer.eos_token

    # padding
    pad = tokenizer.encode(tokenizer.pad_token)[0]
    pad_matrix = torch.full([args.bs, 1], pad)

    for idx, seq in enumerate(tqdm(dataloader, desc='generating candidates')):
        sequence_gt = torch.cat((seq['query'][:, -50:], seq['response'][:, -50:]), dim=1)
        # sequence_gt = torch.cat((seq['pre_query'][:, -150:], seq['response'][:, -50:]), dim=1)
        gt[args.bs * idx:args.bs * (idx + 1)] = sequence_gt
        activation_gt = torch.zeros(args.bs * args.trail, num_of_layers, seq_len, emb_dim)
        activation_pre = torch.zeros(args.bs * args.trail, num_of_layers, seq_len, emb_dim)
        for i in range(num_of_trials):
            # inputs = seq['pre_query'][:, -150:]
            inputs = seq['query'][:, -50:]
            with torch.no_grad():
                generation = model.generate(
                    # input truncation for collecting generation with similar semantic while distinct structure
                    # inputs[:150 - i * 30].to(args.device),
                    inputs[:50 - i * 10].to(args.device),
                    max_length=seq_len,
                    min_length=seq_len,
                    num_beams=1,
                    do_sample=True,
                    top_k=24,
                    top_p=0.3,
                    temperature=0.58,
                    repetition_penalty=1.04,
                    length_penalty=0,
                    return_dict_in_generate=True,
                    pad_token_id=50256
                )
                # get sequences and activations(left padded)
                candidate = generation.sequences.cpu()
                pad_candidate = torch.cat((pad_matrix, candidate), dim=1)
                pad_gt = torch.cat((pad_matrix, sequence_gt), dim=1)
                act_pre = model(
                    pad_candidate.to(args.device), labels=pad_candidate.to(args.device))
                act_gt = model(
                    pad_gt.to(args.device), labels=pad_gt.to(args.device))
                # storage
                for t in range(args.bs):
                    sequence_pre[i + num_of_trials * (t + args.bs * idx)] = candidate[t]
                    for layer_idx in range(num_of_layers):
                        activation_gt[
                        i + num_of_trials * t, layer_idx, :] = act_gt.hidden_states[layer_idx + 1][t, :-1]
                        activation_pre[
                        i + num_of_trials * t, layer_idx, :] = act_pre.hidden_states[layer_idx + 1][t, :-1]
        if torch.equal(activation_gt, activation_pre) is not True:
            # deriving steering vector from activation differences
            sub = (activation_pre - activation_gt).mean(dim=0).mean(dim=1).unsqueeze(0)
            steering_direction = torch.cat((steering_direction, sub), dim=0)
        del activation_gt, activation_pre
    root = f'./output'
    if not os.path.exists(root):
        os.makedirs(root)
    torch.save(
        gt.to(torch.long), f'./output/seq_gt_{args.checkpoint}_{args.checkpoint + args.chunk}.pt')
    torch.save(sequence_pre.to(
        torch.long), f'./output/seq_pre_{args.checkpoint}_{args.checkpoint + args.chunk}_{args.model}.pt')
    print(f'\n-----finishing generating {steering_direction.size(0)} candidates and recording activations-----')
    st = steering_direction.mean(dim=0)
    _, root = get_steering_vector(args)
    torch.save(st, root)
    return steering_direction.mean(dim=0)




