# Credit: https://github.com/nrimsky/LM-exp/blob/main/refusal/refusal_steering.ipynb

from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights, load_checkpoint_in_model
from util.my_util import cal_js_div
from util.my_util import get_model
from util.config_for_all import parse_args
import torch
import transformers
import warnings
from tqdm import tqdm
import time
import os
from util.my_util import create_dataloaders_ppo, get_steering_vector

args = parse_args()
warnings.filterwarnings('ignore')
cuda_list = '0, 1'.split(',')
memory = '12.5GiB'
cuda_memory = {int(cuda): memory for cuda in cuda_list}
# model, _, num_of_layers, emb_dim, projection_matrix = get_model()
# projection_matrix = projection_matrix.to(torch.float)


# def generate_and_save_steering_vectors(
#     model, bank_base, bank_pre, start_layer=0, end_layer=num_of_layers, each_token=True, index=0):
#     layers = list(range(start_layer, end_layer))
#     gt_activations = dict([(layer, []) for layer in layers])
#     can_activations = dict([(layer, []) for layer in layers])
#     model.set_save_internal_decodings(False)
#     model.reset_all()


class AttnWrapper(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        self.activations = output[0]
        return output


class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block, unembed_matrix, norm, tokenizer):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm
        self.tokenizer = tokenizer

        self.block.self_attn = AttnWrapper(self.block.attn)
        # self.post_attention_layernorm = self.block.ln_2

        self.attn_out_unembedded = None
        self.intermediate_resid_unembedded = None
        self.mlp_out_unembedded = None
        self.block_out_unembedded = None

        self.activations = None
        self.add_activations = None
        self.after_position = None

        self.save_internal_decodings = False

        self.calc_dot_product_with = None
        self.dot_products = []

    # this is used when each token activation is applied
    def add_vector_after_position(self, matrix, vector, position_ids, after=None):
        after_id = after
        if after_id is None:
            after_id = position_ids.min().item() - 1
        mask = position_ids > after_id
        mask = mask.unsqueeze(-1)
        matrix += mask.float() * vector

        return matrix

    # this is used when each token activation is not applied
    # def add_vector_after_position(self, matrix, vector, position_ids, after=None):
    #     after_id = after
    #     if after_id is None:
    #         after_id = position_ids.min().item() - 1
    #     mask = position_ids > after_id
    #     mask = mask.unsqueeze(-1)
    #     matrix += mask.float() * vector
    #     return matrix

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        # print(output[0].size())
        # print(kwargs)
        position_ids = torch.tensor(list(range(output[0].size(1))), device='cuda:0')
        self.activations = output[0]
        if self.calc_dot_product_with is not None:
            last_token_activations = self.activations[0, -1, :]
            decoded_activations = self.unembed_matrix(self.norm(last_token_activations))
            top_token_id = torch.topk(decoded_activations, 1)[1][0]
            top_token = self.tokenizer.decode(top_token_id)
            dot_product = torch.dot(last_token_activations, self.calc_dot_product_with)
            self.dot_products.append((top_token, dot_product.cpu().item()))
        if self.add_activations is not None:
            augmented_output = self.add_vector_after_position(
                matrix=output[0].to("cuda:0"),  # multi-gpu
                vector=self.add_activations,
                position_ids=position_ids,
                # position_ids=torch.tensor(list(range(0, 1)), device='cuda:0'),
                after=self.after_position,
            )
            # this is used when each token activation is not applied
            # output = (augmented_output + self.add_activations,) + output[1:]

            # this is used when each token activation is applied
            output = (augmented_output + self.add_activations,) + output[1:]

        if not self.save_internal_decodings:
            return output

        # Whole block unembedded
        self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))

        # Self-attention unembedded
        attn_output = self.block.self_attn.activations
        self.attn_out_unembedded = self.unembed_matrix(self.norm(attn_output))

        # Intermediate residual unembedded
        attn_output += args[0]
        self.intermediate_resid_unembedded = self.unembed_matrix(self.norm(attn_output))

        # MLP unembedded
        # mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        mlp_output = self.block.mlp(attn_output)
        self.mlp_out_unembedded = self.unembed_matrix(self.norm(mlp_output))

        return output

    def add(self, activations):
        self.add_activations = activations

    def reset(self):
        self.add_activations = None
        self.activations = None
        self.block.self_attn.activations = None
        self.after_position = None
        self.calc_dot_product_with = None
        self.dot_products = []


class LLMHelper:
    def __init__(self):
        if args.model == "gpt-j-6b":
            config = transformers.AutoConfig.from_pretrained('./HuggingFace/gpt-j-6b', output_hidden_states=True)
            self.model = transformers.GPTJForCausalLM.from_pretrained(
                './HuggingFace/gpt-j-6b', config=config,device_map='auto')
            self.device = self.model.device
            self.tokenizer = transformers.GPT2Tokenizer.from_pretrained('./HuggingFace/gpt-j-6b')
        elif args.model == "gpt-neo-1.3b":
            config = transformers.AutoConfig.from_pretrained('./HuggingFace/gpt-neo-1.3b', output_hidden_states=True)
            self.model = transformers.AutoModelForCausalLM.from_pretrained('./HuggingFace/gpt-neo-1.3b', config=config)
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.tokenizer = transformers.GPT2Tokenizer.from_pretrained('./HuggingFace/gpt-neo-1.3b')
        elif args.model == "gpt-neo-125m":
            config = transformers.AutoConfig.from_pretrained('./HuggingFace/gpt-neo-125m', output_hidden_states=True)
            self.model = transformers.AutoModelForCausalLM.from_pretrained('./HuggingFace/gpt-neo-125m', config=config)
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.tokenizer = transformers.GPT2Tokenizer.from_pretrained('./HuggingFace/gpt-neo-125m')
        elif args.model == "gpt-neo-2.7b":
            config = transformers.AutoConfig.from_pretrained('./HuggingFace/gpt-neo-2.7b', output_hidden_states=True)
            self.model = transformers.AutoModelForCausalLM.from_pretrained('./HuggingFace/gpt-neo-2.7b', config=config)
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.tokenizer = transformers.GPT2Tokenizer.from_pretrained('./HuggingFace/gpt-neo-2.7b')
        else:
            print('no model choice!')
        self.END_STR = torch.tensor(self.tokenizer.encode("[/INST]")[1:]).to(
            self.device
        )
        # if your model is dispatched, you should not move device
        # self.model = self.model.half()
        if args.model != 'gpt-j-6b':
            self.model = self.model.to(self.device).half().eval()
        else:
            self.model = self.model.half().eval()
        torch.backends.cudnn.enabled = False
        for i, layer in enumerate(self.model.transformer.h):
            self.model.transformer.h[i] = BlockOutputWrapper(
                layer, self.model.lm_head, self.model.transformer.ln_f, self.tokenizer
            )

    def find_subtensor_position(self, tensor, sub_tensor):
        n, m = tensor.size(0), sub_tensor.size(0)
        if m > n:
            return -1
        for i in range(n - m + 1):
            if torch.equal(tensor[i : i + m], sub_tensor):
                return i
        return -1

    def find_instruction_end_postion(self, tokens, end_str):
        end_pos = self.find_subtensor_position(tokens, end_str)
        return end_pos + len(end_str) - 1

    def set_save_internal_decodings(self, value):
        for layer in self.model.transformer.h:
            layer.save_internal_decodings = value

    def generate(self, tokens):
        # instr_pos = self.find_instruction_end_postion(tokens[0], self.END_STR)
        with torch.no_grad():
            generated = self.model.generate(
                # inputs=tokens['pre_query'][:, -150:].to(self.device),
                inputs=tokens['query'][:, -50:].to(self.device),
                # inputs=tokens['input_ids'][:, :50].to(self.device),
                max_length=args.prefix_len + args.suffix_len,
                min_length=args.prefix_len + args.suffix_len,
                num_beams=1, # use beam we will get (n_beam, token_length, hidden_dim) the activation output
                do_sample=True,
                top_k=24,
                top_p=0.3,
                temperature=0.58,
                repetition_penalty=1.04,
                length_penalty=0,
                return_dict_in_generate=True,
                # num_return_sequence=3,
                # output_scores=True,
                pad_token_id=50256  # Silences warning.
            )
        # return self.tokenizer.batch_decode(generated)[0]
        # print('----We get a new generated seq after activation steering----')
        return generated.sequences.cpu()

    def get_logits(self, tokens):
        with torch.no_grad():
            logits = self.model(tokens).logits
            return logits

    def get_last_activations(self, layer):
        return self.model.transformer.h[layer].last_hidden_state

    def set_add_activations(self, layer, activations):
        self.model.transformer.h[layer].add(activations)

    def set_calc_dot_product_with(self, layer, vector):
        self.model.transformer.h[layer].calc_dot_product_with = vector

    def get_dot_products(self, layer):
        return self.model.transformer.h[layer].dot_products

    def reset_all(self):
        for layer in self.model.transformer.h:
            layer.reset()

    def get_activation_data(self, decoded_activations, topk=10):
        softmaxed = torch.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = torch.topk(softmaxed, topk)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        return list(zip(tokens, probs_percent)), list(zip(tokens, values.tolist()))


def defense_layer_selection(args, model, num_of_layers, projection_matrix):
    js_idx = torch.zeros(num_of_layers)
    selection = 0
    js_summary = torch.zeros(num_of_layers - 1)
    dataloader = create_dataloaders_ppo(args)
    for idx, seq in enumerate(tqdm(dataloader, desc='defense layer selection')):
        inputs = torch.cat((seq['query'][:, -50:], seq['response'][:, -50:]), dim=1)
        activation_base = model(inputs.to(args.device), labels=inputs.to(args.device)).hidden_states
        activation_base_ = torch.stack(activation_base).squeeze(1)[1:, :, 50:, :]
        # print(activation_base_.size())
        projection_matrix_ = projection_matrix.to(activation_base_.device).to(torch.half)
        for layer_idx in range(1, num_of_layers):
            for jj in range(args.bs):
                prev_proj = torch.matmul(activation_base_[layer_idx - 1, jj].to(
                            args.device), projection_matrix_.transpose(0, 1))
                post_proj = torch.matmul(activation_base_[layer_idx, jj].to(
                            args.device), projection_matrix_.transpose(0, 1))
                # print(abs(cal_js_div(prev_proj, post_proj).cpu().item()) * 1e5)
                js_summary[layer_idx - 1] += abs(cal_js_div(prev_proj, post_proj).cpu().item()) * 1e5
        _, js_idx = torch.sort(js_summary, descending=True)
        selection = js_idx[0].item()
    return selection, js_idx

def steering_vector_deployment(args, st_idx):
    model_steering = LLMHelper()
    model_steering.set_save_internal_decodings(False)
    model_steering.reset_all()
    steering_vector, _ = get_steering_vector(args)
    steering_vector = steering_vector.to(torch.half)

    dataloader = create_dataloaders_ppo(args)
    seq_post = torch.zeros(args.chunk, args.trail, args.prefix_len + args.suffix_len)
    model_steering.set_add_activations(st_idx, args.steering_strength * steering_vector[st_idx].to(model_steering.device))
    for ii, inputs in enumerate(tqdm(dataloader, desc='activation steering')):
        with torch.no_grad():
            for trail in range(args.trail):
                # steering generation for generation dissimilarity
                post = model_steering.generate(inputs)
                seq_post[ii * args.bs:(ii + 1) * args.bs:, trail] = post

    seq_post = seq_post.reshape(-1, args.prefix_len + args.suffix_len)
    torch.save(seq_post, f'./output/seq_post_{args.checkpoint}_{args.checkpoint + args.chunk}.pt')
