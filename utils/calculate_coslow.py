import gc
import torch
import json
import argparse
import numpy as np
import pandas as pd
import typing as List
import torch.nn.functional as F
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, DataCollatorWithPadding

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_dataloader(data, tokenizer, batch_size=1, tokenize_data=False):
    """Get dataloader for feeding single examples to calculate metrics
    """
    
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        return_tensors="pt"
        )
    if tokenize_data:
        data = [tokenizer(ex['input_text'], add_special_tokens=False) for ex in data]

    loader = DataLoader(
            data,
            batch_size=batch_size,
            collate_fn=collator)
    
    return loader


def generate_text(model, tokenizer, batch, num_rep):
    """Prompt the model {num_rep} times to gte model response
    """

    generation_config = GenerationConfig(
                                        max_new_tokens=30,
                                        pad_token_id=tokenizer.pad_token_id,
                                        eos_token_id=tokenizer.eos_token_id,
                                        num_beam=4,
                                        do_sample=False
                                        )
    batch = {k: v.repeat(num_rep,1).to(DEVICE) for k, v in batch.items()}
    out = model.generate(**batch, generation_config=generation_config)
    del batch

    return out


def mask_non_answer_tokens(model_prefix, batch):
    """Mask additional tokens not masked by the collator; this affects the loss and gradient calculation.
    """
    
    exclude_tok_ids = None
    if model_prefix == 'mistral':
        exclude_tok_ids = [2,29473] # <\s> that does not get masked and ''
    elif model_prefix == 'llama':
        exclude_tok_ids = [271,128009] # \n\n which is at the beginning of the answer; it already excludes eos tokens but still added
    elif model_prefix == 'smollm':
        exclude_tok_ids = [0,2,198] # <|endoftext|>(pad token), <|im_end|> (eos token), \n
    elif model_prefix == 'qwen':
        exclude_tok_ids = [151643,151645,198] #<|endoftext|>, <|im_end|> eos token, \n

    exclude_tok_idxs = sum([batch['labels'] == i for i in exclude_tok_ids]).bool()
    batch['labels'][exclude_tok_idxs] = -100

    return batch


def select_examples_to_annotate(data, probas, confidence_metric, batch_size):
    """
    Based on per-sample confidence metric measure, sample batch_size many examples
    among the top-10 % low-confidence segment (either high PPL or low avg. confidence)
    """

    df_proba = pd.DataFrame(probas).T.reset_index().rename(columns={'index':'example_num'})
    if confidence_metric == 'avg_confidence':
        df_proba['rank'] = df_proba[confidence_metric].rank(pct=True, ascending=True)
    else:
        df_proba['rank'] = df_proba[confidence_metric].rank(pct=True, ascending=False)

    df_proba['rank_decile'] = pd.qcut(df_proba['rank'], q = np.arange(0,1.1,0.1), labels=False)
    idx_to_annotate = df_proba[df_proba['rank_decile']==0].sample(batch_size, replace=False)['example_num'].tolist()
    examples_to_annotate = [data[idx] for idx in idx_to_annotate]

    return examples_to_annotate


def cosine_similarity(v1, v2):
    """Input: dictionary of model layer name: gradients
       Output: cosine similarity value, dot product value
    """

    num = 0
    norm_v1 = 0
    norm_v2 = 0
    target_dim = 1
    seed=0
    for name in v1:
        if len(v1[name].shape)==1:
            continue
        if name not in v2:
            continue
        num += torch.dot(v1[name].flatten() , v2[name].flatten())
        norm_v1 += (v1[name] ** 2).sum()
        norm_v2 += (v2[name] ** 2).sum()
        seed += 1

    norm_v1 = torch.sqrt(norm_v1)
    norm_v2 = torch.sqrt(norm_v2)
    cosine_sim = float(num / (norm_v1 * norm_v2))

    return cosine_sim


def calculate_cosine_similarity(grads):
    """Iterate through example gradients (fixed by batch size) to calculate local cosine similarity among the examples
    """

    cosines = {}
    sample_keys = list(grads.keys()) 
    cosines = {}
    cosines['all'] = []

    for idx, ex_i in enumerate(sample_keys):
        cosines[str(ex_i)] = {}
        for ex_j in sample_keys[idx:]:
            if ex_i != ex_j:
                temp1 = grads[ex_i]
                temp2 = grads[ex_j]
                cos_val = cosine_similarity(temp1, temp2) 
                # store cosine
                cosines[str(ex_i)][str(ex_j)] = cos_val
                cosines['all'].append(cos_val)
    
    return np.median(cosines['all'])


def calculate_model_proba_stats(batch, output):
    """Calculate the probability assigned to the highest proba  
    """

    # 1. get the target idxs
    target_pos = (batch['labels'] != -100).nonzero()[:,1] # target token positions

    # 2. get the softmax probas at the target positions
    softmaxed_proba = F.softmax(output.logits[:, target_pos - 1, :], dim = -1).squeeze(0) # -1 to shift tokens to the left by one
    model_pred_proba, model_pred = torch.max(softmaxed_proba, dim=-1)
   
    # 3. Calculate relevant metrics
    avg_confidence = torch.mean(model_pred_proba).item()

    return {
        'model_pred_proba': model_pred_proba.tolist(),
        'model_pred': model_pred.tolist(),
        'avg_confidence': avg_confidence,
        'ppl': torch.exp(output.loss).item()
    }


def calculate_model_confidence(model, tokenizer, data, model_prefix, num_rep):
    """Iterate through data and calculate gradient related stats
    """
    
    probas = {}
    loader = get_dataloader(data=data, tokenizer=tokenizer, batch_size=1, tokenize_data=True)

    for idx, batch0 in enumerate(loader):

        # Step 1: Get model generation
        batch0 = {k: v.to(DEVICE) for k,v in batch0.items()}
        generated_ids = generate_text(model, tokenizer, batch0, num_rep)

        # Step 2: Get model confidence
        batch1 = {}
        batch1['input_ids'] = generated_ids.clone()
        prompt_len = batch0['input_ids'].shape[-1]
        labels = batch1['input_ids'].clone()
        labels[:, :prompt_len] = -100
        batch1['labels'] = labels
        batch1 = mask_non_answer_tokens(model_prefix, batch1)
        output = model(**{k: v.to(DEVICE) for k, v in batch1.items() if k in ['input_ids','labels']})
        
        # Step 3: parse model confidence stats
        per_sample_proba = calculate_model_proba_stats(batch=batch1, output=output)
        probas[idx] = per_sample_proba

        del output
        
    gc.collect()
    torch.cuda.empty_cache()

    return probas 


def compute_coslow(model_name,
                   model_prefix,
                   torch_dtype,
                   tokenizer_pad_token,
                   data
                   ):
    
    # Load the base model
    torch_dtype=torch.bfloat16 if torch_dtype=='bfloat16' else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer_pad_token
    peft_config = LoraConfig(
                    r=64,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules= ['v_proj', 'up_proj', 'k_proj', 'gate_proj', 'q_proj', 'o_proj', 'down_proj']
                )
    model = get_peft_model(model, peft_config)
    model.to(DEVICE)
    model.eval()

    # Tokenize input prompt and ground truth labels
    grads = {}
    for idx, ex in enumerate(data):
        grads[idx] = {}
        prompt_tokenized = tokenizer(ex['input_text'], add_special_tokens=False)
        prompt_len = len(prompt_tokenized['input_ids'])
        batch = tokenizer(ex['input_text'] + str(ex['ground_truth']), add_special_tokens=False, return_tensors='pt')
        labels = batch['input_ids'].clone()
        labels[:, :prompt_len] = -100
        batch['labels'] = labels
        batch = mask_non_answer_tokens(model_prefix, batch)

        output = model(**{k: v.to(DEVICE) for k,v in batch.items()})
        output.loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                grads[idx][name] = param.grad.clone().detach()
    
        # once done, empty the gradients
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        # del output
        gc.collect()
        torch.cuda.empty_cache()

    # Compute cosine similarity
    cosine_stats = calculate_cosine_similarity(grads)
    
    return cosine_stats


def compute_confidence(model_name: str,
        model_prefix: str,
        torch_dtype,
        tokenizer_pad_token: str,
        data: List,
        num_rep: int,
        confidence_metric: str,
        batch_size: int
        ):

    NUM_REP=num_rep
    BATCH_SIZE=batch_size
    
    # Load the base model
    torch_dtype=torch.bfloat16 if torch_dtype=='bfloat16' else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer_pad_token

    model.to(DEVICE)
    model.eval()

    # First, calculate the model probability to identify low-confidence examples
    per_sample_probas = calculate_model_confidence(model=model,
                                tokenizer=tokenizer,
                                data=data,
                                model_prefix=model_prefix,
                                num_rep=NUM_REP)
    # Store all probability stats
    with open(f"../results/coslow_data/task_data_per_sample_model_confidence.json", "w") as f:
        json.dump(per_sample_probas, f, indent=4)
    
    # Sample a batch of data to annotate among the top 10% of low-confidence examples
    examples_to_annotate = select_examples_to_annotate(data=data,
                                                       probas=per_sample_probas,
                                                       confidence_metric=confidence_metric,
                                                       batch_size=BATCH_SIZE)
    with open(f"../results/coslow_data/low_conf_examples_to_annotate.json", "w") as f:
        json.dump(examples_to_annotate, f, indent=4)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                    type=str,
                    help='Base model name')
    parser.add_argument('--model_prefix',
                    type=str,
                    help='Base model prefix')
    parser.add_argument('--torch_dtype',
                    type=str,
                    help='torch dtype for loading model weights')
    parser.add_argument('--tokenizer_pad_token',
                        type=str,
                        default="<|finetune_right_pad_id|>",
                        help='custom tokenizer pad token')
    parser.add_argument('--data_path',
                    type=str,
                    help='path to data json file'
                    )
    parser.add_argument('--num_rep',
                    default=None,
                    type=int,
                    help='specify if generating model output for num_rep many times'
                    )
    parser.add_argument('--confidence_metric',
                    default='avg_confidence',
                    type=str,
                    help='specify either avg_confidence or ppl to use as confidence measure'
                    )
    parser.add_argument('--batch_size',
                    default=32,
                    type=int,
                    help='specify batch size of examples to annotate'
                    )
    parser.add_argument('--compute_probability',
                    action='store_true',
                    help='use the flag if computing model confidence'
                    )
    parser.add_argument('--compute_coslow',
                    action='store_true',
                    help='use the flag if computing CoS-Low'
                    )

    args = parser.parse_args()

    with open(args.data_path, 'r') as f:
        data = json.load(f)

    assert not (args.compute_probability and args.compute_coslow), "Only one of the flags should be used"
    
    if args.compute_probability:
        compute_confidence(
            model_name=args.model_name,
            model_prefix=args.model_prefix,
            torch_dtype=args.torch_dtype,
            tokenizer_pad_token=args.tokenizer_pad_token,
            data=data,
            num_rep=args.num_rep,
            confidence_metric=args.confidence_metric,
            batch_size=args.batch_size
        )
    elif args.compute_coslow:
        coslow_value = compute_coslow(
            model_name=args.model_name,
            model_prefix=args.model_prefix,
            torch_dtype=args.torch_dtype,
            tokenizer_pad_token=args.tokenizer_pad_token,
            data=data,
        )

        print(f"CoS-Low value: {coslow_value}")
