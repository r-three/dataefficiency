import time
import os
import logging
import argparse
import random
import json
import torch
import numpy as np
import pandas as pd

from transformers import GenerationConfig
from torch.utils.data import DataLoader
#from pynvml import *
import pynvml
import psutil
from scripts.llama_child import Llama
from scripts.mistral_child import Mistral
from scripts.smollm_child import SmolLM
from scripts.qwen_child import Qwen
from utils.misc_utils import *
from utils.calculate_eval_metrics import *
#calculate_bleu, calculate_rouge, calculate_exact_string_match, calculate_sequence_accuracy, calculate_f1

parser = argparse.ArgumentParser(description='PyTorch roberta finetuning')

parser.add_argument('--dataset_name',
                    default="",
                    type=str,
                    help='Dataset name (glue, raft etc.)')
parser.add_argument('--task',
                    default=None,
                    type=str,
                    help='finetuning task')
parser.add_argument('--batch_size',
                    default=8,
                    type=int,
                    help='batch size for eval')
parser.add_argument('--model_name',
                    type=str,
                    help='path to load the model from')
parser.add_argument('--tokenizer_path',
                    default=None,
                    type=str,
                    help='path to tokenizer if different from model name (e.g. when loading hf experts)')
parser.add_argument('--use_quantized',
                    action="store_true",
                    help='whether to load quantized base model'
                    )
parser.add_argument('--split',
                    default="valid",
                    type=str,
                    help='valid/test split to evaluate the performance')
parser.add_argument('--max_seq_len',
                    type=int,
                    default=4096,
                    help='max sequence length for tokenizer'
                    )
parser.add_argument('--seed',
                    type=int,
                    default=123,
                    help='rand seed'
                    )
parser.add_argument('--data_size',
                    type=int,
                    default=None,
                    help='data size this model was trained on; used for logging results'
                    )
parser.add_argument('--r',
                    type=int,
                    help='rank this model was trained on; used for logging results'
                    )
parser.add_argument('--density',
                    type=float,
                    help='density spiel was trained on; used for logging results'
                    )
parser.add_argument('--result_path',
                    type=str,
                    default="",
                    help='path to log the result to one file.'
                    )
parser.add_argument('--use_flash_attention',
                    action='store_true',
                    help='whether to use flash attentionv2 for inference'
                    )
parser.add_argument('--use_safetensor',
                    action='store_true',
                    help='whether to load the model in safetensor or not'
                    )
parser.add_argument('--on_vector',
                    action="store_true"
                    )
parser.add_argument('--calculate_all_metrics',
                    action='store_true',
                    help='whether to calculate all eval metrics (bleu, rouge, seq accuracy)'
                    )


args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

log_prefix=get_model_prefix(args.model_name)

logger = logging.getLogger("LLM-eval")
curr_datetime = time.strftime("%Y%m%d%H%M%S")

logging.basicConfig(filename=f"../logs/{log_prefix}_eval_{args.task}_{args.split}_{curr_datetime}.log",
                    format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %I:%M:%S',
                    level=logging.INFO)
logger.info(f"""Initializing finetuning script with the following params:
            - model_name {args.model_name}
            - dataset_name {args.dataset_name}
            - task {args.task}
            - use_quantized: {args.use_quantized}
            - split: {args.split}
            - data_size: {args.data_size}
            """
            )


def get_dataloader(data, split, tokenizer, batch_size, mycollator, max_seq_len, assistant_start_token):
    """Get dataloader for batched evaluation
    """

    model_inputs = []
    # convert 'text' to tokens
    for i in range(len(data[split])):
        prompt = data[split][i]['text'].split(assistant_start_token)[0] + f"\n {assistant_start_token}"
        tok_dict = tokenizer(prompt, add_special_tokens=False)
        if len(tok_dict['input_ids']) < max_seq_len:
            # to retrieve grountruth label in eval_on_heldout
            tok_dict['data_idx'] = i 
            model_inputs.append(tok_dict)

    loader = DataLoader(
            model_inputs,
            collate_fn=mycollator,
            batch_size=batch_size,
            shuffle=False
        )
    return loader


def get_model_class(model_prefix, cls_params):
    """Load model specific child class
    """
    
    model_cls = None
    if model_prefix == "mistral":
        model_cls = Mistral(**cls_params)
    elif model_prefix == "llama":
        model_cls = Llama(**cls_params)
    elif model_prefix == "smollm":
        model_cls = SmolLM(**cls_params)
    elif model_prefix == "qwen":
        model_cls = Qwen(**cls_params)
    else:
        print("Not implemented!")

    return model_cls

def clean_model_output(out, assistant_start_token, eos_token):
    """Extract final model answer from model generation
    """
    
    out = out.split(assistant_start_token)[-1]
    out = out.split('####')[-1] # a few datasets (gsm8k) with #### as the final answer designator; TODO: shouldn't affect other tasks; remove?
    out = out.strip('\n').split('\n')[0] # take the very first answer before the line change
    out = out.split(eos_token)[0].strip().lower() # misc cleaning: split by end of sequence tok, remove space, lower case
    out = out.strip('.').strip() # remove '.' in case of short answers
    
    return out

def clean_target_output(tar):
    """Extract final target answer from task dataset
    """
    
    tar = str(tar)
    tar = tar.split('####')[-1]
    tar = tar.strip('.').strip().lower()

    return tar

def eval_on_heldout(model, data, split, tokenizer, dataloader, answer_len, target_key,
                    assistant_start_token, eos_token):
    """Evaluate the loaded model on the task dataset.
    """

    d_res={'target': [], 'pred_clean': [], 'pred_org': []}
    result=None

    if answer_len == "short":
        max_new_toks = 30
    elif answer_len == "long":
        max_new_toks = 500    
    
    # TODO: this generation config may have to change for non-clsasification task
    generation_config = GenerationConfig(
                                                # top_k=2,
                                                # temperature=0,
                                                max_new_tokens=max_new_toks,
                                                pad_token_id=tokenizer.pad_token_id,
                                                eos_token_id=tokenizer.eos_token_id,
                                                num_beam=4,
                                                # stop_strings=tokenizer.eos_token
                                                do_sample=False
        )

    with torch.no_grad():
        for batch in dataloader:
            batch_in = {k: v.to(device) for k, v in batch.items() if k in ['input_ids','attention_mask']}
            outputs = model.generate(**batch_in, generation_config=generation_config, tokenizer=tokenizer)
            response = tokenizer.batch_decode(outputs)
            response_clean = [clean_model_output(out=r, assistant_start_token=assistant_start_token, eos_token=eos_token) for r in response]
            target_clean = [clean_target_output(data[split][int(idx)][target_key]) for idx in batch['data_idx']]
            d_res['target'] += target_clean
            d_res['pred_clean'] += response_clean
            d_res['pred_org'] += response
    
    logger.info(f"Sample model response (raw): {response}")
    result = pd.DataFrame(d_res)
    result['target'] = result['target'].astype(str)
    result['pred_clean'] = result['pred_clean'].astype(str)

    return result


def record_result(update, path):
    """Output evaluation result to a json file in "{data_size passed by the user}: {evaluation metric}"
    """
    
    if not os.path.exists(path):
        logger.info(f"A new file {path} created")
        with open(path,'w') as f:
            dummy = {}
            json.dump(dummy, f)
            f.close()

    with open(path,'r') as f:
        res = json.load(f)
        key = list(update.keys())[0]
        if str(key) in res:
            res[str(key)] = update[key]
        else:
            res.update(update)
        f.close()

        with open(path, 'w') as f:
            json.dump(res, f, indent=4)
            f.close()


def count_adapter_param(model, peft_method):
    
    kw=None
    if peft_method=='lora':
        kw = 'lora'
    elif peft_method=='spiel':
        kw = 'sft_delta'

    param=0
    for k in model.state_dict().keys():
        if kw in k:
            param += model.state_dict()[k].flatten().shape[0]
    
    return param


def main():
    
    torch.cuda.empty_cache()

    split = args.split
    task = args.task
    model_ckpt = args.model_name
    batch_size = args.batch_size
    dataset_name = args.dataset_name
    on_vector = args.on_vector
    tokenizer_path = args.tokenizer_path
    # for logging
    data_size = args.data_size
    max_seq_len = args.max_seq_len
    r = args.r
    density = args.density
    use_flash_attention = args.use_flash_attention
    use_safetensor = args.use_safetensor
    use_quantized = args.use_quantized
    seed = args.seed
    result_path = args.result_path
    calculate_all_metrics = args.calculate_all_metrics

    model_prefix = get_model_prefix(args.model_name)

    model_cls = get_model_class(
                    model_prefix,
                    cls_params = {
                        "model_name": model_ckpt,
                        "dataset_name":dataset_name,
                        "data_size":None,
                        "task":task,
                        "use_quantized":use_quantized,
                        "use_peft":False,
                        "peft_method":None,
                        "r":None,
                        "density":None,
                        "seed":seed
                    })
    
    prompt = model_cls.load_task_prompt()
    target_key = prompt['assistant_output']
    answer_len = prompt['answer_len']
    assistant_start_token=model_cls.assistant_start_token
    eos_token=model_cls.eos_token
    
    model, tokenizer = model_cls.get_model_and_tokenizer(tokenizer_path=tokenizer_path, use_flash_attention=use_flash_attention, use_safetensors=use_safetensor, on_vector=on_vector)
    tokenizer.padding_side = 'left' # important for batch generation

    model.eval()
    model.to(device)
    print("device for evaluation: ", set([p.device for p in model.parameters()]))

    data = model_cls.get_data(tokenizer=tokenizer, max_seq_len=max_seq_len, filter_long_seq=False)

    # TODO: this is a manual logic to limit the test data to < 5000; make it flexible?
    if data['test'].num_rows > 5000:
        data['test'] = data['test'].shuffle(seed=seed).select(range(5000))
    
    # create a dataloader based on the loaded data
    mycollator=model_cls.get_datacollator(tokenizer=tokenizer, completion_only=True)
    myloader = get_dataloader(data=data, split=split, tokenizer=tokenizer, batch_size=batch_size,
                              mycollator=mycollator, max_seq_len=max_seq_len,
                              assistant_start_token=assistant_start_token
                              )
    eval_result = eval_on_heldout(model=model, data=data, split=split, tokenizer=tokenizer,answer_len=answer_len,dataloader=myloader,
                                 target_key=target_key, assistant_start_token=assistant_start_token, eos_token=eos_token
                                 )
    #eval_result.to_json(f"result_{dataset_name.split('/')[-1].lower()}.csv")
    eval_result.to_json(f"result_{dataset_name.split('/')[-1].replace('.json','').lower()}_generation_task_{data_size}.csv")
    logger.info(f"Evaluation table first five rows: {eval_result.head()}")

    mets = {}
    accuracy = calculate_exact_string_match(preds=eval_result['pred_clean'].str.replace(' ',''),
                                            refs=eval_result['target'].str.replace(' ',''))
    mets.update(accuracy)
    print(f"Exact string match accuracy after training: {accuracy}")
    logger.info(f"Exact string match accuracy after training: {accuracy}")

    if calculate_all_metrics:
        mets.update(calculate_bleu(preds=eval_result['pred_clean'], refs=eval_result['target']))
        mets.update(calculate_rouge(preds=eval_result['pred_clean'], refs=eval_result['target']))
        mets.update(calculate_sequence_accuracy(preds=eval_result['pred_clean'], refs=eval_result['target']))
        mets.update(calculate_f1(preds=eval_result['pred_clean'], refs=eval_result['target']))
        mets.update(calculate_str_contains(preds=eval_result['pred_clean'], refs=eval_result['target']))

    if result_path != "":
        d_update = {}
        # TODO: peft_method is set to None for now which only affects result json file; either remove or support
        if r:
            adapter_param_count = count_adapter_param(model, peft_method='lora')
            d_update[r] = mets
            d_update[r]['trainable_params'] = adapter_param_count
        elif density:
            adapter_param_count = count_adapter_param(model, peft_method='spiel')/2
            d_update[density] = mets
            d_update[density]['trainable_params'] = adapter_param_count
        else:
            d_update[data_size] = mets
        record_result(update=d_update, path=result_path)


if __name__=='__main__':
    main()
