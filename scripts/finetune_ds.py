import time
import os
import gc
import logging
import argparse

import torch
import torch.distributed as dist
from transformers import set_seed, EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig
from peft import get_peft_model

from scripts.llama_child import Llama
from scripts.mistral_child import Mistral
from scripts.smollm_child import SmolLM
from scripts.qwen_child import Qwen
from utils.misc_utils import *

# add parsers to take in arguments
parser = argparse.ArgumentParser(description='PyTorch roberta finetuning')
parser.add_argument('--lr',
                    default=5e-5,
                    type=float,
                    help='learning rate')
parser.add_argument('--warmup_ratio',
                    default=0.1,
                    type=float,
                    help='warmup ratio as a percentage of the training steps'
                    )
parser.add_argument('--epoch',
                    default=1,
                    type=int,
                    help='total training epoch')
parser.add_argument('--step',
                    default=-1,
                    type=int,
                    help='total number of steps, override epoch')
parser.add_argument('--dataset_name',
                    default="",
                    type=str,
                    help='Dataset name (glue, raft etc.), either a huggingface path or local path')
parser.add_argument('--task',
                    default=None,
                    type=str,
                    help='finetuning task')
parser.add_argument('--batch_size',
                    default=8,
                    type=int,
                    help='batch size for training and eval')
parser.add_argument('--grad_accumulation_step',
                    default=1,
                    type=int,
                    help='steps to accumulate gradients on')
parser.add_argument('--model_name',
                    type=str,
                    help='model checkpoint from huggingface')
parser.add_argument('--checkpoint_dir',
                    type=str,
                    help='model output checkpoint directory')

parser.add_argument('--use_quantized',
                    action="store_true",
                    help='whether to load quantized base model'
                    )
parser.add_argument('--use_peft',
                    action="store_true",
                    help='whether to do PEFT'
                    )
parser.add_argument('--peft_method',
                    type=str,
                    help="which peft method to use"
                    )
parser.add_argument('--r',
                    type=int,
                    help='PEFT rank if used'
                    )
parser.add_argument('--density',
                    type=float,
                    help='density parameter for spiel rigl'
                    )
parser.add_argument('--ds_config',
                    type=str,
                    default="../ds_configs/zero2_config.json",
                    help='deepspeed config path relative to this script'
                    )
parser.add_argument('--max_seq_len',
                    type=int,
                    help='max sequence length for tokenizer'
                    )
parser.add_argument('--seed',
                    type=int,
                    default=123,
                    help='rand seed'
                    )
parser.add_argument('--bf16',
                    action="store_true",
                    help='argument passed to the trainer if bf16'
                    )
parser.add_argument('--data_size',
                    type=int,
                    default=None,
                    help='total number of dataset size'
                    )
parser.add_argument('--resume_from_checkpoint',
                    action="store_true",
                    help='if the trainer has halted, check for the checkpoint folder and continue from there'
                    )
parser.add_argument('--use_flash_attention',
                    action="store_true",
                    help='whether to use flash attention or not'
                    )
parser.add_argument('--log_and_save_step',
                    type=int,
                    default=10,
                    help="evaluate run results and save every n steps"
                    )
parser.add_argument('--on_vector',
                    action="store_true",
                    help="in vector cluster I don't have enough disc quota to store model weights."
                    )
parser.add_argument('--subset_sample_path',
                    type=str,
                    default=None,
                    help="only train the model on specifically designated subset indexable by the unique task id(default now is incorrect examples)"
                    )

args = parser.parse_args()

logger = logging.getLogger("LLM-FT")
curr_datetime = time.strftime("%Y%m%d%H%M%S")

log_prefix=get_model_prefix(args.model_name)

logging.basicConfig(filename=f"../logs/{log_prefix}_finetune_{args.task}_epoch{args.epoch}_r{args.r}_datasize_{args.data_size}.log",
                    format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %I:%M:%S',
                    level=logging.INFO)
logger.info(f"""Initializing finetuning script for {args.model_name} with the following params:
            - lr {args.lr}
            - epoch {args.epoch}
            - dataset_name {args.dataset_name}
            - task {args.task}
            - batch_size {args.batch_size}
            - grad_accumulation_step {args.grad_accumulation_step}
            - use_quantized: {args.use_quantized}
            - use_peft: {args.use_peft}
            - r (for rank): {args.r}
            - checkpoint_dir: {args.checkpoint_dir}
            - ds_config: {args.ds_config}
            - bf16: {args.bf16}
            - data_size: {args.data_size}
            """
            )


def count_trainable_param(m):
    
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def get_training_arguments(model_prefix):
    
    unique_task_id = args.task if args.task else args.dataset_name.split('/')[-1]
    unique_task_id = unique_task_id.lower().replace('-','_')

    if args.peft_method == 'spiel':
        run_name = f"{model_prefix}_finetune_{unique_task_id}_d{args.density}_data_size{args.data_size}"
    elif args.peft_method == 'lora':
        run_name = f"{model_prefix}_finetune_{unique_task_id}_r{args.r}_lr{args.lr}_data_size{args.data_size}"
    else:
        run_name = f"{model_prefix}_finetune_{unique_task_id}_full_data_size{args.data_size}"

    if args.subset_sample_path:
        run_name += "_conf0_subset"

    training_arguments = SFTConfig(
        # save dir
        output_dir=f"{args.checkpoint_dir}/{run_name}",
        do_eval=True,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=args.log_and_save_step,
        # save_only_model=True, # do not save optimizer state etc, takes up space; but can't use with load_best_model_at_end
        logging_steps=args.log_and_save_step, # same as eval_steps
        save_total_limit=1, # only save best one + last checkpoint
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=run_name, 
        # steps
        num_train_epochs=args.epoch,
        max_steps=args.step,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size= args.batch_size,
        # grad
        gradient_accumulation_steps = args.grad_accumulation_step,
        # lr
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        # optimizer
        #optim="adamw_bnb_8bit", jje: commented out because it was causing failure when doing full model ft
        # warmup
        warmup_ratio = args.warmup_ratio,
        # bf16
        bf16=args.bf16, #jje: fp32 + bf16 is OOM for fmft... commented this out because mixed precision + fp36 was OOM for fmft
        #dataset_text_field="text",
        #max_seq_length=args.max_seq_len
        )

    return training_arguments

def get_model_class(model_prefix, cls_params):
    
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
    

def main():

    gc.collect()    
    torch.cuda.empty_cache()

    get_mem_usage_stats(f"init memory in the beginning")
    args.task = None if args.task == 'None' else args.task    
    seed = args.seed
    model_name = args.model_name
    dataset_name = args.dataset_name
    task = args.task
    use_quantized = args.use_quantized
    use_peft = args.use_peft
    peft_method = args.peft_method
    r = args.r
    density = args.density
    max_seq_len = args.max_seq_len
    data_size = args.data_size
    resume_from_checkpoint = args.resume_from_checkpoint
    use_flash_attention = args.use_flash_attention
    on_vector = args.on_vector # jje: for ease of training on vector cluster..
    subset_sample_path = args.subset_sample_path
    model_prefix = get_model_prefix(args.model_name)
    training_arguments = get_training_arguments(model_prefix=model_prefix)
    # add gradient checkpointing arguments
    training_arguments.gradient_checkpointing_kwargs = {
            "use_reentrant": True
    }

    # setting seed and adding CUDA related dependencies
    set_seed(seed)

    # init model class that has initialization methods
    model_cls = get_model_class(
                    model_prefix=model_prefix,
                    cls_params = {
                        "model_name": model_name,
                        "dataset_name":dataset_name,
                        "data_size":data_size,
                        "task":task,
                        "use_quantized":use_quantized,
                        "use_peft":use_peft,
                        "peft_method":peft_method,
                        "r":r,
                        "density":density,
                        "seed":seed
                    })

    model, tokenizer = model_cls.get_model_and_tokenizer(use_flash_attention=use_flash_attention, on_vector=on_vector)
    data = model_cls.get_data(tokenizer=tokenizer, max_seq_len=max_seq_len, subset_sample_path=subset_sample_path)
    logger.info(f"train count: {data['train'].num_rows}, valid count: {data['valid'].num_rows}, test count: {data['test'].num_rows}")
    
    mycollator=model_cls.get_datacollator(tokenizer=tokenizer, completion_only=True)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=10)]

    org_count = count_trainable_param(model)
    print(f"Original trainable param: {org_count}")
    logger.info(f"Original trainable parameter count: {org_count}")

    get_mem_usage_stats(f"after loading model, tokenizer, and data")

    if use_peft:
        peft_config = model_cls.get_peft_config()
        model = get_peft_model(model, peft_config)
        post_peft_count = count_trainable_param(model)
        logger.info(f"Trainable parameter count after PEFT set up: {post_peft_count}")

        if peft_method == 'lora':
            trainer = SFTTrainer(
                model=model,
                train_dataset=data['train'],
                eval_dataset=data['valid'],
                peft_config=peft_config,
                dataset_text_field="text",
                args=training_arguments,
                #processing_class=tokenizer,
                tokenizer=tokenizer,
                max_seq_length=max_seq_len,
                data_collator=mycollator,
                callbacks=callbacks,
            )
        elif peft_method == 'spiel':
            from peft import SftTrainer
            trainer = SftTrainer(SFTTrainer)
            trainer = trainer(
                model=model,
                train_dataset=data['train'],
                eval_dataset=data['valid'],
                sft_config=peft_config, # jje: this lineis different
                dataset_text_field="text",
                args=training_arguments,
                #processing_class=tokenizer,
                tokenizer=tokenizer,
                max_seq_length=max_seq_len,
                data_collator=mycollator,
                callbacks=callbacks,
            )

        # handle PEFT + FSDP
        trainer.model.print_trainable_parameters()
        if getattr(trainer.accelerator.state, "fsdp_plugin", None):
            from peft.utils.other import fsdp_auto_wrap_policy
            logger.info("fsdp auto wrap policy to handle PEFT model")
            fsdp_plugin = trainer.accelerator.state.fsdp_plugin
            fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

    else:
        trainer = SFTTrainer(
            model=model,
            train_dataset=data['train'],
            eval_dataset=data['valid'],
            dataset_text_field="text",
            args=training_arguments,
            #processing_class=tokenizer,
            tokenizer=tokenizer,
            data_collator=mycollator,
            max_seq_length=max_seq_len,
            callbacks=callbacks,
        )
    
    get_mem_usage_stats(f"{trainer.accelerator.local_process_index} - before train init")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint) #resume_from_checkpoint=resume_from_checkpoint
    get_mem_usage_stats(f"{trainer.accelerator.local_process_index} - After training")

    if trainer.is_fsdp_enabled:
        logger.info("is_fsdp_enabled is true, will save the model")
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()


if __name__=='__main__':

    main()
