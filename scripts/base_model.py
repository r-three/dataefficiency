import os
import json
import yaml
from pynvml import *
import torch
from abc import ABC, abstractmethod
from peft import LoraConfig
from trl import DataCollatorForCompletionOnlyLM
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorWithPadding, DataCollatorForSeq2Seq

class BaseModel(ABC):

    def __init__(self, dataset_name, data_size, task, use_quantized, use_peft, peft_method, r, density, model_name, seed):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.data_size = data_size
        self.task = task if task != 'None' else None
        self.seed = seed
        self.use_peft = use_peft
        self.peft_method = peft_method
        self.use_quantized = use_quantized
        self.task_prompt = self.load_task_prompt()
        self.peft_config = None
        self.bnb_config = None

        if use_peft:
            if peft_method == 'lora':
                self.peft_config = LoraConfig(
                    r=r,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules= ['v_proj', 'up_proj', 'k_proj', 'gate_proj', 'q_proj', 'o_proj', 'down_proj'] #['q_proj']
                )
            elif peft_method == 'spiel':
                from peft import SftConfig, TaskType
                self.peft_config = SftConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    reselection_steps=40,
                    density=density, # test 1e-7, 1e-6, 1e-5, 1e-4, 1e-3
                    selection_algorithm="rigl", # or "sm3" for moment approximation SFT
                    target_modules=["q_proj", "v_proj"],
                )

        if use_quantized:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_storage=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )

    
    def get_datacollator(self, tokenizer, use_flash_attention=False, completion_only = True):

        padding_free = False
        if completion_only:
            if use_flash_attention:
                padding_free = True
            return DataCollatorForCompletionOnlyLM(response_template=self.assistant_start_token, tokenizer=tokenizer, padding_free=padding_free)
        else:
            return DataCollatorForSeq2Seq(tokenizer=tokenizer)


    def get_peft_config(self):
        
        return self.peft_config
    
    
    def get_model_and_tokenizer(self, tokenizer_path=None, use_flash_attention=True, use_safetensors=False, load_dtype=None, on_vector=False):
        
        if on_vector:
            args = {'pretrained_model_name_or_path': "/model-weights/Meta-Llama-3.1-8B-Instruct" 
                }
        else:
            args = {'pretrained_model_name_or_path': self.model_name}

        if use_flash_attention:
            args['attn_implementation'] = "flash_attention_2"
        
        if use_safetensors:
            args['use_safetensors'] = use_safetensors

        # get the model
        if self.use_quantized:
            args['quantization_config'] = self.bnb_config
            args['torch_dtype'] = torch.bfloat16 if not load_dtype else load_dtype #torch.bfloat16    
        else:
            args['torch_dtype'] = torch.bfloat16 if not load_dtype else load_dtype
            #jje: temp fix for spiel support
            #args['torch_dtype'] = torch.float32 if self.peft_method=='spiel' else args['torch_dtype']
        
        print('using args: ', args)
        model = AutoModelForCausalLM.from_pretrained(**args)
        # get the tokenizer
        if tokenizer_path == None or tokenizer_path == "":
            tokenizer_path = self.model_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        if tokenizer.pad_token != self.pad_token:
            tokenizer.pad_token_id = tokenizer(self.pad_token, add_special_tokens=False)['input_ids'][0]
            print(f"Padding tokenizer pad token to {tokenizer.pad_token_id}")
        
        return model, tokenizer

    
    def get_data(self, tokenizer, max_seq_len,
                 subset_sample_path=None, filter_long_seq = True, load_from_cache_file = False):
        
        data_split = list(self.task_prompt['split'].values()) if 'split' in self.task_prompt else None
        data_split_keys = list(self.task_prompt['split'].keys()) if 'split' in self.task_prompt else None
        subset = self.task_prompt['subset'].split(' ') if 'subset' in self.task_prompt else None

        # multiple subtasks should be merged if specified in the prompts yaml file
        if subset is not None:
            ds = {}
            # merge the datasets by their splits
            for sub in subset:
                temp = load_dataset(self.dataset_name, sub, split=data_split, trust_remote_code=True)
                temp_splits = data_split_keys if data_split_keys is not None else list(temp.keys()) # train, valid(ation), test split across subtasks
                for idx, spl in enumerate(temp_splits):
                    if spl not in ds:
                        ds[spl] = []
                    split_key = idx if data_split_keys is not None else spl # standardize data splits as "train, valid, test"
                    temp[split_key] = temp[split_key].add_column("category",[sub] * len(temp[split_key]))
                    ds[spl].append(temp[split_key])
            data = DatasetDict({k: concatenate_datasets(ds[k]) for k in ds})
            del ds
        elif 'json' in self.dataset_name: # loading from local disk
            from pathlib import Path
            home_dir = str(Path.home())
            data = load_dataset('json', data_files=f"{home_dir}/ft-intrinsic-dim/{self.dataset_name}", split=data_split)
        else: # default way of loading data, usually if only one subtask
            data = load_dataset(self.dataset_name, self.task, split=data_split, trust_remote_code=True)
            # standardize data splits as "train, valid, test"
            if data_split is not None:
                ds = {k: [] for k in data_split_keys}
                for idx, k in enumerate(data_split_keys):
                    ds[k].append(data[idx])
                data = DatasetDict({k: concatenate_datasets(ds[k]) for k in ds})
                del ds

        # if valid wrong name
        if 'validation' in data:
            data['valid'] = data.pop("validation")
        # if no validation provided at all
        elif 'validation' not in data and 'valid' not in data:
            # if test not in data, train:valid:test should have 8:1:1 ratio
            train_split = data['train'].train_test_split(0.2, seed=self.seed)
            data['train'] = train_split.pop('train')
            if 'test' not in data:
                valid_test_split = train_split['test'].train_test_split(0.5, seed=self.seed) # halve the valid,test
                data['valid'] = valid_test_split.pop('train')
                data['test'] = valid_test_split.pop('test')
            else:
                data['valid'] = train_split.pop('test')
        # if test missing ## train, valid existed and only test was missing; then just take 10% of train
        if 'valid' in data and 'test' not in data:
            test_split = data['train'].train_test_split(0.1, seed=self.seed)
            data['train'] = test_split.pop('train')
            data['test'] = test_split.pop('test')

        # limit the training data size
        if self.data_size:
            data['train'] = data['train'].shuffle(seed=self.seed).select(range(self.data_size))
        
        # scale down valid dataset size if greater than 20% of the train dataset
        if data['valid'].num_rows > data['train'].num_rows * 0.2:
            data['valid'] = data['valid'].shuffle(seed=self.seed).select(range(int(data['train'].num_rows * 0.2)))

        data = data.map(lambda x: {"text": tokenizer.apply_chat_template(
                                                                    self.format_chat_template(row=x),
                                                                    tokenize=False,
                                                                    add_special_token=False,
                                                                    add_generation_prompt=False)},
                                                                    load_from_cache_file=load_from_cache_file,
                                                                    #num_proc=os.cpu_count() - 1
                                                                    )
        ## TODO: to filter out long sequence data to avoid NaN loss during training. can handle more efficiently?
        if filter_long_seq:
            print(f"(pre-filtering) Data train size : {data['train'].num_rows} Validation: {data['valid'].num_rows} Test: {data['test'].num_rows}")
            data = data.map(lambda x: tokenizer(x['text'], add_special_tokens=False))
            data = data.filter(lambda x: len(x['input_ids']) < max_seq_len)
            data = data.remove_columns(['input_ids','attention_mask'])

        # if only a specific subset of train data is used (using an index file)
        if subset_sample_path:
            with open(subset_sample_path, "r") as f:
                tmp = json.load(f)
            unique_task_id = self.task if self.task else self.dataset_name.split('/')[-1]
            unique_task_id = unique_task_id.replace('-','_').lower()
            example_subset = tmp[unique_task_id]['example_num']
            data['train'] = data['train'].select(example_subset)
            data['train'] = data['train'].shuffle(seed=self.seed) # shuffle one more time because a specific ordering may cause an issue 

        print(f"Data train size : {data['train'].num_rows} Validation: {data['valid'].num_rows} Test: {data['test'].num_rows}")

        return data
    
    def load_task_prompt(self):
        with open('../prompts/prompts_by_task_modified.yaml','r') as f:
            prompt = yaml.safe_load(f)

        return prompt[self.dataset_name][self.task]


    def parse_data_split(self):

        if 'split' in self.task_prompt:
            return list(self.task_prompt['split'].values())
        else:
            return None

    @abstractmethod
    def format_chat_template(self, row):
        """Takes input from create_prompt but implemented at child class
        """

    def create_prompt(self, row):

        instruction = self.task_prompt['instruction']
        usr_input = self.task_prompt['user_input']
        assistant_output = self.task_prompt['assistant_output']

        # when there are multiple user inputs
        if isinstance(usr_input, dict):
            s=""
            for k in usr_input:
                if isinstance(usr_input[k], dict):
                    s += "\n"+ k + " - "
                    data_key = usr_input[k]['key_name']
                    data_field = row[data_key] # e.g. choices
                    fields = usr_input[k]['fields'] # e.g. subfield of choices; labels, description
                    concat_symbol = usr_input[k]['concat_symbol'] # how to concat the fields
                    
                    if concat_symbol:
                        s += "\n"
                        for pair in zip(data_field[fields[0]], data_field[fields[1]]):
                            s += pair[0] + " " + concat_symbol + " " + pair[1] + "\n"
                    else:
                        s += data_field[fields[0]]
                else:
                    s += k + " - " + str(row[usr_input[k]])+ " [SEP] \n"

            usr_query = s.strip().strip("\n")[:-len("[SEP]")].strip()
        else:
            usr_query = row[usr_input]
        
        return instruction, usr_query, str(row[assistant_output])
