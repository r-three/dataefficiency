#from pynvml import *
import pynvml
from transformers import DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForCausalLM
from trl import DataCollatorForCompletionOnlyLM
from scripts.base_model import BaseModel
import torch

class Llama(BaseModel):
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)

        self.assistant_start_token = "<|start_header_id|>assistant<|end_header_id|>"
        self.eos_token = "<|eot_id|>"
        self.pad_token = "<|finetune_right_pad_id|>"#"<|eot_id|>"

    
    def format_chat_template(self, row):

        instruction, usr_query, assistant_output = self.create_prompt(row)

        row_json = [{"role": "system", "content": instruction},
                {"role": "user", "content": usr_query},
                {"role": "assistant", "content": assistant_output}]

        return row_json
