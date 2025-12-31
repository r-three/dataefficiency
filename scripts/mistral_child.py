#from pynvml import *
import pynvml
from transformers import DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForCausalLM
from trl import DataCollatorForCompletionOnlyLM
from scripts.base_model import BaseModel
import torch

class Mistral(BaseModel):
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)

        self.assistant_start_token = "[/INST]"
        self.eos_token = "</s>"
        self.pad_token = "<unk>"


    def format_chat_template(self, row):

        instruction, usr_query, assistant_output = self.create_prompt(row)

        row_json = [{"role": "user", "content": instruction + " \n" + usr_query},
                {"role": "assistant", "content": assistant_output}]

        return row_json
    
