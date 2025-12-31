from scripts.base_model import BaseModel

class Qwen(BaseModel):
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)

        self.assistant_start_token = "<|im_start|>assistant"
        self.eos_token = "<|im_end|>"
        self.pad_token = "<|endoftext|>"

    
    def format_chat_template(self, row):

        instruction, usr_query, assistant_output = self.create_prompt(row)

        row_json = [{"role": "system", "content": instruction},
                {"role": "user", "content": usr_query},
                {"role": "assistant", "content": assistant_output}]

        return row_json
