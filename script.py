from transformers import AutoModelForCausalLM, AutoTokenizer
import accelerate, bitsandbytes
import torch, transformers 

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=False,
    load_in_8bit= True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)


class Model:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained('kibru/llama2-lyric-completer',
                                             torch_dtype=torch.bfloat16,
                                             device_map='auto' , 
                                             quantization_config=bnb_config)
                                                 
        self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
        self.tokenizer.pad_token = self.tokenizer.eos_token 
    def generate(self, text):
        ins = self.tokenizer (text, return_tensors='pt', padding=True)
        with torch.no_grad():
            out = self.model.generate( **ins.to('cuda'),
                max_new_tokens=512,
                do_sample=False,
                temperature=0.5,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=2.,)
        return self.tokenizer.decode(out.squeeze())