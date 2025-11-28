from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
class Chatbot:
    def __init__(self,model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.system_prompt = ''
        self.chat_history_ids = None
        self.arguments = {'do_sample':False,'max_new_tokens':32}
        self.reset_history()

    def set_arguments(self,arguments):
        self.arguments = arguments

    def add_system_prompt(self,prompt):
        self.system_prompt = prompt
        self.reset_history()


    def reset_history(self):
        self.chat_history_ids = self.encode_prompt('<|system|>\n' + self.system_prompt+'<|end|>\n')['input_ids']


    def encode_prompt(self,prompt):
        return self.tokenizer(prompt,return_tensors="pt")

    def decode_reply(self,reply_ids):
        return self.tokenizer.decode(reply_ids,skip_special_tokens=True)

    def generate_reply(self,prompt):
        prompt = '<|end|>\n' + '<|user|>\n' + prompt +'<|end|>\n' + "<|assistant|>\n"
        encoded = self.encode_prompt(prompt)
        input_ids = torch.cat((self.chat_history_ids, encoded['input_ids']),1)


        outputs = self.model.generate(
        input_ids = input_ids,
        attention_mask = torch.ones_like(input_ids),
        pad_token_id= self.tokenizer.eos_token_id,
        **self.arguments,)


        new_tokens = outputs[0][input_ids.shape[1]:]
        decoded_outputs = self.tokenizer.decode(new_tokens,skip_special_tokens=True)
        self.chat_history_ids = outputs

        return decoded_outputs


