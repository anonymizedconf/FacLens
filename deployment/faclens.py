import random
import torch
from networks.model import MLP
import os
import numpy as np
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    return

class FacLens():
    def __init__(self) -> None:
        self.hidden_dim = 256
        setup_seed(123)
        self.device = "cuda:0"

        #---------------------------------------- LLaMA -----------------------------------
        self.llama_classifier = MLP(4096, self.hidden_dim, 2).to(self.device)
        # load the ckpt
        self.llama_classifier.load_state_dict(torch.load('./ckpt/classifier_checkpoint_llama.pth', map_location="cpu")['model_state_dict'])
        self.llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype = torch.float16).to(self.device)
        self.llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        
        self.llama_classifier.eval()
        self.llama_model.eval()
        
        #---------------------------------------- Mistral -----------------------------------
        self.mistral_classifier = MLP(4096, self.hidden_dim, 2).to(self.device)
        # load the ckpt
        self.mistral_classifier.load_state_dict(torch.load('./ckpt/classifier_checkpoint_mistral.pth', map_location="cpu")['model_state_dict'])
        self.mistral_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", torch_dtype = torch.float16).to(self.device)
        self.mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

        self.mistral_classifier.eval()
        self.mistral_model.eval()

        #---------------------------------------- LLaMA-3 -----------------------------------
        self.llama3_classifier = MLP(4096, self.hidden_dim, 2).to(self.device)
        # load the ckpt
        self.llama3_classifier.load_state_dict(torch.load('./ckpt/classifier_checkpoint_llama3.pth', map_location="cpu")['model_state_dict'])
        self.llama3_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype = torch.float16).to(self.device)
        self.llama3_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

        self.llama3_classifier.eval()
        self.llama3_model.eval()
    
    def infer_fact(self, question, model_name):
        t0 = time.time()
        # get the hidden states
        messages = [{"role": "user", "content": question}]

        if model_name == "LLaMA2-7B-Chat":
            classifier = self.llama_classifier
            model = self.llama_model
            tokenizer = self.llama_tokenizer
            pos_th = 0.8175499
            neg_th = 0.56329066
        elif model_name == "Mistral-7B-Instruct-v0":
            classifier = self.mistral_classifier
            model = self.mistral_model
            tokenizer = self.mistral_tokenizer
            pos_th = 0.8045934
            neg_th = 0.495729
        elif model_name == "Meta-Llama-3-8B-Instruct":
            classifier = self.llama3_classifier
            model = self.llama3_model
            tokenizer = self.llama3_tokenizer
            pos_th = 0.66621387
            neg_th = 0.37309915

        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        input_ids = encodeds.to(self.device)
        outputs = model(input_ids, output_hidden_states=True)
        print("layer idx:", int(model.config.num_hidden_layers/2))
        input_data = outputs.hidden_states[int(model.config.num_hidden_layers/2)][0][-1].detach().cpu().numpy().reshape(1,-1)
        input_data = torch.Tensor(input_data).to(self.device)
        
        with torch.no_grad():
            output = classifier(input_data)[1].detach()
            prob = F.softmax(output, dim=1)

        print("question:", question)
        if prob[0][1] >= pos_th:
            decision = 1
            print("The LLM doesn't know the factual answer 😭")
        elif prob[0][1] <= neg_th:
            decision = 0
            print("The LLM knows the factual answer 😊")
        else:
            decision = 0.5
            print("I am not sure if the LLM knows the factual answer 😂")

        t1 = time.time()
        print("Infer time: ", t1 - t0, "s")
        print("=="*25+"\n\n\n")

        return decision
    
    def infer_answer(self, question, model_name):
        messages = [{"role": "user", "content": question}]

        if model_name == "LLaMA2-7B-Chat":
            model = self.llama_model
            tokenizer = self.llama_tokenizer
        elif model_name == "Mistral-7B-Instruct-v0":
            model = self.mistral_model
            tokenizer = self.mistral_tokenizer
        elif model_name == "Meta-Llama-3-8B-Instruct":
            model = self.llama3_model
            tokenizer = self.llama3_tokenizer

        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        input_len = encodeds.shape[1]
        input_ids = encodeds.to(self.device)
        #----------------------------------------show the output of the LLM-----------------------------------
        generated_ids = model.generate(input_ids, max_new_tokens=256, do_sample=False, num_beams=1)
        decoded = tokenizer.decode(generated_ids[0, input_len:], skip_special_tokens = True)
        
        if model_name == "Meta-Llama-3-8B-Instruct":
            decoded = decoded[len("assistant"):].strip()
        
        print("question:",  question)
        print("response:", decoded)

        return decoded
