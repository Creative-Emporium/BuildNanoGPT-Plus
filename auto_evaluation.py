import requests

from models.gpt2 import *
from presets import *
from utils import *
import tiktoken

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

enc = tiktoken.get_encoding("gpt2")
device = torch.device('cuda')
device_type = 'cuda'


model = GPT(model_presets['gpt2']['500m'])
checkpoint_path = './gpt2_500m_log/model_22000.pt'


model = load_model(model,checkpoint_path)

model = model.to(device)
model2 = GPT(model_presets['gpt2']['124m'])
checkpoint_path = './log/model_19072.pt'

model2 = load_model(model2,checkpoint_path)

model2 = model2.to(device)


num_return_sequences = 1
generate_max_length = 64
model_imp = 'custom'

first_prompt ='Please generate 50 pieces of texts for an LLM to continue to complete to test the completion abilities of an LLM. Please directly give the samples without numbers, each a line without any other comments.'
msg_history = []
msg_history.append({"role": "user", "content": first_prompt})

headers = {
    "Content-Type": "application/json"
}
data = {
    "mode": "instruct",
    "messages": msg_history
}
host = "http://127.0.0.1:5000/v1/chat/completions"
response = requests.post(host, headers=headers, json=data, verify=False)
answer = response.json()['choices'][0]['message']['content']

print("Answer: ", answer)
samples = answer.split('\n')
samples = [sam for sam in samples if len(sam) > 10]

win1 = 0
win2 = 0
for sam in samples:
    comp1 = completion(model, enc, sam, device, device_type, model_imp, generate_max_length, num_return_sequences,
                       greedy=True)
    comp2 = completion(model2, enc, sam, device, device_type, model_imp, generate_max_length, num_return_sequences,
                       greedy=True)
    comp1 = comp1[0]
    comp2 = comp2[0]

    prompt = f'Below are two completions from two LLMs from the prompt "{sam}". The first one is "{comp1}". The second one is "{comp2}". Please verify which one is better. Please start your answer with 1 or 2 (because I will extract your choice from the first 5 characters), followed by reason. Even if neither makes much sense, please still say which one is slightly better'
    msg_history = []
    msg_history.append({"role": "user", "content": prompt})
    data = {
        "mode": "instruct",
        "temperature": 0,
        "messages": msg_history
    }
    response = requests.post(host, headers=headers, json=data, verify=False)
    answer = response.json()['choices'][0]['message']['content']

    print("=====\nAnswer: ", answer)
    if '1' in answer[:5]:
        win1 += 1
    elif '2' in answer[:5]:
        win2 += 1
    else:
        print("answer wrong")
print(win1, win2)
