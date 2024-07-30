import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
import logging
import datetime
import time

# Setup logging
logging.basicConfig(filename='train_log.txt', level=logging.INFO)

# Load the model and tokenizer
model_path = 'DrNicefellow/Nano-GPT2-500m-29k_steps'
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

tokenizer.pad_token = tokenizer.eos_token

# Prepare the data
dataset = load_dataset("DrNicefellow/CHAT-ALL-IN-ONE-v1", split="train")

def tokenize_function(examples):
    inputs = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=1024, return_tensors="pt")
    inputs['labels'] = inputs.input_ids.detach().clone()
    return inputs
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
train_dataset = tokenized_datasets.with_format('torch')
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
train_dataloader = DataLoader(tokenized_datasets, batch_size=16, shuffle=True, collate_fn=data_collator)

# Move model to GPU and cast to bfloat16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)#.to(torch.bfloat16)  # Cast model to bfloat16

# Setup optimizer with offload to CPU
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
num_epochs = 1
logging_steps = 1
save_steps = 500
accumulation_steps = 16
output_dir = "./results"
model.train()

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    time_taken = 0
    for step, batch in enumerate(train_dataloader):
        # Cast batch to bfloat16 and move to device
        t2 = time.time()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss / accumulation_steps
        loss.backward()
        time_taken += time.time() - t2

        if (step + 1) % accumulation_steps == 0:
            t1 = time.time()
            optimizer.step()
            optimizer.zero_grad()
            print("Optimizer step done at time: ", datetime.datetime.now(), 'Time taken: ', time.time() - t1)

        if step % logging_steps == 0:
            print(f"Step {step}: Loss {loss.item()} Time taken: {time_taken}")
            logging.info(f"Epoch: {epoch + 1}, Step: {step}, Loss: {loss.item()} Time taken: {time_taken}")
            time_taken = 0

        if step % save_steps == 0:
            model.save_pretrained(f"{output_dir}/model_{epoch + 1}_{step}")

model.save_pretrained("./final")
tokenizer.save_pretrained("./final")
