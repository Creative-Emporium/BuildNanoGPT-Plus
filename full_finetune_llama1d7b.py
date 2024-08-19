



import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
import logging
import datetime
import time
from tqdm import tqdm
import wandb
import os
from datasets import load_from_disk

max_seq_length = 1024
batch_size = 2

# Setup logging
logging.basicConfig(filename='train_log.txt', level=logging.INFO)

# Load the model and tokenizer
model_path = '/src/nano-llama-9000_steps'
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

alpaca_prompt = """
### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }


d_p = "./code_bagel_preprocessed_dataset"
if os.path.exists(d_p):
    tokenized_datasets = load_from_disk(d_p)
else:
    from datasets import load_dataset
    dataset = load_dataset("mhenrichsen/alpaca_2k_test", split = "train")
    dataset = dataset.map(formatting_prompts_func, batched = True, num_proc=32)
    def tokenize_function(examples):
        inputs = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=max_seq_length, return_tensors="pt")
        inputs['labels'] = inputs.input_ids.detach().clone()
        return inputs

    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=32, remove_columns=['text', 'instruction', 'input', 'output'])
    tokenized_datasets = tokenized_datasets.with_format('torch')
    tokenized_datasets.save_to_disk(d_p)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

train_dataloader = DataLoader(tokenized_datasets, batch_size=batch_size, shuffle=True, collate_fn=data_collator)

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
# Move model to GPU and cast to bfloat16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)#.to(torch.bfloat16)  # Cast model to bfloat16

# Setup optimizer with offload to CPU
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cpu()

project_name = "finetune_alpaca"
wandb.init(project=project_name)

# Training loop
num_epochs = 1
logging_steps = 1
save_steps = 1000
accumulation_steps = 4
output_dir = "./results"
model.train()

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    time_taken = 0
    progress_bar = tqdm(train_dataloader, desc="Training", total=len(train_dataloader))
    for step, batch in enumerate(progress_bar):
        # Cast batch to bfloat16 and move to device
        t2 = time.time()
        batch = {k: v.to(device) for k, v in batch.items()}  # Cast inputs to bfloat16
        outputs = model(**batch)
        loss = outputs.loss / accumulation_steps
        loss.backward()
        time_taken += time.time() - t2

        if (step + 1) % accumulation_steps == 0:
            t1 = time.time()
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data = param.grad.data.cpu()
                    param.data = param.data.cpu()
            optimizer.step()
            optimizer.zero_grad()
            for param in model.parameters():
                if device == torch.device('cuda'):
                    param.data = param.data.to(device)
            optimizer.zero_grad(set_to_none=True)
            print("Optimizer step done at time: ", datetime.datetime.now(), 'Time taken: ', time.time() - t1)

        if step % logging_steps == 0:
            print(f"Step {step}: Loss {loss.item()} Time taken: {time_taken}")
            logging.info(f"Epoch: {epoch + 1}, Step: {step}, Loss: {loss.item()} Time taken: {time_taken}")
            wandb.log({"loss": loss.item()})
            time_taken = 0

        if step % save_steps == 0:
            model.save_pretrained(f"{output_dir}/model_{epoch + 1}_{step}")

        # Update the progress bar
        progress_bar.set_postfix({'loss': loss.item()}, refresh=True)

    progress_bar.close()

model.save_pretrained("./final")
tokenizer.save_pretrained("./final")


# Push the model to the Hugging Face Hub
model.push_to_hub("your-model-name")

# Push the tokenizer to the Hugging Face Hub
tokenizer.push_to_hub("your-model-name")