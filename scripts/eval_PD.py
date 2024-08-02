import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from math import ceil

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate perplexity scores for model responses.")
    parser.add_argument('--model_name', type=str, default='GPT-2', choices=['Phi3', 'GPT2'], help='Keyword for selecting the model.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSONL file.')
    parser.add_argument('--partial_answers_file', type=str, default='test', choices=['test', 'dev'], help='Specify "test" or "dev" to select the partial answers file.')
    return parser.parse_args()

def get_model_path(keyword):
    # Dictionary mapping keywords to model paths
    model_paths = {
        'Phi3': "microsoft/Phi-3-mini-128k-instruct",
        'GPT2': "gpt2"
    }
    return model_paths[keyword]

def calculate_conditional_perplexity_batch(model, tokenizer, device, contexts_x, contexts_y):
    contexts = [x + y for x, y in zip(contexts_x, contexts_y)]
    
    encodings = tokenizer(contexts, return_tensors='pt', padding=True, truncation=True)
    input_ids = encodings.input_ids.to(device)
    attention_mask = encodings.attention_mask.to(device)
    
    labels = input_ids.clone()
    for i, (x, y) in enumerate(zip(contexts_x, contexts_y)):
        len_x = len(tokenizer.encode(x, add_special_tokens=False))
        labels[i, :len_x] = -100
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)

    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_attention_mask = attention_mask[:, 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_labels.size()) * shift_attention_mask

    per_token_loss = loss.sum(dim=1) / shift_attention_mask.sum(dim=1)
    perplexities = torch.exp(per_token_loss)

    return perplexities

def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path = get_model_path(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype="auto").to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    partial_answers_path = f'../dataset/{args.partial_answers_file}.jsonl'
    
    with open(partial_answers_path, 'r') as f:
        partial_answers = {json.loads(line)['id']: json.loads(line) for line in f}

    with open(args.input_file, 'r') as f:
        data = [json.loads(line) for line in f]

    output_data = []

    for entry in data:
        entry_id = entry['id']
        response = entry['generation']

        partial_answer_set = partial_answers.get(entry_id)
        if partial_answer_set is None:
            continue
        
        contexts_x = []
        contexts_y = []
        
        for partial_answer in partial_answer_set["partial_answers"]:
            response_text = f"{partial_answer['point_of_view']}. {partial_answer['explanation']}"
            chat = [
                {"role": "user", "content": response},
            ]
            context_x = tokenizer.apply_chat_template(chat, tokenize=False)
            context_y = response_text

            contexts_x.append(context_x)
            contexts_y.append(context_y)

        # Batch calculate perplexity
        batch_size = 20
        all_perplexities = []
        num_batches = ceil(len(contexts_x) / batch_size)

        for batch_idx in range(num_batches):
            batch_contexts_x = contexts_x[batch_idx * batch_size: min((batch_idx + 1) * batch_size, len(contexts_x))]
            batch_contexts_y = contexts_y[batch_idx * batch_size: min((batch_idx + 1) * batch_size, len(contexts_y))]
            batch_perplexities = calculate_conditional_perplexity_batch(model, tokenizer, device, batch_contexts_x, batch_contexts_y).cpu().numpy()
            all_perplexities.extend(batch_perplexities)

        average_perplexity = np.mean(all_perplexities)
        output_data.append(average_perplexity)

    average_score = np.mean(output_data)
    print(f"File: {args.input_file}, Average P.D. score: {average_score}")

if __name__ == "__main__":
    main()
