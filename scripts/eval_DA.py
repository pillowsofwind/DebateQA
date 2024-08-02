import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Determine the debatability of answers to questions.")
    parser.add_argument('--model_name', type=str, default='MiniCPM', choices=['Phi3', 'MiniCPM'], help='Keyword for selecting the model.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSONL file containing questions and answers.')
    parser.add_argument('--partial_answers_file', type=str, default='test', help='Specify "test" or "dev" to select the partial answers file.')
    return parser.parse_args()

def get_model_path(model_name):
    model_paths = {
        'Phi3': "microsoft/Phi-3-mini-128k-instruct",
        'MiniCPM': "openbmb/MiniCPM-2B-dpo-bf16"
    }
    return model_paths.get(model_name, "openbmb/MiniCPM-2B-dpo-bf16")

def setup_model(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype="auto").to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, device

def is_debatable_answer(model, tokenizer, device, question, answer):
    prompt_template = (
        "Here is a question paired with an answer. Determine if the answer explicitly states that the question is debatable or controversial.\n\n"
        "Examples:\n"
        "Question: Is it ethical to use animals for scientific research?\n"
        "Answer: The use of animals in scientific research is a highly debated topic. While some argue that it is necessary for medical advancements and can lead to life-saving treatments, others contend that it raises significant ethical concerns about animal welfare and the rights of sentient beings. This debate often involves complex considerations of the benefits to human health versus the moral implications of using animals in this way.\n"
        "Response: 1\n\n"
        "Question: What is the boiling point of water at sea level?\n"
        "Answer: The boiling point of water at sea level is 100 degrees Celsius, a well-established scientific fact.\n"
        "Response: 0\n\n"
        "Question: Should schools replace physical education with computer coding classes?\n"
        "Answer: Schools are increasingly incorporating coding into their curriculums to prepare students for the digital age, but this does not necessarily mean that physical education should be replaced.\n"
        "Response: 0\n\n"
        "Instruction:\n"
        "- Respond with \"1\" if the answer explicitly states that the question is open to debate or considered controversial.\n"
        "- Respond with \"0\" if the answer does not acknowledge any debate or controversy regarding the question."
        "- Please only consider whether there is explicit statement regarding controversy, do not judge on other aspects of the answer, e.g., quality, truthfulness."
        "Question: {question}\nAnswer: {answer}\n\n"
        "Important: Your answer should only contain one digit 0 or 1.\n\n"
    )
    prompt = prompt_template.format(question=question, answer=answer)
    
    model_inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    generated_ids = model.generate(model_inputs.input_ids, max_length=2048)
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    digit = next((char for char in response if char in ['0', '1']))
    return int(digit)

def main():
    args = parse_args()
    model_path = get_model_path(args.model_name)
    model, tokenizer, device = setup_model(model_path)

    partial_answers_path = f'../dataset/{args.partial_answers_file}.jsonl'
    
    with open(partial_answers_path, 'r') as f:
        partial_answers = {json.loads(line)['id']: json.loads(line) for line in f}

    with open(args.input_file, 'r') as f:
        data = [json.loads(line) for line in f]

    score_sum = 0
    cnt = 0
    for entry in data:
        entry_id = entry['id']
        # print("entry_id: ", entry_id)
        response = entry['generation']

        partial_answer_set = partial_answers.get(entry_id)
        if partial_answer_set is None:
            continue
        
        question = partial_answer_set["question"]
        
        try:
            score = is_debatable_answer(model, tokenizer, device, question, response)
            score_sum += score
            cnt += 1
        except Exception as e:
            print(f"Error processing Q&A pair: {e}")

    if cnt > 0:
        average_score = score_sum / cnt
        print(f"File: {args.input_file}, Average D.A. Score: {average_score}")
    else:
        print(f"File: {args.input_file}, No valid Q&A pairs processed.")

if __name__ == "__main__":
    main()
