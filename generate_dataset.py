import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import csv

# Load local model (GPT-Neo 125M, smaller and easier to run locally)
model_name = "EleutherAI/gpt-neo-125M"  # Change to "gpt-j-6B" if you have a larger setup
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load sentence transformer for scoring
scoring_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define prompts and ideal responses for scoring
data = [
    {"prompt": "What are the benefits of exercise?", "ideal_response": "Exercise provides both mental and physical health benefits."},
    {"prompt": "Explain photosynthesis.", "ideal_response": "Photosynthesis is a process where plants make food using sunlight, carbon dioxide, and water."},
    # Add more prompts as needed
]

# List to hold prompt-response-score data
results = []

# Function to generate response
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Function to calculate score based on similarity
def calculate_score(response, ideal_response):
    response_embedding = scoring_model.encode(response, convert_to_tensor=True)
    ideal_embedding = scoring_model.encode(ideal_response, convert_to_tensor=True)
    score = util.pytorch_cos_sim(response_embedding, ideal_embedding).item()
    return score

# Loop through each prompt, generate response, score, and save results
for entry in data:
    prompt = entry["prompt"]
    ideal_response = entry["ideal_response"]
    
    # Generate response from the model
    response = generate_response(prompt)
    
    # Calculate similarity score with the ideal response
    score = calculate_score(response, ideal_response)
    
    # Append results to the list
    results.append({"Prompt": prompt, "Response": response, "Score": score})

# Save results to CSV
csv_file = "prompt_responses.csv"
with open(csv_file, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["Prompt", "Response", "Score"])
    writer.writeheader()
    for result in results:
        writer.writerow(result)

print(f"Responses and scores saved to {csv_file}")
