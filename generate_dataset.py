import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import csv

# Load local LLM model (e.g., GPT-Neo 125M) for response generation
# model_name = "EleutherAI/gpt-neo-1.3B"
model_name = "Meta-Llama-3.1-8B-Instruct-GGUF"
data = 'databricks/databricks-dolly-15k'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load a sentence embedding model for scoring
scoring_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define prompts to generate responses
prompts = [
    "What are the benefits of exercise?",
    "Explain photosynthesis.",
    "Describe the importance of sleep for health.",
    # Add more prompts as needed
]

# List to store results
results = []

# Function to generate response from the LLM
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Function to calculate semantic similarity between prompt and response
def calculate_similarity(prompt, response):
    prompt_embedding = scoring_model.encode(prompt, convert_to_tensor=True)
    response_embedding = scoring_model.encode(response, convert_to_tensor=True)
    score = util.pytorch_cos_sim(prompt_embedding, response_embedding).item()
    return score

# Loop through each prompt, generate response, score, and save results
for prompt in prompts:
    # Generate response
    response = generate_response(prompt)
    
    # Calculate relevance score based on prompt-response similarity
    score = calculate_similarity(prompt, response)
    
    # Append results to the list
    results.append({"Prompt": prompt, "Response": response, "Score": score})

# Save results to CSV
csv_file = "generated_responses.csv"
with open(csv_file, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["Prompt", "Response", "Score"])
    writer.writeheader()
    for result in results:
        writer.writerow(result)

print(f"Responses and scores saved to {csv_file}")
