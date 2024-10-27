import datasets
import torch
import pandas as pd
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForQuestionAnswering, AdamW

#load dataset
dataset = load_dataset("squad")
dataset

#split train and test
train_df = dataset['train'].to_pandas()
test_df = dataset['validation'].to_pandas()
train_df.to_csv('train_data.csv')
test_df.to_csv('test_data.csv')
# print(train_df.head())
# print(test_df.head())

train_df = train_df[['question','answers']]
test_df = test_df[['question','answers']]

# Extract answers from dictionary
train_df['answers'] = train_df['answers'].apply(lambda x: x['text'][0])
test_df['answers'] = test_df['answers'].apply(lambda x: x['text'][0])

train_df.head()
print('Train data size: ',len(train_df))
print('Test data size: ',len(test_df))

# Load the tokenizer (BERT model)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Example tokenization of question-answer pairs
def tokenize_data(df):
    return tokenizer(df["question"].tolist(), 
                     df["answers"].tolist(), 
                     truncation=True, 
                     padding=True, 
                     return_tensors="pt")

# Tokenize the training and test data
train_encodings = tokenize_data(train_df)
test_encodings = tokenize_data(test_df)


class QADataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

# Create PyTorch datasets
train_dataset = QADataset(train_encodings)
test_dataset = QADataset(test_encodings)

# Load the pre-trained BERT model for QA
model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")

# Define optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

print('End of program.')

print('dataset')