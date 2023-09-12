import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Define your data and labels here
import pandas as pd

df = pd.read_csv(r'C:\Users\Eli Brignac\Downloads\Finantial_Sentiment\Financial-Sentiment-Analysis\sentiment_data.csv')
texts = df['Sentence'].tolist()
labels = df['Sentiment'].tolist()

# Split the dataset into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)

# Convert text data to BERT input format
def tokenize_text(texts, tokenizer, max_length):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks

max_seq_length = 128
train_input_ids, train_attention_masks = tokenize_text(train_texts, tokenizer, max_seq_length)
val_input_ids, val_attention_masks = tokenize_text(val_texts, tokenizer, max_seq_length)

# Convert labels to tensors
train_labels = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)
val_labels = torch.tensor(val_labels, dtype=torch.float32).unsqueeze(1)

# Create DataLoader for training and validation sets
batch_size = 32
train_data = TensorDataset(train_input_ids, train_attention_masks, train_labels)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

val_data = TensorDataset(val_input_ids, val_attention_masks, val_labels)
val_dataloader = DataLoader(val_data, batch_size=batch_size)

# Define training parameters
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 5
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
        
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{epochs} - Average Training Loss: {avg_train_loss:.4f}")

# Evaluation
model.eval()
val_loss = 0

predictions = []
true_labels = []
with torch.no_grad():
    for batch in val_dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
        
        outputs = model(**inputs)
        loss = outputs.loss
        val_loss += loss.item()

        predictions.extend(torch.sigmoid(outputs.logits).cpu().numpy())
        true_labels.extend(inputs['labels'].cpu().numpy())

avg_val_loss = val_loss / len(val_dataloader)
predictions = np.array(predictions)
true_labels = np.array(true_labels)
predicted_classes = (predictions > 0.5).astype(np.float32)
val_accuracy = accuracy_score(true_labels, predicted_classes)

print(true_labels)
print(labels)
print(predictions)
print(f"Validation Loss: {avg_val_loss:.4f} - Validation Accuracy: {val_accuracy:.4f}")
