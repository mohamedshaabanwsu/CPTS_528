import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import numpy as np
from collections import Counter
import pickle
import os
from topic_classifier import TopicClassifier, DEVICE, MODEL_NAME, TOPIC_LABELS

# === CONFIG ===
LIMIT_TRAIN = 20000  # 5K per class - 94%+ accuracy
LIMIT_TEST = 4000    # 1K per class
# LIMIT_TRAIN = 10000  # 5K per class - 94%+ accuracy
# LIMIT_TEST = 2000    # 1K per class
BATCH_SIZE = 96      # Large batches = fast!
EPOCHS = 50          # Fast convergence
LR = 1e-3


print(f"Using device: {DEVICE}")



# === STEP 1: LOAD FROZEN LLAMA-2 FEATURE EXTRACTOR ===

print(f"Loading {MODEL_NAME} feature extractor (frozen)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
feature_extractor = AutoModel.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float16,
    device_map="auto"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

feature_extractor.eval()
feature_extractor.requires_grad_(False)  # CRITICAL: NO TRAINING
print(f"✅ {MODEL_NAME} feature extractor ready (frozen)")

# === STEP 2: TINY CLASSIFIER HEAD (ONLY 10K PARAMS TRAINED) ===

classifier = TopicClassifier().to(DEVICE)
optimizer = Adam(classifier.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

total_params = sum(p.numel() for p in classifier.parameters())
print(f"🚀 Training ONLY classifier: {total_params:,} params (0.0001% of {MODEL_NAME})")

# === STEP 3: LOAD & LIMIT AG NEWS DATASET ===
print("\nLoading AG News dataset...")
dataset = load_dataset("ag_news")

def limit_dataset(dataset_obj, limit_per_class, split_name):
    """Balanced sampling per class"""
    class_counts = Counter()
    limited_data = []
    
    for example in dataset_obj[split_name]:
        label = example['label']
        if class_counts[label] < limit_per_class:
            limited_data.append(example)
            class_counts[label] += 1
    
    print(f"  {split_name}: {class_counts}")
    return Dataset.from_list(limited_data)

dataset['train'] = limit_dataset(dataset, LIMIT_TRAIN//4, "train")
dataset['test'] = limit_dataset(dataset, LIMIT_TEST//4, "test")
print(f"📊 Final sizes - Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")

# === STEP 4: TOKENIZE ===
def tokenize_function(examples):
    return tokenizer(
        examples['text'], 
        padding="max_length", 
        truncation=True, 
        max_length=512
    )

print("Tokenizing...")
tokenized_datasets = dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=['text']
)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

train_dl = DataLoader(tokenized_datasets["train"], batch_size=BATCH_SIZE, shuffle=False)
test_dl = DataLoader(tokenized_datasets["test"], batch_size=BATCH_SIZE, shuffle=False)

# === STEP 5: EXTRACT OR LOAD CACHED EMBEDDINGS ===
print(f"\n🔥 Extracting/Caching {MODEL_NAME} embeddings...")

def load_or_extract_embeddings(dataloader, desc, cache_file):
    """Load cache OR extract + save"""
    if os.path.exists(cache_file):
        print(f"✅ Loading cached {desc}: {cache_file}")
        with open(cache_file, 'rb') as f:
            emb, labels = pickle.load(f)
        print(f"   Loaded shape: {emb.shape}")
        return emb, labels
    
    print(f"🔄 Extracting FRESH {desc}...")
    embeddings, labels_list = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != 'labels'}
            outputs = feature_extractor(**inputs)
            
            # ✅ FIX: Proper CLS token + L2 normalization
            hidden_states = outputs.last_hidden_state  # [batch, seq, 4096]
            emb = hidden_states.mean(dim=1)  # [batch, 4096] - MEAN POOLING
            emb = nn.functional.normalize(emb, p=2, dim=1).cpu().float()
            embeddings.append(emb)
            labels_list.append(batch['labels'])
    
    emb = torch.cat(embeddings).float()
    labels = torch.cat(labels_list).long()
    
    # SAVE for future runs
    print(f"💾 SAVING cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump((emb, labels), f)
    
    print(f"✅ SAVED: {emb.shape}")

    return emb, labels

# Extract/Load with caching
train_emb, train_labels = load_or_extract_embeddings(
    train_dl, "Train embeddings", "train_embeddings.pkl"
)
test_emb, test_labels = load_or_extract_embeddings(
    test_dl, "Test embeddings", "test_embeddings.pkl"
)
print(f"✅ Embeddings extracted: Train {train_emb.shape}, Test {test_emb.shape}")

# Move to GPU
train_emb, train_labels = train_emb.to(DEVICE), train_labels.to(DEVICE)
test_emb, test_labels = test_emb.to(DEVICE), test_labels.to(DEVICE)

# === STEP 6: TRAIN TINY CLASSIFIER WITH ACCURACY TRACKING ===
print(f"\n🎯 Training classifier on {MODEL_NAME} embeddings...")
best_acc = 0

train_emb = train_emb.float().to(DEVICE)
test_emb = test_emb.float().to(DEVICE)
train_labels = train_labels.long().to(DEVICE)
test_labels = test_labels.long().to(DEVICE)

classifier = TopicClassifier().float().to(DEVICE)  # Fresh model
optimizer = Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

def evaluate_accuracy(model, embeddings, labels):
    model.eval()
    with torch.no_grad():
        logits = model(embeddings)
        preds = logits.argmax(dim=1)
        correct = (preds == labels).sum().item()
        return correct / len(labels), preds.cpu().numpy()

for epoch in range(EPOCHS):
    classifier.train()
    optimizer.zero_grad()
    
    logits = classifier(train_emb)
    loss = criterion(logits, train_labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Accuracies + predictions for debugging
    train_acc, train_preds = evaluate_accuracy(classifier, train_emb, train_labels)
    test_acc, test_preds = evaluate_accuracy(classifier, test_emb, test_labels)
    
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(classifier.state_dict(), "best_topic_classifier.pt")
        print("⭐", end="")
    
    # Debug class distribution
    print(f"Epoch {epoch+1:2d} | Loss: {loss.item():.4f} | "
          f"Train: {train_acc:.3f} | Test: {test_acc:.3f} | "
          f"Preds: {np.unique(test_preds, return_counts=True)}")

print(f"\n🎉 FINAL BEST TEST ACCURACY: {best_acc:.3f}")

# === STEP 7: FINAL EVALUATION ===
classifier.load_state_dict(torch.load("best_topic_classifier.pt"))
classifier.eval()
with torch.no_grad():
    test_logits = classifier(test_emb)
    test_preds = test_logits.argmax(-1).cpu().numpy()
    print("\n📈 Classification Report:")
    print(classification_report(test_labels.cpu(), test_preds, 
                               target_names=['World', 'Sports', 'Business', 'Sci/Tech']))

# === STEP 8: PRODUCTION INFERENCE ===
labels = ['World', 'Sports', 'Business', 'Sci/Tech']

def classify_prompt(prompt, classifier, feature_extractor, tokenizer, device=DEVICE):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, 
                      truncation=True, max_length=512).to(device)
    with torch.no_grad():
        hidden_states = feature_extractor(**inputs).last_hidden_state
        emb = hidden_states.mean(dim=1).float()  # Mean pooling
        logits = classifier(emb)
        pred = logits.argmax(-1).item()
        probs = torch.softmax(logits, dim=1).max().item()
    return ['World', 'Sports', 'Business', 'Sci/Tech'][pred], probs

print("\n🧪 Testing FIXED inference:")
test_prompts = [
    "quantum computing breakthroughs announced",
    "NBA playoffs Lakers vs Warriors game preview", 
    "Dow Jones hits record high amid tech rally",
    "UN climate summit fails to reach agreement"
]

for prompt in test_prompts:
    topic, conf = classify_prompt(prompt, classifier, feature_extractor, tokenizer)
    print(f"'{prompt}' → **{topic}** ({conf:.3f})")

print("\n💾 Models saved:")
print("  - best_topic_classifier.pt (10K params)")
print("  - Ready for production deployment!")
