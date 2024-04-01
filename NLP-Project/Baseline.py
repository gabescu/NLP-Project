#import spacy
#
# This code uses Spacy pretrained NER model, which is to say it is not trained by us
#nlp = spacy.load('en_core_web_sm')
#ner_categories = ["PERSON", "GPE", "LOC"]
#
#f = open("en_ewt-ud-test-masked.iob2", "r")
#text = f.readlines()
#f.close()
#phrases = []
#for line in text:
#    if line.find("text =") != -1:
#        phrases.append(line[line.find("text =") + 6 : line.find("\n")])
#        
#for phrase in phrases:
#    doc = nlp(phrase)
#
#    entities = []
#    for ent in doc.ents:
#        if ent.label_ in ner_categories:
#            entities.append((ent.text, ent.label_))
#        
#    i = 1
#    f = open("gold_file", "a")
#    f.write(f"Text = {phrase}\n")
#    for entity, category in entities:
#        f.write(f"{i}\t{entity}\t{category}\n")
#        i += 1
#    f.write("\n")
#    f.close()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Import necessary libraries
import torch
from transformers import BertTokenizer, BertForTokenClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels = NUM_LABELS)  # Set NUM_LABELS according to your dataset

# Tokenization
def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        tokenized_sentence.extend(tokenized_word)
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels

# Assuming `sentences` and `labels` are lists containing sentences and their corresponding labels
tokenized_texts_and_labels = [tokenize_and_preserve_labels(sent, labs) for sent, labs in zip(sentences, labels)]
tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

# Convert tokenized text and labels to tensor format
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts], maxlen=MAX_LEN, dtype="long", value=0.0, truncating="post", padding="post")
tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels], maxlen=MAX_LEN, value=tag2idx["PAD"], padding="post", dtype="long", truncating="post")

# Create attention masks to ignore padded tokens
attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

# Split data into train and validation sets and wrap them in DataLoader for efficient batching

# Fine-tuning setup
optimizer = AdamW(model.parameters(), lr=3e-5, eps=1e-8)
total_steps = len(train_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
for _ in range(EPOCHS):
    model.train()
    for batch in train_dataloader:
        b_input_ids, b_labels, b_masks = batch
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_masks, labels=b_labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.zero_grad()
    # Validation loop can be added here

model.save_pretrained("./ner_model")