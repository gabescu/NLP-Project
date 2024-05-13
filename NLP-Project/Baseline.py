from datasets import Dataset
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
import pandas as pd

batch_size = 16

def read_iob2_file(path):
    data = []
    current_words = []
    current_tags = []

    for line in open(path, encoding='utf-8'):
        line = line.strip()

        if line:
            if line[0] == '#':
                continue
            tok = line.split('\t')

            current_words.append(tok[1])
            current_tags.append(tok[2])
        else:
            if current_words:
                data.append((current_words, current_tags))
            current_words = []
            current_tags = []

    if current_tags != []:
        data.append((current_words, current_tags))

    df = pd.DataFrame(data, columns=['words', 'tags'])
    df['id'] = df.index
    df = df[['id', 'words', 'tags']]
    
    return df

def tokenize_and_align_labels(dataset, word_column, tag_column, tokenizer):
    tokenized_inputs = tokenizer(dataset[word_column].tolist(), truncation = True, is_split_into_words = True, padding = True)

    labels = []
    for i, label in enumerate(dataset[tag_column]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if True else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs.data

dataset_test = read_iob2_file("en_ewt-ud-test-masked.iob2")
dataset_eval = read_iob2_file("en_ewt-ud-dev.iob2")
dataset_train = read_iob2_file("en_ewt-ud-train.iob2")

labels_to_idx = {"O": 0, "B-LOC": 1, 'I-LOC': 2, 'B-PER': 3, 'B-ORG': 4, 'I-ORG': 5, 'I-PER': 6}

model_checkpoint = "distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, padding = True)

dataset_test['tag_idx'] = dataset_test['tags'].apply(lambda x: [labels_to_idx[tag] for tag in x])
dataset_train['tag_idx'] = dataset_train['tags'].apply(lambda x: [labels_to_idx[tag] for tag in x])
dataset_eval['tag_idx'] = dataset_eval['tags'].apply(lambda x: [labels_to_idx[tag] for tag in x])

tokenized_data_test = tokenize_and_align_labels(dataset_test, "words", "tag_idx", tokenizer)
tokenized_data_train = tokenize_and_align_labels(dataset_train, "words", "tag_idx", tokenizer)
tokenized_data_eval = tokenize_and_align_labels(dataset_eval, "words", "tag_idx", tokenizer)

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(labels_to_idx))

training_args = TrainingArguments(output_dir = "test_trainer")

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis = -1)
    return metric.compute(predictions = predictions, references = labels)

training_args = TrainingArguments(
    output_dir = "test_trainer",
    evaluation_strategy = "epoch",
    learning_rate = 2e-5,
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size = batch_size,
    num_train_epochs = 3
)

data_collator = DataCollatorForTokenClassification(tokenizer)
metric = load_metric("seqeval")

test_dataset = Dataset.from_dict({
    'id': range(len(tokenized_data_train['input_ids'])),
    'input_ids': tokenized_data_train['input_ids'],
    'attention_mask': tokenized_data_train['attention_mask'],
    'labels': tokenized_data_train['labels']
})
train_dataset = Dataset.from_dict({
    'id': range(len(tokenized_data_train['input_ids'])),
    'input_ids': tokenized_data_train['input_ids'],
    'attention_mask': tokenized_data_train['attention_mask'],
    'labels': tokenized_data_train['labels']
})
eval_dataset = Dataset.from_dict({
    'id': range(len(tokenized_data_eval['input_ids'])),
    'input_ids': tokenized_data_eval['input_ids'],
    'attention_mask': tokenized_data_eval['attention_mask'],
    'labels': tokenized_data_eval['labels']
})

trainer = Trainer(
    data_collator = data_collator,
    tokenizer = tokenizer,
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    compute_metrics = compute_metrics,
)

trainer.train()