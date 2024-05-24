from datasets import Dataset
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments
from transformers import TrainingArguments, Trainer
import numpy as np
import pandas as pd
import os


batch_size = 16

labels_to_idx = {"O": 0, "B-CRT": 1, "I-CRT": 2, "B-POT": 3, "I-POT": 4, "B-SPL": 5, "I-SPL": 6, "B-HOUSE": 7, "I-HOUSE": 8, 
                 "B-MAG": 9, "I-MAG": 10, "B-CRS": 11, "I-CRS": 12, "B-MIT": 13, "I-MIT": 14, "B-PER": 15, "I-PER": 16, 
                 "B-ORG": 17, "I-ORG": 18, "B-LOC": 19, "I-LOC": 20, "B-EVN": 21, "I-EVN": 22, "B-TTL": 23, "I-TTL": 24}
idx_to_labels = ["O", "B-CRT", "I-CRT", "B-POT", "I-POT", "B-SPL", "I-SPL", "B-HOUSE", "I-HOUSE", 
                 "B-MAG", "I-MAG", "B-CRS", "I-CRS", "B-MIT", "I-MIT", "B-PER", "I-PER", 
                 "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-EVN", "I-EVN", "B-TTL", "I-TTL"]

data1 = []
data2 = []
data3 = []
def read_iob2_file(path, data):
    current_words = []
    current_tags = []

    for line in open(path, encoding = "cp1252"):
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

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis = 2)

    true_predictions = [
        [idx_to_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [idx_to_labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
    
cwd = os.getcwd()
dataset_train = read_iob2_file(f"{cwd[0:-11]}HP custom labels\hp1_custom_labels.iob2", data1)
dataset_train = read_iob2_file(f"{cwd[0:-11]}HP custom labels\hp2_custom_labels.iob2", data1)
dataset_train = read_iob2_file(f"{cwd[0:-11]}HP custom labels\hp3_custom_labels.iob2", data1)
dataset_train = read_iob2_file(f"{cwd[0:-11]}HP custom labels\hp4_custom_labels.iob2", data1)
dataset_eval = read_iob2_file(f"{cwd[0:-11]}HP custom labels\hp5-custom-labels.iob2", data2)
dataset_eval = read_iob2_file(f"{cwd[0:-11]}HP custom labels\hp6_custom_labels.iob2", data2)
dataset_test = read_iob2_file(f"{cwd[0:-11]}HP custom labels\hp7_custom_labels.iob2", data3)

model_checkpoint = "distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, padding = True)

dataset_test["tag_idx"] = dataset_test["tags"].apply(lambda x: [labels_to_idx[tag] for tag in x])
dataset_train["tag_idx"] = dataset_train["tags"].apply(lambda x: [labels_to_idx[tag] for tag in x])
dataset_eval['tag_idx'] = dataset_eval['tags'].apply(lambda x: [labels_to_idx[tag] for tag in x])

tokenized_data_test = tokenize_and_align_labels(dataset_test, "words", "tag_idx", tokenizer)
tokenized_data_train = tokenize_and_align_labels(dataset_train, "words", "tag_idx", tokenizer)
tokenized_data_eval = tokenize_and_align_labels(dataset_eval, "words", "tag_idx", tokenizer)

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(labels_to_idx))

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

training_args = TrainingArguments(
    output_dir = "test_trainer",
    evaluation_strategy = "epoch",
    learning_rate = 2e-5,
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size = batch_size,
    num_train_epochs = 5
)

data_collator = DataCollatorForTokenClassification(tokenizer)
metric = load_metric("seqeval")

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

trainer.save_model("Harry_Potter_Model_2")

predictions, labels, _ = trainer.predict(test_dataset)
predictions = np.argmax(predictions, axis = 2)

true_predictions = [
    [idx_to_labels[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
true_labels = [
    [idx_to_labels[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

results = metric.compute(predictions = true_predictions, references = true_labels)

print(results)
