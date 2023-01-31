import numpy as np
import torch
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import DistilBertTokenizerFast
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

class emotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def read_data(path, filename, emotion_encode):
    texts = []
    labels = []
    fl = path + filename
    with open(fl) as f:
        content = f.readlines()
    for s in content:
        texts.append(s.strip().split(";")[0])
        labels.append(emotion_encode[s.strip().split(";")[1]])
    return texts, labels

def read_data_swahili_emotion(path, filename, emotion_encode):
    texts = []
    labels = []
    fl = path + filename
    with open(fl, errors='ignore') as f:
        content = f.readlines()
    for s in content: 
        if len(s.strip().rsplit(",", 1)) != 2:
            print(s)
        text = s.strip().rsplit(",", 1)[0]
        label = s.strip().rsplit(",", 1)[1]
        if not ";" in label: 
            label = label.replace('[', '')
            label = label.replace(']', '')
        else:
            label = label[2] # third character, ignore second label for now
        texts.append(text)
        # print("debug:", label)
        labels.append(int(label))
    return texts, labels

def compute_metrics(p):    
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    # recall = recall_score(y_true=labels, y_pred=pred)
    # precision = precision_score(y_true=labels, y_pred=pred)
    # f1 = f1_score(y_true=labels, y_pred=pred)
    # return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1} 
    # for pr, la in zip(pred, labels):
    #     if pr != la : print("pred: ", pr, "  labels: ", la)
    return {"accuracy": accuracy} 

def main():
    emotion_encode = {
        "sadness" : 0,
        "love"    : 1,
        "anger"   : 2,
        "joy"     : 3,
        "fear"    : 4,
        "surprise": 5
    }
    emotion_decode = {
        "0" : "sadness",
        "1" : "love",
        "2" : "anger",
        "3" : "joy",
        "4" : "fear",
        "5" : "surprise"
    }
    # path = '../data/emotion/'
    path = '/dccstor/sentient1/git/SwahBERT/emotion_dataset/'
    num_labels = 7
    max_length = 64
    # model_path = '../models/swahbert-base-uncased/'
    model_path = '/dccstor/colbert-ir/yulongl/other_experiments/unified_infoNce_token_align_swahbert/tau10_conRatio10/checkpoint-epoch0016-step000100000/'
    # tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    # model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels).to("cuda")

    # train_texts, train_labels = read_data(path, "train.txt", emotion_encode)
    # test_texts,  test_labels = read_data(path, "test.txt", emotion_encode)
    # val_texts, val_labels = read_data(path, "val.txt", emotion_encode)
    train_texts, train_labels = read_data_swahili_emotion(path, "train.csv", emotion_encode)
    test_texts,  test_labels = read_data_swahili_emotion(path, "test.csv", emotion_encode)
    val_texts, val_labels = read_data_swahili_emotion(path, "valid.csv", emotion_encode)

    train_encodings = tokenizer(train_texts, truncation=False, padding=True)
    val_encodings = tokenizer(val_texts, truncation=False, padding=True)
    test_encodings = tokenizer(test_texts, truncation=False, padding=True)

    train_dataset = emotionDataset(train_encodings, train_labels)
    val_dataset = emotionDataset(val_encodings, val_labels)
    test_dataset = emotionDataset(test_encodings, test_labels)

    training_args = TrainingArguments(
        output_dir='../snap/emotion_results',          # output directory
        num_train_epochs=8,              # total number of training epochs
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=32,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='../snap/emotion_results/logs',            # directory for storing logs
        evaluation_strategy='epoch',
        logging_steps=10,
        learning_rate=5e-5,
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        compute_metrics=compute_metrics,
        eval_dataset=val_dataset             # evaluation dataset
    )

    trainer.train()
    trainer.evaluate()

if __name__ == "__main__":
    main()