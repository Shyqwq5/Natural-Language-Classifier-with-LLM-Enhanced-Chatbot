# natural-language-classifier

## 🛠 Usage

### 0. Set the nesscery environment

Run the script to set environment:

```bash
python -m venv venv
source venv/bin/activate
export PYTHONPATH=$(pwd)
pip install --upgrade pip
pip install -r requirements.txt
```

### 1. Preprocess the dataset

Run the ingestion script to clean and prepare the data:

```bash
python ingest.py
```

### 2. Train the model

Run the train model script to train and save the model, vectorizer, mapping:

```bash
python train_model.py
```

### 3.Talk with the model

Simple run simple_interface.py and you can start chat with model in terminal.

```bash
python simple_interface.py
```

### 3.Talk with the model with llm

If you want use natrual language talk with model, simple run chatbot_interface.py! It will using llm to analyse input and polish output

```bash
python chatbot_interface.py
```
