# bert-base-uncased-imdb-finetuned

## Model
- Base: bert-base-uncased (HuggingFace Transformers)
- Fine-tuned for: IMDB sentiment dataset (stanfordnlp/imdb)
- Model class: BertForSequenceClassification
- Output dim (classifier): 20

## Dataset
- Source: stanfordnlp/imdb (train: 25000, test: 25000, unsupervised: 50000)
- Features: ['text', 'label']
- Note: IMDB is a binary sentiment dataset (labels: ['neg', 'pos'])

## Tokenization / Preprocessing
- Tokenizer: BertTokenizer (bert-base-uncased)
- Max length used: 128, truncation=True, padding=max_length

## Fine-tuning details
- TrainingArguments (selected):
    - per_device_train_batch_size: 8
    - per_device_eval_batch_size: 8
    - num_train_epochs: 1
    - learning_rate: 5e-05
    - weight_decay: 0.01
    - eval_strategy: IntervalStrategy.EPOCH
    - save_total_limit: 2

- Important observation: model instantiated with num_labels=20. The IMDB dataset is binary; ensure num_labels matches dataset labels (2) to avoid label-mismatch issues.

## Evaluation
- Eval results (trainer.evaluate): {'eval_loss': 0.2963371276855469}
- Example single inference (from this notebook):
    - sample_text (truncated): I love sci-fi and am willing to put up with a lot. Sci-fi movies/TV are usually underfunded, under-appreciated and misunderstood. I tried to like this, I really did, but it is to good TV sci-fi as Babylon 5 is to Star Trek (the original). Silly prosthetics, cheap cardboard sets, stilted dialogues, C...
    - predicted_label: 0
    - logits: [[7.872928142547607, 3.568333148956299, -3.0614819526672363, -3.769695281982422, -3.231588125228882, -3.00354266166687, -2.8586370944976807, -3.4951975345611572, -3.32694411277771, -3.206875801086426, -3.5429444313049316, -3.2764666080474854, -3.6933398246765137, -3.0684289932250977, -3.331453323364258, -3.3510794639587402, -3.703106641769409, -3.4574875831604004, -3.8161957263946533, -3.5096828937530518]]

