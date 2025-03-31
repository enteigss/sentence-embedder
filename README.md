# Project Title

## Setup and Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`

## Running the Project
How to run with terminal:
```bash
python main.py
```

## Docker Support
To build and run with Docker:
```bash
docker build -t your-project-name .
docker run your-project-name
```

## Task 1: 
The transformer was built on top of Huggingface’s AutoTokenizer and AutoTransformer models. Using these classes from Hugginface allows the user to choose what type of model they want, utilizing the AutoModel’s from_pretrained method. By default, the tokenizer uses padding and truncation to force the token sequences to be the right length, but this can be turned off. The tokenizer class is used to turn a sentence in the form of a string into a sequence of tokens that can be fed into the transformer. 

I designed the transformer class to allow the user to again select the model, the pooling strategy, normalization setting, and if they would like to project the final embedding. The transformer will take the sequence of tokens outputted by the tokenizer and transform it into an embedding representing the sentence. 

The pooling strategy I would choose to use if I had to would depend on the circumstances. If working with smaller sentences and/or have fewer computational resources, using CLS token pooling would be a good option because it is computationally efficient, but may not capture the full semantic meaning of a large sentence. On the other hand, mean pooling generally produces better semantic representations because it incorporates information from all words, which would be a good option if dealing with long sentences or if the resources available allows for it. 

I would choose to use L2 normalization when using the model for tasks that don’t value magnitudinal information. For example, when detecting similarities between documents of varying lengths magnitudinal information is not useful, and may bog the model down. Generally, however, normalization would not help the model. 

## Task 2: 
I built the multi-task model on top of the transformer, simply adding a classification head and a sentiment analysis head to allow for multi-task capabilities. The tasks will leverage a shared representation of the sentence given by the transformer built in task 1. The classification head is a sequential module with dropout and a simple linear layer outputting data of the dimensions set by the user, and the sentiment analysis layer outputs a single number representing the sentiment score, suitable for regression-based sentiment scoring. Using dropout decreases the chances of overfitting, and encourages better representation across tasks. This way, the model will be forced to use alternative pathways, making it more likely to find representations that benefit both tasks. Normalization is turned off for the multi-task model to preserve as much information as possible. The forward pass inputs the sentence embeddings into both of the heads and returns both of the outputs. The prediction method will take the logits produced by the forward pass for task A and use it to generate probabilities using softmax and the predictions themselves using whichever class had the highest probability. It will also output the sentiment score for task B. 

## Task 3:
## 1. Entire Network Frozen

### Implications
- Parameters cannot update, so model cannot change or learn

### Advantages
- "Training" would be really fast, but wouldn't really be training because the model would not change
- Keeps all knowledge from pretraining, so it cannot get worse or forget
- Cannot overfit

### How it should be used
- Useful as a baseline to see how well the model performs without any tuning, and can be compared to later as the model learns
- If you would like to use this part of the network as simply a feature extractor for use later downstream

## 2. Only Transformer Frozen

### Implications
- Task specific parts of the model will have to work with the feature representation of the pretrained encoder
- The transformer backbone won't change, so its feature representation will remain generic

### Advantages
- Faster training because there are less parameters to update
- Cannot forget or get worse
- Still allows for task specific parts to learn from the data
- Reduces risk of overfitting

### How it should be used
- If the tasks are well aligned with what the encoder model was trained on, the pre-trained model may be a fine approach
- If your tasks are very domain specific, training the encoder may be necessary to further refine the representation of the text
- If you have limited training data, freezing the transformer will reduce the risk of overfitting

## 3. Only One Task-Specific Head Frozen

### Implications
- Transformer backbone and one task head can learn
- One task is fixed and other is still learning
- Task of frozen head may decline

### Advantages/Disadvantages
- Protects performance of frozen head
- Model can still adapt while preserving the frozen head
- Model may shift to benefit unfrozen task head, but frozen task head cannot adapt to these changes

### How it should be used
- If you are happy with the performance of one task head and would like to focus on the other

## Part 2: Practical Considerations

### 1. Choosing Pre-Trained Model
- Choose a model pre-trained in a domain similar or adjacent to your intended use, as its representation will likely have useful information for your task
- BERT works well for generic tasks, but you could use more domain-specific models like BioBERT
- Consider model size: larger models likely perform better but are more costly

### 2. Freezing/Unfreezing Layer Strategy
- Start with freezing the transformer backbone layer, allowing only task-specific heads to train (especially with limited data to avoid overfitting)
- Depending on available data and need for domain-specific learning, slowly unfreeze layers from top to bottom over successive training rounds
- If data allows, fine-tune the entire model end-to-end for highest quality (with strong regularization to prevent overfitting)

### 3. Rationale
- Lower layers detect more general patterns, so keeping bottom layers frozen preserves general linguistic information while allowing higher layers to learn domain-specific patterns
- Freezing the backbone reduces training time and risk of catastrophic forgetting
- Allowing later unfreezing of layers helps avoid quality loss due to domain mismatching


## Task 4: 
Data Handling
The MultiTaskDataset class handles data for both tasks. It properly returns data when indexed.

Loss Function
The loss function is handled by a custom class MultiTaskLoss that combines losses for both tasks. The user can configure the task weights to prioritize one task over another. I used cross entropy loss for the classification task, and MSE for the regression task. The class then combines these into a single loss. 

Training Loop
The training loop processes the data in batches while keeping track of various metrics, including the predictions for each sample. This is then used later to calculate metrics such as accuracy and more. 

Metrics Tracking
The loop keeps track of various metrics allowing the training to be monitored. This allows for detection of task imbalances, over/underfitting, and making better informed decisions making changes to the models hyperparameters. 

Validation Loop
There is a validation loop to allow the model to be evaluated on unseen data. 

Forward Pass
The model performs task specific forward passes, and will conduct a forward pass for each task in each loop. This isolates the passes for each task, allowing for easier debugging, task-specific inference post training, and a more modular architecture allowing for adding more tasks. 

Synthetic Data 
I created synthetic data and trained the model on this data to make sure it works. 
