import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, r2_score


# Multi-Task Learning Model
class MultiTaskModel(nn.Module):
    def __init__(
        self,
        model_name="bert-base-uncased",
        pooling_strategy="mean",
        task_a_num_classes=5,  # Example: 5 classes for the classification task
        task_b_type="sentiment"  # Example: sentiment analysis as Task B
    ):
        """
        Multi-task learning model implementation.
        
        Args:
            model_name: Pre-trained transformer model name
            pooling_strategy: Strategy for sentence embedding pooling ('mean', 'cls', 'max')
            task_a_num_classes: Number of classes for Task A (sentence classification)
            task_b_type: Type of Task B 
        """
        super(MultiTaskModel, self).__init__()
        
        # Shared encoder for both tasks
        self.encoder = SentenceTransformerEncoder(
            model_name=model_name,
            pooling_strategy=pooling_strategy,
            normalize_embeddings=False  # Disable normalization for the multi-task scenario
        )
        
        # Get embedding dimension from the encoder
        self.embedding_dim = self.encoder.get_embedding_dimension()
        
        # Task A: Sentence Classification Head
        self.task_a_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.embedding_dim, task_a_num_classes)
        )
        
        # Task B: Second NLP Task (Sentiment Analysis in this example)
        self.task_b_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.embedding_dim, 1)  # Regression for sentiment score
        )
        
        self.task_b_type = task_b_type
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, task=None):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tokenized input IDs
            attention_mask: Mask for padding tokens
            token_type_ids: Segment IDs for BERT
            task: Which task to perform ('task_a', 'task_b', or None for both)
            
        Returns:
            Dictionary with task-specific outputs
        """
        # Get embeddings from the encoder
        sentence_embeddings = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        outputs = {}
        
        # Task A: Sentence Classification
        if task is None or task == "task_a":
            task_a_logits = self.task_a_classifier(sentence_embeddings)
            outputs["task_a"] = task_a_logits
        
        # Task B: Sentiment Analysis or Named Entity Recognition
        if task is None or task == "task_b":
            # For sentence-level tasks like sentiment analysis
            task_b_outputs = self.task_b_head(sentence_embeddings)
            outputs["task_b"] = task_b_outputs
        
        return outputs
    
    def predict(self, input_ids, attention_mask=None, token_type_ids=None, task=None):
        """
        Make predictions using the trained model.
        
        Args:
            input_ids: Tokenized input IDs
            attention_mask: Mask for padding tokens
            token_type_ids: Segment IDs for BERT
            task: Which task to perform ('task_a', 'task_b', or None for both)
            
        Returns:
            Dictionary with task-specific predictions
        """

        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                task=task
            )

            predictions = {}

            if task is None or task == "task_a":
                if "task_a" in outputs:
                    task_a_logits = outputs["task_a"]
                    task_a_probs = F.softmax(task_a_logits, dim=1)
                    task_a_preds = torch.argmax(task_a_probs, dim=1)
                    predictions["task_a"] = {
                        "logits": task_a_logits,
                        "probabilities": task_a_probs,
                        "predictions": task_a_preds
                    }

            if task is None or task == "task_b":
                if "task_b" in outputs:
                    if self.task_b_type == "sentiment":
                        predictions["task_b"] = {
                            "scores": outputs["task_b"]
                        }

            return predictions