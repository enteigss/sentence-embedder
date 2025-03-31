import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from torch.nn.functional import normalize

class SentenceTransformerEncoder(nn.Module):
    def __init__(
        self,
        model_name="bert-base-uncased",
        pooling_strategy="mean",
        embedding_dimension=None,
        normalize_embeddings=True,
        max_length=128
    ):
        """
        Initialize the sentence transformer encoder.
        
        Args:
            model_name (str): Name of the pre-trained transformer model to use
            pooling_strategy (str): Strategy for pooling ('mean', 'cls', 'max')
            embedding_dimension (int, optional): Size of the final embeddings after projection
                                               If None, no projection is applied
            normalize_embeddings (bool): Whether to L2-normalize the embeddings
            max_length (int): Maximum sequence length
        """
        super(SentenceTransformerEncoder, self).__init__()
        
        # Load pre-trained transformer model
        self.transformer = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        
        # Set pooling strategy
        self.pooling_strategy = pooling_strategy
        self.normalize_embeddings = normalize_embeddings
        
        # Get hidden size of the transformer model
        self.hidden_size = self.config.hidden_size
        
        # Create projection layer if embedding dimension is specified
        self.embedding_dimension = embedding_dimension
        if embedding_dimension is not None and embedding_dimension != self.hidden_size:
            self.projection = nn.Linear(self.hidden_size, embedding_dimension)
        else:
            self.projection = None
            self.embedding_dimension = self.hidden_size
            
        self.max_length = max_length
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Forward pass through the model.
        
        Args:
            input_ids (torch.Tensor): Tokenized input IDs
            attention_mask (torch.Tensor, optional): Mask for padding tokens
            token_type_ids (torch.Tensor, optional): Segment IDs for BERT
            
        Returns:
            torch.Tensor: Sentence embeddings
        """
        
        # Get transformer outputs 
        # last_hidden_state shape: [batch_size, sequence_length, hidden_size]
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # Get token embeddings from the last hidden state
        token_embeddings = transformer_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Apply pooling strategy
        if self.pooling_strategy == "cls":
            # Use [CLS] token embedding
            sentence_embeddings = token_embeddings[:, 0, :]
        
        elif self.pooling_strategy == "mean":
            # Apply mean pooling - take average of all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        elif self.pooling_strategy == "max":
            # Apply max pooling - take max over each dimension
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            sentence_embeddings = torch.max(token_embeddings, 1)[0]
        
        else:
            raise ValueError(f"Unsupported pooling strategy: {self.pooling_strategy}")
        
        # Apply projection if specified
        if self.projection is not None:
            sentence_embeddings = self.projection(sentence_embeddings)
        
        # Apply L2 normalization if specified
        if self.normalize_embeddings:
            sentence_embeddings = normalize(sentence_embeddings, p=2, dim=1)
        
        return sentence_embeddings
    
    def encode(self, tokenized_input):
        """
        Encode tokenized inputs to sentence embeddings.
        
        Args:
            tokenized_input: Dictionary with keys 'input_ids', 'attention_mask', 'token_type_ids'
            
        Returns:
            torch.Tensor: Sentence embeddings
        """
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            return self.forward(
                input_ids=tokenized_input['input_ids'],
                attention_mask=tokenized_input['attention_mask'],
                token_type_ids=tokenized_input.get('token_type_ids', None)
            )
    
    def get_embedding_dimension(self):
        """
        Returns the dimension of the generated sentence embeddings.
        """
        return self.embedding_dimension