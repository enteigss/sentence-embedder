import torch
from transformers import AutoTokenizer

class SentenceTokenizer:
    def __init__(self, model_name="bert-base-uncased", max_length=128):
        """
        
        Initialize the tokenizer with a pre-trained model.

        Args:
            model_name (str): Name of the pre-trained model to use
            max_length (int): Maximum sequence length
        """
       
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def encode(self, sentences, padding=True, truncation=True, return_tensors="pt"):
        """
        
        Tokenize a single sentence or a batch of sentences.

        Args:
            sentences: A string or list of strings to tokenize
            padding: Whether to pad sequences to max_length
            truncation: whether to truncate sequences longer than max_length
            return_tensors: Format of the returned tensors ('pt' for PyTorch)

        Returns: 
            A dictionary containing:
                - input_ids: Token IDs
                - attention_mask: Mask indicating non-padded tokens
                - token_type_ids: Segment IDs
        
        """
        # Convert single sentence to list for batch processing
        if isinstance(sentences, str):
            sentences = [sentences]

        # Tokenize the sentences (__call__ method of tokenizer)
        encoded_input = self.tokenizer(
            sentences,
            padding=padding,
            truncation=truncation,
            max_length=self.max_length,
            return_tensors=return_tensors
        )

        return encoded_input
    
    def decode(self, token_ids):
        """
        
        Convert token Ids back to text.

        Args:
            token_ids: Tensor of token IDs

        Returns: 
            List of decoded sentences
        
        """
        
        # Convert to list if it's a tensor
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        # Handle both single sentences and batches
        if isinstance(token_ids[0], int):
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        else:
            return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in token_ids]
