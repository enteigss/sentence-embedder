import torch
from transformers import AutoTokenizer

from src.models import SentenceTokenizer, SentenceTransformerEncoder

def main():
    # Initialize tokenizer
    tokenizer = SentenceTokenizer(model_name="bert-base-uncased", max_length=128)
    
    # Example sentences
    sentences = [
        "This is an example sentence.",
        "This is another example sentence that is longer.",
        "I'm not good at thinking about example sentences."
    ]
    
    # Tokenize sentences
    encoded = tokenizer.encode(sentences)
    print("Sentences to encode:", sentences)
    print(f"Input IDs shape: {encoded['input_ids'].shape}")
    print(f"Attention mask shape: {encoded['attention_mask'].shape}")
    
    # Create sentence embeddings
    model = SentenceTransformerEncoder(
        model_name="bert-base-uncased", 
        pooling_strategy="cls",
        embedding_dimension=300, 
        normalize_embeddings=True
    )
    
    # Generate embeddings
    with torch.no_grad():
        sentence_embeddings = model.encode(encoded)
    
    print(f"Sentence embeddings shape: {sentence_embeddings.shape}")
    print(f"Embeddings are normalized: {torch.allclose(torch.norm(sentence_embeddings, p=2, dim=1), torch.ones(len(sentences)))}")
    print(f"Sentence embeddings: {sentence_embeddings}")

if __name__ == "__main__":
    main()
