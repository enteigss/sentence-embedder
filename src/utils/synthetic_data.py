import random

def create_synthetic_data(num_samples=100, num_classes=5):
    """
    Create synthetic data for multi-task learning with balanced classes.
    
    Args:
        num_samples: Number of samples to generate
        num_classes: Number of classes for Task A
        
    Returns:
        Dictionary with task_a_data and task_b_data
    """
    # For Task A, ensure each class has enough examples
    task_a_data = {
        "sentences": [],
        "labels": []
    }
    
    task_b_data = {
        "sentences": [],
        "labels": []
    }
    
    # Define example sentences for each class
    class_sentences = {
        0: ["This is a news article about current events.", 
            "Breaking news: new developments in politics.",
            "The newspaper reported on the economic situation."],
        1: ["The movie was excellent and I enjoyed it.",
            "This product exceeded my expectations.",
            "The service at the restaurant was outstanding."],
        2: ["The technical manual explains how to use the API.",
            "This documentation covers machine learning concepts.",
            "The programming tutorial shows how to implement algorithms."],
        3: ["Today's weather is sunny with a chance of rain.",
            "The forecast predicts snow for the weekend.",
            "Temperatures will rise throughout the week."],
        4: ["General information about various topics.",
            "Miscellaneous content that doesn't fit other categories.",
            "Random facts about the world around us."]
    }
    
    # Sentiment words for Task B
    sentiment_words = {
        "positive": ["excellent", "amazing", "great", "good", "wonderful"],
        "negative": ["terrible", "awful", "bad", "poor", "disappointing"],
        "neutral": ["average", "okay", "neutral", "standard", "regular"]
    }
    
    # Ensure balanced class distribution for Task A
    samples_per_class = num_samples // num_classes
    
    for class_idx in range(num_classes):
        for _ in range(samples_per_class):
            # Get a base sentence for this class
            base_sentence = random.choice(class_sentences[class_idx])
            words = base_sentence.split()
            
            # Slightly modify the sentence for variety
            if len(words) > 3:
                idx = random.randint(0, len(words) - 1)
                words[idx] = random.choice(["amazing", "good", "bad", "neutral", "interesting"])
            
            task_a_sentence = " ".join(words)
            task_a_data["sentences"].append(task_a_sentence)
            task_a_data["labels"].append(class_idx)
    
    # Generate balanced sentiment data for Task B
    sentiment_types = ["positive", "negative", "neutral"]
    samples_per_sentiment = num_samples // len(sentiment_types)
    
    for sentiment_type in sentiment_types:
        for _ in range(samples_per_sentiment):
            # Choose a random base sentence
            base_sentence = random.choice([s for class_sentences in class_sentences.values() for s in class_sentences])
            words = base_sentence.split()
            
            # Modify for sentiment
            if len(words) > 3:
                idx = random.randint(0, len(words) - 1)
                words[idx] = random.choice(sentiment_words[sentiment_type])
            
            task_b_sentence = " ".join(words)
            
            # Assign sentiment score
            if sentiment_type == "positive":
                sentiment_score = random.uniform(0.7, 1.0)
            elif sentiment_type == "negative":
                sentiment_score = random.uniform(0.0, 0.3)
            else:
                sentiment_score = random.uniform(0.4, 0.6)
            
            task_b_data["sentences"].append(task_b_sentence)
            task_b_data["labels"].append(sentiment_score)
    
    # If needed, add more samples to reach the desired total
    remaining_samples = num_samples - (samples_per_class * num_classes)
    for _ in range(remaining_samples):
        # Task A extra samples
        class_idx = random.randint(0, num_classes - 1)
        task_a_sentence = random.choice(class_sentences[class_idx])
        task_a_data["sentences"].append(task_a_sentence)
        task_a_data["labels"].append(class_idx)
        
        # Task B extra samples
        sentiment_type = random.choice(sentiment_types)
        task_b_sentence = random.choice([s for class_sentences in class_sentences.values() for s in class_sentences])
        
        if sentiment_type == "positive":
            sentiment_score = random.uniform(0.7, 1.0)
        elif sentiment_type == "negative":
            sentiment_score = random.uniform(0.0, 0.3)
        else:
            sentiment_score = random.uniform(0.4, 0.6)
        
        task_b_data["sentences"].append(task_b_sentence)
        task_b_data["labels"].append(sentiment_score)
    
    return {
        "task_a_data": task_a_data,
        "task_b_data": task_b_data
    }