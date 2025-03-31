import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.models import MultiTaskModel
from src.data import MultiTaskDataset
from src.utils import create_synthetic_data, train_multitask_model

def main():
    # Create synthetic data
    num_samples = 200
    print(f"Creating synthetic data with {num_samples} samples...")
    synthetic_data = create_synthetic_data(num_samples=num_samples)
    
    # Split into train and validation
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    
    train_data = {
        "task_a_data": {
            "sentences": synthetic_data["task_a_data"]["sentences"][:train_size],
            "labels": synthetic_data["task_a_data"]["labels"][:train_size]
        },
        "task_b_data": {
            "sentences": synthetic_data["task_b_data"]["sentences"][:train_size],
            "labels": synthetic_data["task_b_data"]["labels"][:train_size]
        }
    }
    
    val_data = {
        "task_a_data": {
            "sentences": synthetic_data["task_a_data"]["sentences"][train_size:],
            "labels": synthetic_data["task_a_data"]["labels"][train_size:]
        },
        "task_b_data": {
            "sentences": synthetic_data["task_b_data"]["sentences"][train_size:],
            "labels": synthetic_data["task_b_data"]["labels"][train_size:]
        }
    }
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Create datasets
    train_dataset = MultiTaskDataset(
        train_data["task_a_data"],
        train_data["task_b_data"],
        tokenizer,
        max_length=128
    )
    
    val_dataset = MultiTaskDataset(
        val_data["task_a_data"],
        val_data["task_b_data"],
        tokenizer,
        max_length=128
    )
    
    print(f"Created datasets: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
    
    # Create data loaders
    batch_size = 8
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize multi-task model
    model = MultiTaskModel(
        model_name="bert-base-uncased",
        pooling_strategy="mean",
        task_a_num_classes=5,
        task_b_type="sentiment"
    )
    
    # Set task weights
    task_weights = {"task_a": 1.0, "task_b": 0.5}
    print(f"Training with task weights: {task_weights}")
    
    # Train model
    epochs = 3 
    model, training_stats = train_multitask_model(
        model,
        train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        learning_rate=2e-5,
        task_weights=task_weights
    )

    print(f"  Task A accuracy: {training_stats['task_a_metrics']['accuracy'][-1]:.4f}")
    print(f"  Task B RMSE: {training_stats['task_b_metrics']['rmse'][-1]:.4f}")
    
    if training_stats["val_task_a_metrics"] is not None:
        print("Final validation metrics:")
        print(f"  Task A accuracy: {training_stats['val_task_a_metrics']['accuracy'][-1]:.4f}")
        print(f"  Task B RMSE: {training_stats['val_task_b_metrics']['rmse'][-1]:.4f}")
