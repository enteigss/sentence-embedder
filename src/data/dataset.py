# Training function for multi-task learning
def train_multitask_model(model, train_dataloader, val_dataloader=None, epochs=3, learning_rate=2e-5, task_weights=None):
    """
    Train the multi-task model.
    
    Args:
        model: The multi-task model
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        task_weights: Dictionary mapping task names to their weights in the loss
        
    Returns:
        Trained model
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Define loss function and optimizer
    criterion = MultiTaskLoss(task_weights=task_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    training_stats = {
        "epoch_losses": [],
        "task_a_losses": [],
        "task_b_losses": [],
        "task_a_metrics": {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": []
        },
        "task_b_metrics": {
            "mse": [],
            "rmse": [],
            "r2": []
        },
        "val_losses": [] if val_dataloader else None,
        "val_task_a_metrics": {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": []
        } if val_dataloader else None,
        "val_task_b_metrics": {
            "mse": [],
            "rmse": [],
            "r2": []
        } if val_dataloader else None,
    }
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        task_a_losses = 0.0
        task_b_losses = 0.0

        # Accumulate metrics
        task_a_preds = []
        task_a_true = []
        task_b_preds = []
        task_b_true = []

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            # Process Task A data
            task_a_input_ids = batch["task_a_input_ids"].to(device)
            task_a_attention_mask = batch["task_a_attention_mask"].to(device)
            task_a_token_type_ids = batch.get("task_a_token_type_ids")
            if task_a_token_type_ids is not None:
                task_a_token_type_ids = task_a_token_type_ids.to(device)
            task_a_label = batch["task_a_label"].to(device)
            
            # Process Task B data
            task_b_input_ids = batch["task_b_input_ids"].to(device)
            task_b_attention_mask = batch["task_b_attention_mask"].to(device)
            task_b_token_type_ids = batch.get("task_b_token_type_ids")
            if task_b_token_type_ids is not None:
                task_b_token_type_ids = task_b_token_type_ids.to(device)
            task_b_label = batch["task_b_label"].to(device)
            
            # Forward pass for Task A
            task_a_outputs = model(
                input_ids=task_a_input_ids,
                attention_mask=task_a_attention_mask,
                token_type_ids=task_a_token_type_ids,
                task="task_a"
            )
            
            # Forward pass for Task B
            task_b_outputs = model(
                input_ids=task_b_input_ids,
                attention_mask=task_b_attention_mask,
                token_type_ids=task_b_token_type_ids,
                task="task_b"
            )
            
            # Combine outputs
            outputs = {
                "task_a": task_a_outputs["task_a"],
                "task_b": task_b_outputs["task_b"]
            }
            
            # Combine targets
            targets = {
                "task_a": task_a_label,
                "task_b": task_b_label
            }
            
            # Calculate loss
            optimizer.zero_grad()
            loss, task_losses = criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            task_a_losses += task_losses["task_a"].item()
            task_b_losses += task_losses["task_b"].item()

            # Calculate metrics
            # Task A
            task_a_logits = outputs["task_a"]
            task_a_probs = F.softmax(task_a_logits, dim=1)
            task_a_pred = torch.argmax(task_a_probs, dim=1)
            task_a_preds.extend(task_a_pred.cpu().numpy())
            task_a_true.extend(task_a_label.cpu().numpy())

            # Task B
            task_b_pred = outputs["task_b"].squeeze()
            if task_b_pred.dim() == 0:
                task_b_pred = task_b_pred.unsqueeze(0)
            task_b_preds.extend(task_b_pred.cpu().detach().numpy())
            task_b_true.extend(task_b_label.cpu().numpy())

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "task_a_loss": f"{task_losses['task_a'].item():.4f}",
                "task_b_loss": f"{task_losses['task_b'].item():.4f}"
            })
        
        # Print epoch statistics
        task_a_accuracy = accuracy_score(task_a_true, task_a_preds)
        task_a_precision, task_a_recall, task_a_f1, _ = precision_recall_fscore_support(
            task_a_true, task_a_preds, average='weighted', zero_division=0)
        
        task_b_mse = mean_squared_error(task_b_true, task_b_preds)
        task_b_rmse = np.sqrt(task_b_mse)
        task_b_r2 = r2_score(task_b_true, task_b_preds)

        avg_loss = total_loss / len(train_dataloader)
        avg_task_a_loss = task_a_losses / len(train_dataloader)
        avg_task_b_loss = task_b_losses / len(train_dataloader)

        training_stats["epoch_losses"].append(avg_loss)
        training_stats["task_a_losses"].append(avg_task_a_loss)
        training_stats["task_b_losses"].append(avg_task_b_loss)
        training_stats["task_a_metrics"]["accuracy"].append(task_a_accuracy)
        training_stats["task_a_metrics"]["precision"].append(task_a_precision)
        training_stats["task_a_metrics"]["recall"].append(task_a_recall)
        training_stats["task_a_metrics"]["f1"].append(task_a_f1)
        training_stats["task_b_metrics"]["mse"].append(task_b_mse)
        training_stats["task_b_metrics"]["rmse"].append(task_b_rmse)
        training_stats["task_b_metrics"]["r2"].append(task_b_r2)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Total Loss: {avg_loss:.4f}")
        print(f"  Task A Loss: {avg_task_a_loss:.4f}")
        print(f"  Task B Loss: {avg_task_b_loss:.4f}")
        
        if val_dataloader is not None:
            model.eval()
            val_loss = 0.0
            val_task_a_loss = 0.0
            val_task_b_loss = 0.0

            task_a_preds = []
            task_a_true = []
            task_b_preds = []
            task_b_true = []
            
            with torch.no_grad():
                for batch in val_dataloader:
                    # Process Task A data
                    task_a_input_ids = batch["task_a_input_ids"].to(device)
                    task_a_attention_mask = batch["task_a_attention_mask"].to(device)
                    task_a_token_type_ids = batch.get("task_a_token_type_ids")
                    if task_a_token_type_ids is not None:
                        task_a_token_type_ids = task_a_token_type_ids.to(device)
                    task_a_label = batch["task_a_label"].to(device)
                    
                    # Process Task B data
                    task_b_input_ids = batch["task_b_input_ids"].to(device)
                    task_b_attention_mask = batch["task_b_attention_mask"].to(device)
                    task_b_token_type_ids = batch.get("task_b_token_type_ids")
                    if task_b_token_type_ids is not None:
                        task_b_token_type_ids = task_b_token_type_ids.to(device)
                    task_b_label = batch["task_b_label"].to(device)
                    
                    # Forward passes
                    task_a_outputs = model(
                        input_ids=task_a_input_ids,
                        attention_mask=task_a_attention_mask,
                        token_type_ids=task_a_token_type_ids,
                        task="task_a"
                    )
                    
                    task_b_outputs = model(
                        input_ids=task_b_input_ids,
                        attention_mask=task_b_attention_mask,
                        token_type_ids=task_b_token_type_ids,
                        task="task_b"
                    )
                    
                    # Combine outputs and targets
                    outputs = {
                        "task_a": task_a_outputs["task_a"],
                        "task_b": task_b_outputs["task_b"]
                    }
                    
                    targets = {
                        "task_a": task_a_label,
                        "task_b": task_b_label
                    }
                    
                    # Calculate loss
                    loss, task_losses = criterion(outputs, targets)
                    
                    # Update statistics
                    val_loss += loss.item()
                    val_task_a_loss += task_losses["task_a"].item()
                    val_task_b_loss += task_losses["task_b"].item()

                    # Calculate metrics
                    # Task A
                    task_a_logits = outputs["task_a"]
                    task_a_probs = F.softmax(task_a_logits, dim=1)
                    task_a_pred = torch.argmax(task_a_probs, dim=1)
                    task_a_preds.extend(task_a_pred.cpu().numpy())
                    task_a_true.extend(task_a_label.cpu().numpy())

                    # Task B
                    task_b_pred = outputs["task_b"].squeeze()
                    if task_b_pred.dim() == 0:
                        task_b_pred = task_b_pred.unsqueeze(0)
                    task_b_preds.extend(task_b_pred.cpu().detach().numpy())
                    task_b_true.extend(task_b_label.cpu().numpy())

            # Print epoch statistics
            task_a_accuracy = accuracy_score(task_a_true, task_a_preds)
            task_a_precision, task_a_recall, task_a_f1, _ = precision_recall_fscore_support(
            task_a_true, task_a_preds, average='weighted', zero_division=0)
        
            task_b_mse = mean_squared_error(task_b_true, task_b_preds)
            task_b_rmse = np.sqrt(task_b_mse)
            task_b_r2 = r2_score(task_b_true, task_b_preds)
            
            # Calculate average validation losses
            avg_val_loss = val_loss / len(val_dataloader)
            avg_val_task_a_loss = val_task_a_loss / len(val_dataloader)
            avg_val_task_b_loss = val_task_b_loss / len(val_dataloader)
            
            # Store validation statistics
            if training_stats["val_losses"] is None:
                training_stats["val_losses"] = []
            training_stats["val_losses"].append(avg_val_loss)

            # Store validation metrics
            if training_stats["val_task_a_metrics"] is None:
                training_stats["val_task_a_metrics"] = {
                    "accuracy": [],
                    "precision": [],
                    "recall": [],
                    "f1": []
                }

            if training_stats["val_task_b_metrics"] is None:
                training_stats["val_task_b_metrics"] = {
                    "mse": [],
                    "rmse": [],
                    "r2": []
                }

            # Append validation metrics
            training_stats["val_task_a_metrics"]["accuracy"].append(task_a_accuracy)
            training_stats["val_task_a_metrics"]["precision"].append(task_a_precision)
            training_stats["val_task_a_metrics"]["recall"].append(task_a_recall)
            training_stats["val_task_a_metrics"]["f1"].append(task_a_f1)

            training_stats["val_task_b_metrics"]["mse"].append(task_b_mse)
            training_stats["val_task_b_metrics"]["rmse"].append(task_b_rmse)
            training_stats["val_task_b_metrics"]["r2"].append(task_b_r2)

            
            
            # Print validation statistics
            print(f"  Val Loss: {avg_val_loss:.4f}")
            print(f"  Val Task A Loss: {avg_val_task_a_loss:.4f}")
            print(f"  Val Task B Loss: {avg_val_task_b_loss:.4f}")
    
    return model, training_stats