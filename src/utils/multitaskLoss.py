# Combined loss function for multi-task learning
class MultiTaskLoss(nn.Module):
    def __init__(self, task_weights=None):
        """
        Multi-task loss function.
        
        Args:
            task_weights: Dictionary mapping task names to their weights in the combined loss
        """
        super(MultiTaskLoss, self).__init__()
        
        self.task_weights = task_weights or {"task_a": 1.0, "task_b": 1.0}
        
        # Define task-specific loss functions
        self.task_a_loss = nn.CrossEntropyLoss()
        self.task_b_loss = nn.MSELoss()  # For sentiment analysis (regression)
    
    def forward(self, outputs, targets):
        """
        Calculate combined loss for multiple tasks.
        
        Args:
            outputs: Dictionary mapping task names to their model outputs
            targets: Dictionary mapping task names to their target values
            
        Returns:
            Combined loss value
        """
        total_loss = 0.0
        losses = {}
        
        # Task A: Classification Loss
        if "task_a" in outputs and "task_a" in targets:
            task_a_loss = self.task_a_loss(outputs["task_a"], targets["task_a"])
            losses["task_a"] = task_a_loss
            print("Task weights:", self.task_weights)
            total_loss += self.task_weights["task_a"] * task_a_loss
        
        # Task B: Sentiment Analysis Loss
        if "task_b" in outputs and "task_b" in targets:
            
            task_b_output = outputs["task_b"].squeeze()
            task_b_target = targets["task_b"]

            if task_b_output.dim() == 0 and task_b_target.dim() == 1:
                task_b_output = task_b_output.unsqueeze(0)
            elif task_b_output.dim() == 1 and task_b_target.dim() == 0:
                task_b_target = task_b_target.squeeze(0)

            task_b_loss = self.task_b_loss(task_b_output, task_b_target)
            losses["task_b"] = task_b_loss
            total_loss += self.task_weights["task_b"] * task_b_loss
        
        return total_loss, losses