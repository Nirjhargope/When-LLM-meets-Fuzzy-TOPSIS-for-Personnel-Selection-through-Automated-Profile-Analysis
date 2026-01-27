import torch
from sklearn.metrics import classification_report, hamming_loss
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
num_labels = len(set(labels))  

# Function to evaluate the model and compute metrics
def evaluate_model(model, val_dataloader, device, num_labels):
    model.eval()  
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
           
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
          
            _, preds = torch.max(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
           
            probs = torch.softmax(logits, dim=1)
            all_probs.extend(probs.cpu().numpy())
    
  
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Classification Report
    target_names = [f"Class {i}" for i in range(num_labels)]
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm)
    
    # Hamming Loss
    hamming = hamming_loss(all_labels, all_preds)
    print(f"Hamming Loss: {hamming:.4f}")
evaluate_model(model, val_dataloader, device, num_labels)

# Plotting Confusion Matrix

def plot_confusion_matrix(cm):
    plt.figure(figsize=(6,6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="coolwarm",  
        cbar=True,
        xticklabels=["Class 0","Class 1","Class 2"], 
        yticklabels=["Class 0","Class 1","Class 2"]
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()


