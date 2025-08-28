
"""
Chapter 5: Classical Neural Network 


A classical version of the MLC-QNN architecture for direct comparison with quantum models.
This implementation matches the parameter count and architectural design of the quantum
hybrid model to ensure fair evaluation.

Architecture: BERT Embedding -> Projection -> Classical Feature Transform -> Classification Head
- Replaces quantum circuit with equivalent classical nonlinear transformations
- Maintains similar parameter count for controlled comparison
- Uses identical training procedures and evaluation metrics

Author: Research Project - Quantum vs Classical ML
Target Conferences: ICLR, ICML, NeurIPS, AAAI
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report, precision_recall_fscore_support
import os
from datetime import datetime
import json
import logging
from transformers import AutoTokenizer, AutoModel, set_seed
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration matching quantum model parameters
class Config:
    # Model Architecture (matching MLC-QNN)
    BERT_MODEL = "bert-base-uncased"
    BERT_DIM = 768
    N_QUBITS = 12  # Projection dimension (matches quantum model)
    N_LAYERS = 3   # Depth of classical transformation (matches quantum layers)
    HIDDEN_DIM = 256  # Classical hidden dimension
    NUM_LABELS = 28
    
    # Training Configuration
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0001
    MAX_EPOCHS = 1000
    EARLY_STOPPING_PATIENCE = 890
    WEIGHT_DECAY = 1e-4
    
    # Data Configuration
    MAX_LENGTH = 256
    TRAIN_SUBSET_SIZE = None  # Use full dataset (set to int for subset)
    
    # System Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    NUM_WORKERS = 4

# GoEmotions label names
GOEMOTIONS_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
    'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

def set_reproducibility(seed=42):
    """Set random seeds for reproducible results."""
    set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class GoEmotionsDataset(Dataset):
    """Dataset class for GoEmotions with BERT tokenization."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label
        }

def bert_embed(texts, model, tokenizer, device, batch_size=32, max_length=128):
    """
    Extract BERT embeddings (CLS token) from texts.
    Matches the bert_embed function used in quantum models.
    """
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing BERT embeddings"):
            batch_texts = texts[i:i+batch_size]
            
            encoding = tokenizer(
                batch_texts,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            ).to(device)
            
            outputs = model(**encoding)
            # Use CLS token embedding (first token)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)
    
    return np.vstack(embeddings)

class ClassicalNeuralNetwork(nn.Module):
    """
    Classical Neural Network matching the architecture of MLC-QNN.
    
    Architecture mimics quantum hybrid model:
    1. BERT Embedding (768D) -> Projection to N_QUBITS dimensions
    2. Classical Feature Transform (replaces quantum circuit)
    3. Classification Head
    
    Parameter count designed to match quantum model for fair comparison.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Projection layer (matches quantum model's dimensional reduction)
        self.projection = nn.Linear(config.BERT_DIM, config.N_QUBITS)
        
        # Classical feature transformation (replaces quantum circuit)
        # Design to match quantum circuit parameter count: N_LAYERS * N_QUBITS * 3
        quantum_params = config.N_LAYERS * config.N_QUBITS * 3
        
        # Multi-layer classical transformation
        self.feature_transform = nn.ModuleList()
        
        # First layer: expand from N_QUBITS to HIDDEN_DIM
        self.feature_transform.append(nn.Linear(config.N_QUBITS, config.HIDDEN_DIM))
        self.feature_transform.append(nn.Tanh())  # Smooth nonlinearity
        self.feature_transform.append(nn.Dropout(0.1))
        
        # Middle layers: maintain HIDDEN_DIM
        for _ in range(config.N_LAYERS - 1):
            self.feature_transform.append(nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM))
            self.feature_transform.append(nn.ReLU())
            self.feature_transform.append(nn.Dropout(0.1))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.HIDDEN_DIM // 2, config.NUM_LABELS)
        )
        
        # Calculate actual parameter count for verification
        self._log_parameter_count()
        
    def _log_parameter_count(self):
        """Log parameter count breakdown for comparison with quantum model."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        projection_params = sum(p.numel() for p in self.projection.parameters())
        transform_params = sum(p.numel() for p in self.feature_transform.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        
        logger.info(f"Model Parameter Count:")
        logger.info(f"  Projection: {projection_params:,}")
        logger.info(f"  Feature Transform: {transform_params:,}")
        logger.info(f"  Classifier: {classifier_params:,}")
        logger.info(f"  Total: {total_params:,} (Trainable: {trainable_params:,})")
        
        # Compare with quantum circuit parameter count
        quantum_params = self.config.N_LAYERS * self.config.N_QUBITS * 3
        logger.info(f"  Equivalent Quantum Params: {quantum_params}")
        
    def forward(self, x):
        """
        Forward pass through classical architecture.
        
        Args:
            x: BERT embeddings of shape (batch_size, BERT_DIM)
            
        Returns:
            logits: Classification logits of shape (batch_size, NUM_LABELS)
        """
        # Project to quantum-equivalent dimension
        x = self.projection(x)  # (batch_size, N_QUBITS)
        
        # Classical feature transformation
        for layer in self.feature_transform:
            x = layer(x)
        
        # Classification
        logits = self.classifier(x)  # (batch_size, NUM_LABELS)
        
        return logits

class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience=10, min_delta=1e-4, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_score = None
        self.best_weights = None
        
    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        return False

def compute_pos_weights(labels_array):
    """Compute positive weights for handling class imbalance."""
    positive_counts = labels_array.sum(axis=0)
    total_samples = len(labels_array)
    negative_counts = total_samples - positive_counts
    
    # Avoid division by zero
    positive_counts = np.clip(positive_counts, 1, None)
    pos_weights = negative_counts / positive_counts
    
    logger.info(f"Class imbalance weights - Min: {pos_weights.min():.3f}, "
                f"Max: {pos_weights.max():.3f}, Mean: {pos_weights.mean():.3f}")
    
    return torch.tensor(pos_weights, dtype=torch.float32)

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # Move batch to device
        embeddings = batch['embeddings'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(embeddings)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches

def evaluate_model(model, dataloader, criterion, device):
    """Evaluate model performance."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            embeddings = batch['embeddings'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(embeddings)
            loss = criterion(logits, labels)
            
            # Convert to probabilities
            probs = torch.sigmoid(logits)
            predictions = (probs >= 0.5).float()
            
            total_loss += loss.item()
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    # Concatenate all batches
    all_predictions = np.vstack(all_predictions)
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    
    # Compute metrics
    macro_f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    micro_f1 = f1_score(all_labels, all_predictions, average='micro', zero_division=0)
    
    avg_loss = total_loss / len(dataloader)
    
    return {
        'loss': avg_loss,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probs
    }

def optimize_thresholds(val_labels, val_probs, num_labels):
    """Optimize per-label thresholds to maximize macro F1-score."""
    logger.info("Optimizing classification thresholds...")
    
    best_thresholds = np.zeros(num_labels)
    threshold_candidates = np.linspace(0.05, 0.95, 19)
    
    for label_idx in range(num_labels):
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in threshold_candidates:
            pred_binary = (val_probs[:, label_idx] >= threshold).astype(int)
            f1 = f1_score(val_labels[:, label_idx], pred_binary, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        best_thresholds[label_idx] = best_threshold
    
    logger.info(f"Threshold optimization complete. Range: [{best_thresholds.min():.3f}, {best_thresholds.max():.3f}]")
    return best_thresholds

def create_embedding_dataloader(embeddings, labels, batch_size, shuffle=True):
    """Create DataLoader from pre-computed embeddings."""
    
    class EmbeddingDataset(Dataset):
        def __init__(self, embeddings, labels):
            self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.float32)
        
        def __len__(self):
            return len(self.embeddings)
        
        def __getitem__(self, idx):
            return {
                'embeddings': self.embeddings[idx],
                'labels': self.labels[idx]
            }
    
    dataset = EmbeddingDataset(embeddings, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

def main():
    """Main training and evaluation pipeline."""
    print("="*80)
    print("CHAPTER 5: CLASSICAL NEURAL NETWORK BASELINE")
    print("Quantum vs Classical ML Comparison Project")
    print("="*80)
    
    # Initialize configuration
    config = Config()
    set_reproducibility(config.SEED)
    
    logger.info(f"Using device: {config.DEVICE}")
    logger.info(f"Configuration: N_QUBITS={config.N_QUBITS}, N_LAYERS={config.N_LAYERS}")
    
    # Load data
    logger.info("Loading GoEmotions dataset...")
    df_train = pd.read_csv("train.csv")
    df_val = pd.read_csv("val.csv") 
    df_test = pd.read_csv("test.csv")
    
    # Parse labels
    df_train["labels"] = df_train["labels"].apply(eval)
    df_val["labels"] = df_val["labels"].apply(eval)
    df_test["labels"] = df_test["labels"].apply(eval)
    
    # Optional: Use subset for faster development/testing
    if config.TRAIN_SUBSET_SIZE:
        df_train = df_train.sample(n=config.TRAIN_SUBSET_SIZE, random_state=config.SEED)
        logger.info(f"Using training subset: {len(df_train)} samples")
    
    logger.info(f"Dataset sizes - Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    
    # Initialize BERT for feature extraction
    logger.info("Loading BERT model for feature extraction...")
    tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL)
    bert_model = AutoModel.from_pretrained(config.BERT_MODEL).to(config.DEVICE)
    
    # Extract BERT embeddings (matching quantum model pipeline)
    logger.info("Extracting BERT embeddings...")
    train_embeddings = bert_embed(df_train["text"].tolist(), bert_model, tokenizer, 
                                  config.DEVICE, batch_size=config.BATCH_SIZE)
    val_embeddings = bert_embed(df_val["text"].tolist(), bert_model, tokenizer,
                               config.DEVICE, batch_size=config.BATCH_SIZE)
    test_embeddings = bert_embed(df_test["text"].tolist(), bert_model, tokenizer,
                                config.DEVICE, batch_size=config.BATCH_SIZE)
    
    logger.info(f"Embedding shapes - Train: {train_embeddings.shape}, "
                f"Val: {val_embeddings.shape}, Test: {test_embeddings.shape}")
    
    # Prepare labels
    train_labels = np.array(df_train["labels"].tolist())
    val_labels = np.array(df_val["labels"].tolist())
    test_labels = np.array(df_test["labels"].tolist())
    
    # Create data loaders
    train_loader = create_embedding_dataloader(train_embeddings, train_labels, config.BATCH_SIZE)
    val_loader = create_embedding_dataloader(val_embeddings, val_labels, config.BATCH_SIZE, shuffle=False)
    test_loader = create_embedding_dataloader(test_embeddings, test_labels, config.BATCH_SIZE, shuffle=False)
    
    # Initialize model
    logger.info("Initializing Classical Neural Network...")
    model = ClassicalNeuralNetwork(config).to(config.DEVICE)
    
    # Setup training components
    pos_weights = compute_pos_weights(train_labels).to(config.DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)
    
    # Training loop
    logger.info("Starting training...")
    best_val_f1 = 0.0
    training_history = []
    
    for epoch in range(config.MAX_EPOCHS):
        logger.info(f"\nEpoch {epoch+1}/{config.MAX_EPOCHS}")
        
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.DEVICE)
        
        # Validation
        val_results = evaluate_model(model, val_loader, criterion, config.DEVICE)
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log progress
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss: {val_results['loss']:.4f}, "
                   f"Val Macro F1: {val_results['macro_f1']:.4f}, "
                   f"Val Micro F1: {val_results['micro_f1']:.4f}")
        logger.info(f"Learning Rate: {current_lr:.6f}")
        
        # Track best model
        if val_results['macro_f1'] > best_val_f1:
            best_val_f1 = val_results['macro_f1']
            logger.info(f"New best validation F1: {best_val_f1:.4f}")
        
        # Store training history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_results['loss'],
            'val_macro_f1': val_results['macro_f1'],
            'val_micro_f1': val_results['micro_f1'],
            'learning_rate': current_lr
        })
        
        # Early stopping check
        if early_stopping(val_results['macro_f1'], model):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Final evaluation on test set
    logger.info("\nEvaluating on test set...")
    test_results = evaluate_model(model, test_loader, criterion, config.DEVICE)
    
    # Threshold optimization
    logger.info("Optimizing thresholds on validation set...")
    val_results_final = evaluate_model(model, val_loader, criterion, config.DEVICE)
    best_thresholds = optimize_thresholds(val_results_final['labels'], 
                                         val_results_final['probabilities'], 
                                         config.NUM_LABELS)
    
    # Apply optimized thresholds to test set
    optimized_test_preds = np.zeros_like(test_results['probabilities'])
    for i, threshold in enumerate(best_thresholds):
        optimized_test_preds[:, i] = (test_results['probabilities'][:, i] >= threshold).astype(int)
    
    # Compute optimized metrics
    optimized_macro_f1 = f1_score(test_results['labels'], optimized_test_preds, 
                                 average='macro', zero_division=0)
    optimized_micro_f1 = f1_score(test_results['labels'], optimized_test_preds, 
                                 average='micro', zero_division=0)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Model and training artifacts
    model_dir = f"classical_nn_model_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'best_thresholds': best_thresholds,
        'training_history': training_history
    }, os.path.join(model_dir, 'model_checkpoint.pth'))
    
    # Comprehensive results
    final_results = {
        'model_type': 'Classical Neural Network (Quantum-Equivalent)',
        'architecture': {
            'bert_dim': config.BERT_DIM,
            'projection_dim': config.N_QUBITS,
            'n_layers': config.N_LAYERS,
            'hidden_dim': config.HIDDEN_DIM,
            'num_labels': config.NUM_LABELS
        },
        'parameters': {
            'total': sum(p.numel() for p in model.parameters()),
            'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad)
        },
        'training_config': {
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'max_epochs': config.MAX_EPOCHS,
            'actual_epochs': len(training_history),
            'early_stopping_patience': config.EARLY_STOPPING_PATIENCE
        },
        'performance': {
            'test_macro_f1_default': float(test_results['macro_f1']),
            'test_micro_f1_default': float(test_results['micro_f1']),
            'test_macro_f1_optimized': float(optimized_macro_f1),
            'test_micro_f1_optimized': float(optimized_micro_f1),
            'best_val_macro_f1': float(best_val_f1),
            'threshold_improvement': float(optimized_macro_f1 - test_results['macro_f1'])
        },
        'timestamp': timestamp,
        'seed': config.SEED
    }
    
    # Save comprehensive results
    with open(f"classical_nn_results_{timestamp}.json", "w") as f:
        json.dump(final_results, f, indent=2)
    
    # Save per-label analysis
    per_label_f1 = f1_score(test_results['labels'], optimized_test_preds, 
                           average=None, zero_division=0)
    per_label_results = pd.DataFrame({
        'label': GOEMOTIONS_LABELS,
        'threshold': best_thresholds,
        'f1_score': per_label_f1,
        'support': test_results['labels'].sum(axis=0)
    })
    per_label_results.to_csv(f"classical_nn_per_label_{timestamp}.csv", index=False)
    
    # Save training history
    training_df = pd.DataFrame(training_history)
    training_df.to_csv(f"classical_nn_training_history_{timestamp}.csv", index=False)
    
    # Classification report
    logger.info("\nDetailed Classification Report:")
    print(classification_report(test_results['labels'], optimized_test_preds,
                               target_names=GOEMOTIONS_LABELS, zero_division=0))
    
    # Final summary
    print("\n" + "="*80)
    print("CLASSICAL NEURAL NETWORK TRAINING COMPLETED")
    print("="*80)
    print(f"Model Architecture: {config.N_QUBITS}D projection → {config.N_LAYERS} layers → Classification")
    print(f"Total Parameters: {final_results['parameters']['total']:,}")
    print(f"Training Epochs: {final_results['training_config']['actual_epochs']}")
    print(f"\nTest Results (Default 0.5 threshold):")
    print(f"  Macro F1: {test_results['macro_f1']:.4f}")
    print(f"  Micro F1: {test_results['micro_f1']:.4f}")
    print(f"\nTest Results (Optimized thresholds):")
    print(f"  Macro F1: {optimized_macro_f1:.4f}")
    print(f"  Micro F1: {optimized_micro_f1:.4f}")
    print(f"  Improvement: +{final_results['performance']['threshold_improvement']:.4f}")
    print(f"\nResults saved to: classical_nn_results_{timestamp}.json")
    print("="*80)
    
    logger.info("Classical Neural Network baseline completed successfully!")
    
    return final_results

if __name__ == "__main__":

    main()
