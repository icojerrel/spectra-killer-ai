"""
CNN Model Trainer
Implements PyTorch CNN training for trading pattern recognition
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class TradingDataset(Dataset):
    """Dataset class for trading data"""
    
    def __init__(self, data: pd.DataFrame, sequence_length: int = 60, 
                 feature_columns: List[str] = None, target_column: str = 'target_15m'):
        self.data = data
        self.sequence_length = sequence_length
        self.feature_columns = feature_columns or self._get_default_features()
        self.target_column = target_column
        
        # Prepare features and targets
        self.features = self.data[self.feature_columns].values
        self.targets = self.data[self.target_column].values
        
        # Normalize features
        self.feature_means = np.mean(self.features, axis=0)
        self.feature_stds = np.std(self.features, axis=0)
        self.feature_stds[self.feature_stds == 0] = 1.0  # Avoid division by zero
        
        # Normalize
        self.features = (self.features - self.feature_means) / self.feature_stds
        
        self.valid_indices = self._get_valid_indices()
        
    def _get_default_features(self) -> List[str]:
        """Default feature columns"""
        return [
            'open', 'high', 'low', 'close', 'volume',
            'price_change', 'rsi', 'ema_5', 'ema_20', 'bb_position',
            'macd', 'atr', 'volatility_5', 'volatility_15',
            'hour', 'day_of_week', 'is_doji', 'body_size'
        ]
    
    def _get_valid_indices(self) -> np.ndarray:
        """Get valid indices for sequence data"""
        return np.arange(self.sequence_length, len(self.data))
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        
        # Get sequence
        start_idx = actual_idx - self.sequence_length
        sequence = self.features[start_idx:actual_idx]
        
        # Get target
        target = self.targets[actual_idx]
        
        # Convert to tensors
        sequence_tensor = torch.FloatTensor(sequence.T)  # Shape: [features, sequence_length]
        target_tensor = torch.tensor(target, dtype=torch.float32)
        
        return sequence_tensor, target_tensor
    
    def get_normalization_params(self) -> Dict:
        """Get normalization parameters"""
        return {
            'feature_means': self.feature_means.tolist(),
            'feature_stds': self.feature_stds.tolist(),
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }


class TradingCNN(nn.Module):
    """CNN model for trading pattern recognition"""
    
    def __init__(self, input_features: int = 17, sequence_length: int = 60):
        super(TradingCNN, self).__init__()
        
        self.input_features = input_features
        self.sequence_length = sequence_length
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(input_features, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        
        # Calculate flattened size
        conv_output_size = self._get_conv_output_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)
        
        # Classification head
        self.classification_head = nn.Linear(256, 3)  # -1, 0, 1
            
        # Regression head
        self.regression_head = nn.Linear(256, 1)  # Continuous return
        
        self.relu = nn.ReLU()
        
    def _get_conv_output_size(self):
        """Calculate output size after convolutions"""
        x = torch.randn(1, self.input_features, self.sequence_length)
        
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        
        return x.numel()
    
    def forward(self, x):
        # Convolutional layers
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        
        # Multiple outputs
        classification_output = self.classification_head(x)
        regression_output = self.regression_head(x)
        
        return {
            'classification': classification_output,
            'regression': regression_output,
            'features': x
        }


class CNNModelTrainer:
    """CNN Model Training Pipeline"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Training on device: {self.device}")
        
    def _get_default_config(self) -> Dict:
        """Default training configuration"""
        return {
            'model': {
                'input_features': 17,
                'sequence_length': 60,
                'dropout_rate': 0.3,
            },
            'training': {
                'batch_size': 64,
                'learning_rate': 0.001,
                'epochs': 100,
                'patience': 15,
                'min_delta': 1e-6,
                'weight_decay': 1e-5,
            },
            'data': {
                'validation_split': 0.2,
                'sequence_length': 60,
                'target_column': 'target_15m',
            },
            'checkpointing': {
                'save_best_model': True,
                'checkpoint_dir': 'models/checkpoints',
                'model_save_path': 'models/cnn_trading_model.pth',
                'validation_threshold': 0.60,
            }
        }
    
    def setup_model(self, num_features: int, sequence_length: int):
        """Setup model, optimizer, and scheduler"""
        self.model = TradingCNN(num_features, sequence_length)
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=7,
            verbose=True
        )
        
        # Loss functions
        self.classification_criterion = nn.CrossEntropyLoss()
        self.regression_criterion = nn.MSELoss()
        
    def prepare_data_loaders(self, train_data: pd.DataFrame, 
                            val_data: pd.DataFrame) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data loaders"""
        
        # Create datasets
        train_dataset = TradingDataset(
            train_data,
            sequence_length=self.config['data']['sequence_length'],
            target_column=self.config['data']['target_column']
        )
        
        val_dataset = TradingDataset(
            val_data,
            sequence_length=self.config['data']['sequence_length'],
            target_column=self.config['data']['target_column']
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        classification_loss = 0.0
        regression_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (sequences, targets) in enumerate(train_loader):
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)
            
            # Regression targets
            reg_targets = targets.float()
            
            # Classification targets (-1, 0, 1 -> 0, 1, 2)
            cls_targets = ((targets + 1).long())
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(sequences)
            
            # Calculate losses
            cls_loss = self.classification_criterion(outputs['classification'], cls_targets)
            reg_loss = self.regression_criterion(outputs['regression'].squeeze(), reg_targets)
            
            # Combined loss
            combined_loss = 0.7 * cls_loss + 0.3 * reg_loss
            
            # Backward pass
            combined_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += combined_loss.item()
            classification_loss += cls_loss.item()
            regression_loss += reg_loss.item()
            
            # Accuracy
            _, predicted = torch.max(outputs['classification'], 1)
            total += cls_targets.size(0)
            correct += (predicted == cls_targets).sum().item()
            
            if batch_idx % 100 == 0:
                logger.info(
                    f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                    f'Loss: {combined_loss.item():.4f} '
                    f'Acc: {100.0 * correct / total:.2f}%'
                )
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return {
            'total_loss': avg_loss,
            'classification_loss': classification_loss / len(train_loader),
            'regression_loss': regression_loss / len(train_loader),
            'accuracy': accuracy
        }
    
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict:
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0.0
        classification_loss = 0.0
        regression_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                # Regression targets
                reg_targets = targets.float()
                
                # Classification targets
                cls_targets = ((targets + 1).long())
                
                # Forward pass
                outputs = self.model(sequences)
                
                # Calculate losses
                cls_loss = self.classification_criterion(outputs['classification'], cls_targets)
                reg_loss = self.regression_criterion(outputs['regression'].squeeze(), reg_targets)
                
                # Combined loss
                combined_loss = 0.7 * cls_loss + 0.3 * reg_loss
                
                # Statistics
                total_loss += combined_loss.item()
                classification_loss += cls_loss.item()
                regression_loss += reg_loss.item()
                
                # Accuracy
                _, predicted = torch.max(outputs['classification'], 1)
                total += cls_targets.size(0)
                correct += (predicted == cls_targets).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return {
            'total_loss': avg_loss,
            'classification_loss': classification_loss / len(val_loader),
            'regression_loss': regression_loss / len(val_loader),
            'accuracy': accuracy
        }
    
    def train_model(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict:
        """Main training loop"""
        logger.info("Starting CNN model training...")
        
        # Setup model
        self.setup_model(
            num_features=len(self._get_feature_columns(train_data)),
            sequence_length=self.config['data']['sequence_length']
        )
        
        # Prepare data loaders
        train_loader, val_loader = self.prepare_data_loaders(train_data, val_data)
        
        # Training tracking
        best_val_accuracy = 0.0
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        # Create checkpoint directory
        checkpoint_dir = Path(self.config['checkpointing']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        training_history = []
        
        for epoch in range(self.config['training']['epochs']):
            logger.info(f'\nEpoch {epoch + 1}/{self.config["training"]["epochs"]}')
            logger.info('=' * 50)
            
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader, epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['total_loss'])
            
            # Log metrics
            logger.info(f'Train Loss: {train_metrics["total_loss"]:.4f}, Acc: {train_metrics["accuracy"]:.4f}')
            logger.info(f'Val Loss: {val_metrics["total_loss"]:.4f}, Acc: {val_metrics["accuracy"]:.4f}')
            
            # Save training history
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['total_loss'],
                'train_accuracy': train_metrics['accuracy'],
                'val_loss': val_metrics['total_loss'],
                'val_accuracy': val_metrics['accuracy'],
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
            
            # Check for improvement
            if val_metrics['val_accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['val_accuracy']
                best_val_loss = val_metrics['total_loss']
                epochs_without_improvement = 0
                
                # Save best model
                if self.config['checkpointing']['save_best_model']:
                    self._save_checkpoint(checkpoint_dir / 'best_model.pth', epoch, val_metrics, training_history)
                    logger.info(f'New best validation accuracy: {best_val_accuracy:.4f}')
            else:
                epochs_without_improvement += 1
            
            # Early stopping
            if epochs_without_improvement >= self.config['training']['patience']:
                logger.info(f'Early stopping triggered after {epoch + 1} epochs')
                break
        
        # Final model save
        final_model_path = self.config['checkpointing']['model_save_path']
        self._save_checkpoint(final_model_path, epoch, val_metrics, training_history)
        
        logger.info(f'Training completed. Best validation accuracy: {best_val_accuracy:.4f}')
        logger.info(f'Model saved to: {final_model_path}')
        
        return {
            'best_val_accuracy': best_val_accuracy,
            'best_val_loss': best_val_loss,
            'training_history': training_history,
            'epochs_trained': epoch + 1,
            'model_path': final_model_path
        }
    
    def _save_checkpoint(self, path: Path, epoch: int, metrics: Dict, history: List):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': epoch,
            'val_metrics': metrics,
            'training_history': history,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, path)
    
    def load_model(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f'Model loaded from {checkpoint_path}')
        logger.info(f"Epoch {checkpoint['epoch']}, Val Acc: {checkpoint['val_metrics']['accuracy']:.4f}")
        
        return checkpoint
    
    def _get_feature_columns(self, data: pd.DataFrame) -> List[str]:
        """Get feature columns from data"""
        default_features = [
            'open', 'high', 'low', 'close', 'volume',
            'price_change', 'rsi', 'ema_5', 'ema_20', 'bb_position',
            'macd', 'atr', 'volatility_5', 'volatility_15',
            'hour', 'day_of_week', 'is_doji', 'body_size'
        ]
        
        available_features = [col for col in default_features if col in data.columns]
        
        if len(available_features) < len(default_features):
            missing = set(default_features) - set(available_features)
            logger.warning(f"Missing features: {missing}")
        
        logger.info(f"Using {len(available_features)} features for training")
        return available_features
    
    def evaluate_model(self, test_loader: DataLoader) -> Dict:
        """Evaluate model on test data"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sequences, targets in test_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                # Regression targets
                reg_targets = targets.float()
                
                # Classification targets
                cls_targets = ((targets + 1).long())
                
                # Forward pass
                outputs = self.model(sequences)
                
                # Combined loss
                cls_loss = self.classification_criterion(outputs['classification'], cls_targets)
                reg_loss = self.regression_criterion(outputs['regression'].squeeze(), reg_targets)
                combined_loss = 0.7 * cls_loss + 0.3 * reg_loss
                
                total_loss += combined_loss.item()
                
                # Predictions
                _, predicted = torch.max(outputs['classification'], 1)
                
                # Convert back to -1, 0, 1
                predicted_original = predicted.float() - 1
                
                all_predictions.extend(predicted_original.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # Accuracy
                total += cls_targets.size(0)
                correct += (predicted == cls_targets).sum().item()
        
        accuracy = correct / total
        avg_loss = total_loss / len(test_loader)
        
        # Calculate additional metrics
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # Trading-specific metrics
        buy_accuracy = np.mean((predictions == 1) & (targets == 1))
        sell_accuracy = np.mean((predictions == -1) & (targets == -1))
        hold_accuracy = np.mean((predictions == 0) & (targets == 0))
        
        return {
            'test_loss': avg_loss,
            'test_accuracy': accuracy,
            'buy_accuracy': buy_accuracy,
            'sell_accuracy': sell_accuracy,
            'hold_accuracy': hold_accuracy,
            'total_samples': len(all_predictions)
        }
