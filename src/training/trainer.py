"""
Training Module
================
Complete training pipeline for CattleGNN with:
- Triplet loss with online hard negative mining
- Cosine annealing learning rate schedule
- Early stopping
- Comprehensive logging and checkpointing
- Validation-based model selection
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import ensure_dirs, save_stats


class Trainer:
    """
    Training pipeline for CattleGNN model.
    """
    
    def __init__(self, model, loss_fn, config, device='cpu'):
        """
        Args:
            model: CattleGNN model
            loss_fn: Loss function (TripletLossWithMining or CombinedLoss)
            config: Configuration dict
            device: Computation device
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.config = config
        self.device = device

        train_cfg = config['training']

        # ── Mixed-precision (AMP) ─────────────────────────────────────────
        self.use_amp = train_cfg.get('use_amp', False) and torch.cuda.is_available()
        # RTX 5070 (Blackwell sm_120) natively supports bfloat16 — no overflow, no GradScaler needed
        if self.use_amp and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
            self.use_scaler = False   # bf16 is numerically stable; GradScaler unnecessary
        else:
            self.amp_dtype = torch.float16
            self.use_scaler = self.use_amp  # fp16 needs scaler
        self.scaler = GradScaler('cuda', enabled=self.use_scaler)
        if self.use_amp:
            print(f"[INFO] AMP enabled: {self.amp_dtype} | GradScaler: {self.use_scaler}")

        # ── torch.compile ─────────────────────────────────────────────────
        compile_model = train_cfg.get('compile_model', False)
        if compile_model and hasattr(torch, 'compile') and torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print("[INFO] torch.compile enabled (mode=reduce-overhead)")
            except Exception as e:
                print(f"[WARN] torch.compile failed: {e}. Continuing without compilation.")
        
        # Optimizer
        if train_cfg['optimizer'] == 'adamw':
            self.optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=train_cfg['learning_rate'],
                weight_decay=train_cfg['weight_decay']
            )
        else:
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=train_cfg['learning_rate'],
                weight_decay=train_cfg['weight_decay']
            )
        
        # Scheduler
        sched_cfg = train_cfg.get('scheduler', {})
        if sched_cfg.get('type') == 'cosine':
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2,
                eta_min=sched_cfg.get('min_lr', 1e-6)
            )
        elif sched_cfg.get('type') == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5
            )
        else:
            self.scheduler = None
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_active_triplets': [],
            'learning_rates': [],
            'val_rank1_acc': [],
            'epoch_times': [],
        }
        
        # Directories
        self.checkpoint_dir = train_cfg.get('checkpoint_dir', 'outputs/checkpoints')
        ensure_dirs(self.checkpoint_dir)
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_active = 0
        total_triplets = 0
        num_batches = 0
        
        for batch in train_loader:
            batch = batch.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # ── Forward pass with AMP (bfloat16 on Blackwell) ─────────────
            with autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                output = self.model(batch)
                embeddings = output['embedding']
                labels = batch.y

                if hasattr(self.loss_fn, 'ce_weight') and 'logits' in output:
                    loss, stats = self.loss_fn(embeddings, output['logits'], labels)
                else:
                    loss, stats = self.loss_fn(embeddings, labels)

            # ── Backward ──────────────────────────────────────────────────
            if loss.requires_grad:
                grad_clip = self.config['training'].get('grad_clip', 1.0)
                if self.use_scaler:          # fp16 path
                    self.scaler.scale(loss).backward()
                    if grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:                        # bf16 / fp32 path — no scaler needed
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.optimizer.step()

            total_loss += loss.item()
            total_active += stats.get('active_triplets', 0)
            total_triplets += stats.get('total_triplets', 0)
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        active_ratio = total_active / max(total_triplets, 1)
        
        return {
            'loss': avg_loss,
            'active_triplets': total_active,
            'total_triplets': total_triplets,
            'active_ratio': active_ratio,
        }
    
    @torch.no_grad()
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        all_embeddings = []
        all_labels = []
        
        for batch in val_loader:
            batch = batch.to(self.device, non_blocking=True)
            
            output = self.model(batch)
            embeddings = output['embedding']
            labels = batch.y
            
            # Compute loss
            if hasattr(self.loss_fn, 'ce_weight') and 'logits' in output:
                loss, stats = self.loss_fn(embeddings, output['logits'], labels)
            else:
                loss, stats = self.loss_fn(embeddings, labels)
            
            total_loss += loss.item()
            num_batches += 1
            
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Compute Rank-1 accuracy
        if all_embeddings:
            all_embeddings = torch.cat(all_embeddings)
            all_labels = torch.cat(all_labels)
            rank1_acc = self._compute_rank1_accuracy(all_embeddings, all_labels)
        else:
            rank1_acc = 0.0
        
        return {
            'loss': avg_loss,
            'rank1_accuracy': rank1_acc,
        }
    
    def _compute_rank1_accuracy(self, embeddings, labels):
        """Compute Rank-1 identification accuracy."""
        # Cosine similarity matrix
        sim_matrix = torch.mm(embeddings, embeddings.t())
        
        # Set diagonal to -inf (don't match with self)
        sim_matrix.fill_diagonal_(-float('inf'))
        
        # For each query, find the nearest neighbor
        nn_indices = sim_matrix.argmax(dim=1)
        nn_labels = labels[nn_indices]
        
        correct = (nn_labels == labels).float().mean().item()
        return correct
    
    def train(self, train_loader, val_loader=None, epochs=None):
        """
        Full training loop.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            epochs: Number of epochs (defaults to config)
            
        Returns:
            history: Training history dict
        """
        if epochs is None:
            epochs = self.config['training']['epochs']

        early_stop_cfg = self.config['training'].get('early_stopping', {})
        patience = early_stop_cfg.get('patience', 15)
        min_delta = early_stop_cfg.get('min_delta', 0.001)

        # GPU info banner
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            total_vram = props.total_memory / 1024**3
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            print(f"\n{'=' * 60}")
            print(f"Starting Training: {epochs} epochs")
            print(f"Device: {self.device} | GPU: {gpu_name} ({total_vram:.1f} GB VRAM)")
            print(f"CUDA Capability: sm_{props.major}{props.minor} | SM Count: {props.multi_processor_count}")
            print(f"AMP: {self.use_amp} dtype={getattr(self, 'amp_dtype', 'fp32')} | GradScaler: {self.use_scaler}")
            print(f"Batch size: {self.config['training']['batch_size']}")
            print(f"{'=' * 60}\n")
        else:
            print(f"\n{'=' * 60}")
            print(f"Starting Training: {epochs} epochs")
            print(f"Device: {self.device}")
            print(f"{'=' * 60}\n")
        
        for epoch in range(1, epochs + 1):
            self.epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_stats = self.train_epoch(train_loader)
            
            # Validate
            val_stats = {'loss': 0, 'rank1_accuracy': 0}
            if val_loader is not None:
                val_stats = self.validate(val_loader)
            
            # Update scheduler
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_stats['loss'])
                else:
                    self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            # Record history
            self.history['train_loss'].append(train_stats['loss'])
            self.history['val_loss'].append(val_stats['loss'])
            self.history['train_active_triplets'].append(train_stats['active_ratio'])
            self.history['learning_rates'].append(current_lr)
            self.history['val_rank1_acc'].append(val_stats['rank1_accuracy'])
            self.history['epoch_times'].append(epoch_time)
            
            # Print progress
            if torch.cuda.is_available():
                used_vram = torch.cuda.memory_allocated(0) / 1024**3
                peak_vram = torch.cuda.max_memory_allocated(0) / 1024**3
                vram_str = f" | VRAM: {used_vram:.2f}/{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB (peak {peak_vram:.2f}GB)"
                torch.cuda.reset_peak_memory_stats()  # reset each epoch for per-epoch peaks
            else:
                vram_str = ""
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_stats['loss']:.4f} | "
                f"Val Loss: {val_stats['loss']:.4f} | "
                f"R1 Acc: {val_stats['rank1_accuracy']:.4f} | "
                f"Active: {train_stats['active_ratio']:.2f} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {epoch_time:.1f}s"
                + vram_str,
                flush=True
            )
            
            # Checkpointing (save best model)
            if val_stats['rank1_accuracy'] > self.best_val_acc + min_delta:
                self.best_val_acc = val_stats['rank1_accuracy']
                self.best_val_loss = val_stats['loss']
                self.patience_counter = 0
                self._save_checkpoint('best_model.pt', val_stats)
                print(f"  -> New best model! R1 Acc: {self.best_val_acc:.4f}", flush=True)
            else:
                self.patience_counter += 1
            
            # Periodic checkpointing
            save_every = self.config['training'].get('save_every', 10)
            if epoch % save_every == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch}.pt', val_stats)
            
            # Early stopping
            if self.patience_counter >= patience:
                print(f"\n[INFO] Early stopping at epoch {epoch} (patience={patience})")
                break
        
        # Save final model
        self._save_checkpoint('final_model.pt', val_stats)
        
        # Save training history
        self._save_history()
        
        print(f"\n{'=' * 60}")
        print(f"Training Complete!")
        print(f"  Best Validation R1 Accuracy: {self.best_val_acc:.4f}")
        print(f"  Best Validation Loss: {self.best_val_loss:.4f}")
        print(f"  Total Epochs: {epoch}")
        print(f"  Total Time: {sum(self.history['epoch_times']):.1f}s")
        print(f"{'=' * 60}")
        
        return self.history
    
    def _save_checkpoint(self, filename, val_stats):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'val_stats': val_stats,
            'history': self.history,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, filepath):
        """Load model from checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint.get('history', self.history)
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"[INFO] Loaded checkpoint from epoch {self.epoch}")
        return checkpoint.get('val_stats', {})
    
    def _save_history(self):
        """Save training history to JSON."""
        history_path = os.path.join(self.checkpoint_dir, 'training_history.json')
        
        # Convert to serializable format
        serializable_history = {}
        for key, values in self.history.items():
            serializable_history[key] = [float(v) for v in values]
        
        serializable_history['best_val_acc'] = float(self.best_val_acc)
        serializable_history['best_val_loss'] = float(self.best_val_loss)
        serializable_history['total_epochs'] = self.epoch
        serializable_history['total_time'] = sum(self.history['epoch_times'])
        
        save_stats(serializable_history, history_path)
        print(f"[INFO] Training history saved to {history_path}")
