# -*- coding: utf-8 -*-
"""
Cell Type Classifier - Simplified Wrapper Version
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix, top_k_accuracy_score
from tqdm.auto import tqdm
import math

USE_ATTENTION = True      # Use attention mechanism
USE_MIXUP = True         # Use MixUp
USE_CUTMIX = True        # Use CutMix
USE_SAM = True           # Use SAM optimizer
USE_CONTRASTIVE = True   # Use contrastive learning
USE_TTA = True           # Test-time augmentation

_rng = np.random.default_rng()


class CellTypeClassifier:
    """Cell Type Classifier - Simple Wrapper"""
    
    def __init__(self, model, device, classes, norm_meta, config):
        """
        Args:
            model: Initialized model
            device: torch device
            classes: List of classes
            norm_meta: Normalization metadata
            config: Configuration parameter dictionary (contains BATCH_SIZE, LR, WEIGHT_DECAY, etc.)
        """
        self.model = model
        self.device = device
        self.classes = classes
        self.num_classes = len(classes)
        self.norm_meta = norm_meta
        self.config = config
        
    def setup_training(self, weights):
        """Setup loss function and optimizer"""
        # Loss function
        self.criterion = PolyLoss(
            weight=weights.to(self.device), 
            epsilon=2.0, 
            label_smoothing=self.config['LABEL_SMOOTH']
        )
        
        # Contrastive learning loss
        self.contrastive_loss = SupConLoss(
            temperature=self.config['CONTRASTIVE_TEMP']
        ) if self.config['USE_CONTRASTIVE'] else None
        
        # Optimizer
        if self.config['USE_SAM']:
            self.optimizer = SAM(
                self.model.parameters(), 
                torch.optim.AdamW, 
                lr=self.config['LR'], 
                weight_decay=self.config['WEIGHT_DECAY'], 
                rho=self.config['SAM_RHO']
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.config['LR'], 
                weight_decay=self.config['WEIGHT_DECAY']
            )
            
    def setup_scheduler(self, train_loader):
        """Setup learning rate scheduler"""
        base_opt = self.optimizer.base_optimizer if self.config['USE_SAM'] else self.optimizer
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            base_opt,
            max_lr=self.config['LR'] * 10,
            epochs=self.config['MAX_EPOCHS'],
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
    def mk_loader(self, Xnp, ynp, shuffle=False):
        """Create DataLoader"""
        ds = TensorDataset(
            torch.from_numpy(Xnp), 
            torch.from_numpy(ynp).long()
        )
        return DataLoader(
            ds, 
            batch_size=self.config['BATCH_SIZE'], 
            shuffle=shuffle, 
            num_workers=0, 
            pin_memory=False
        )
        
    @torch.no_grad()
    def evaluate(self, loader, desc="Eval", need_report=False):
        """Evaluation function"""
        self.model.eval()
        all_logits, all_true = [], []
        
        pbar = tqdm(loader, desc=desc, unit="batch", leave=False)
        for xb, yb in pbar:
            xb = xb.to(device=self.device, dtype=torch.float32)
            logits = self.model(xb)
            all_logits.append(logits.cpu().numpy())
            all_true.append(yb.numpy())
        
        logits = np.concatenate(all_logits, 0)
        y_true = np.concatenate(all_true, 0)
        y_pred = logits.argmax(1)
        
        topk = min(self.config['TOPK'], self.num_classes)
        metrics = {
            "acc": accuracy_score(y_true, y_pred),
            "bal_acc": balanced_accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average='macro'),
            "f1_weighted": f1_score(y_true, y_pred, average='weighted'),
            f"top{topk}_acc": top_k_accuracy_score(y_true, logits, k=topk)
        }
        
        if need_report:
            metrics["report"] = classification_report(y_true, y_pred, target_names=self.classes, digits=3)
            metrics["cm"] = confusion_matrix(y_true, y_pred)
        
        return metrics
    
    def train_epoch(self, train_loader, epoch):
        """Train one epoch"""
        self.model.train()
        running_loss, running_cls_loss, running_con_loss = 0.0, 0.0, 0.0
        n_seen = 0
        
        # Decide data augmentation strategy
        use_mixup = self.config['USE_MIXUP'] and (epoch < self.config['MAX_EPOCHS'] * 0.8)
        use_cutmix = self.config['USE_CUTMIX'] and (epoch < self.config['MAX_EPOCHS'] * 0.8)
        use_contrastive = self.config['USE_CONTRASTIVE'] and (epoch > 5)
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.config['MAX_EPOCHS']}", unit="batch", leave=True)
        
        for batch_idx, (xb, yb) in enumerate(train_bar):
            xb = xb.to(device=self.device, dtype=torch.float32)
            yb = yb.to(self.device)
            
            # Select data augmentation
            if use_mixup and _rng.random() > 0.5:
                xb_aug, ya, yb_aug, lam = mixup_data(xb, yb, alpha=self.config['MIXUP_ALPHA'])
                aug_type = "mix"
            elif use_cutmix and _rng.random() > 0.5:
                xb_aug, ya, yb_aug, lam = cutmix_data(xb, yb, alpha=self.config['CUTMIX_ALPHA'])
                aug_type = "cut"
            else:
                xb_aug, ya, yb_aug, lam = xb, yb, yb, 1.0
                aug_type = "none"
            
            if self.config['USE_SAM']:
                # SAM first step
                logits, features = self.model(xb_aug, return_features=True)
                
                if lam == 1.0:
                    cls_loss = self.criterion(logits, ya)
                else:
                    cls_loss = mixup_criterion(self.criterion, logits, ya, yb_aug, lam)
                
                con_loss = self.contrastive_loss(features, ya) if use_contrastive else 0
                loss = cls_loss + self.config['CONTRASTIVE_WEIGHT'] * con_loss
                
                loss.backward()
                self.optimizer.first_step(zero_grad=True)
                
                # SAM second step
                logits, features = self.model(xb_aug, return_features=True)
                
                if lam == 1.0:
                    cls_loss = self.criterion(logits, ya)
                else:
                    cls_loss = mixup_criterion(self.criterion, logits, ya, yb_aug, lam)
                
                con_loss = self.contrastive_loss(features, ya) if use_contrastive else 0
                loss = cls_loss + self.config['CONTRASTIVE_WEIGHT'] * con_loss
                
                loss.backward()
                self.optimizer.second_step(zero_grad=True)
            else:
                self.optimizer.zero_grad(set_to_none=True)
                
                logits, features = self.model(xb_aug, return_features=True)
                
                if lam == 1.0:
                    cls_loss = self.criterion(logits, ya)
                else:
                    cls_loss = mixup_criterion(self.criterion, logits, ya, yb_aug, lam)
                
                con_loss = self.contrastive_loss(features, ya) if use_contrastive else 0
                loss = cls_loss + self.config['CONTRASTIVE_WEIGHT'] * con_loss
                
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Update statistics
            bs = yb.size(0)
            n_seen += bs
            running_loss += loss.item() * bs
            running_cls_loss += cls_loss.item() * bs
            if use_contrastive:
                running_con_loss += con_loss.item() * bs if isinstance(con_loss, torch.Tensor) else 0
            
            # Update progress bar
            current_lr = self.scheduler.get_last_lr()[0]
            avg_loss = running_loss / n_seen
            avg_cls_loss = running_cls_loss / n_seen
            avg_con_loss = running_con_loss / n_seen if use_contrastive else 0
            
            train_bar.set_postfix({
                'loss': f"{avg_loss:.4f}",
                'cls': f"{avg_cls_loss:.4f}",
                'con': f"{avg_con_loss:.4f}" if use_contrastive else "off",
                'lr': f"{current_lr:.2e}",
                'aug': aug_type
            })
        
        return avg_loss
    
    def train(self, X_tr, y_tr, X_va, y_va, save_path):
        """Complete training process"""
        train_loader = self.mk_loader(X_tr, y_tr, shuffle=True)
        val_loader = self.mk_loader(X_va, y_va, shuffle=False)
        
        self.setup_scheduler(train_loader)
        
        best_f1 = -1.0
        best_epoch = 0
        patience_cnt = 0
        history = {"train_loss": [], "val_acc": [], "val_f1": []}
        
        print("\n[Start Training]")
        print("=" * 60)
        
        for epoch in range(1, self.config['MAX_EPOCHS'] + 1):
            # Training phase
            avg_loss = self.train_epoch(train_loader, epoch)
            history["train_loss"].append(avg_loss)
            
            # Validation phase
            val_metrics = self.evaluate(val_loader, desc=f"Valid {epoch}", need_report=False)
            history["val_acc"].append(val_metrics["acc"])
            history["val_f1"].append(val_metrics["f1_macro"])
            
            # Check if improved
            improved = val_metrics["f1_macro"] > best_f1 + 1e-4
            
            if improved:
                best_f1 = val_metrics["f1_macro"]
                best_epoch = epoch
                patience_cnt = 0
                
                # Save model
                torch.save({
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict() if not self.config['USE_SAM'] else None,
                    "scheduler": self.scheduler.state_dict(),
                    "best_f1": best_f1,
                    "classes": self.classes,
                    "norm_meta": self.norm_meta,
                    "config": self.config,
                    "history": history
                }, save_path)
                
                topk = min(self.config['TOPK'], self.num_classes)
                print(f"[Epoch {epoch:3d}] "
                      f"Loss: {avg_loss:.4f} | "
                      f"Val Acc: {val_metrics['acc']:.4f} | "
                      f"Bal Acc: {val_metrics['bal_acc']:.4f} | "
                      f"F1: {val_metrics['f1_macro']:.4f} | "
                      f"Top-{topk}: {val_metrics[f'top{topk}_acc']:.4f} "
                      f"✓ BEST")
            else:
                patience_cnt += 1
                topk = min(self.config['TOPK'], self.num_classes)
                print(f"[Epoch {epoch:3d}] "
                      f"Loss: {avg_loss:.4f} | "
                      f"Val Acc: {val_metrics['acc']:.4f} | "
                      f"Bal Acc: {val_metrics['bal_acc']:.4f} | "
                      f"F1: {val_metrics['f1_macro']:.4f} | "
                      f"Top-{topk}: {val_metrics[f'top{topk}_acc']:.4f} "
                      f"({patience_cnt}/{self.config['PATIENCE']})")
                
                if patience_cnt >= self.config['PATIENCE']:
                    print(f"\nEarly stopping triggered! Best Epoch: {best_epoch}, Best F1: {best_f1:.4f}")
                    break
        
        return history
    
    def test(self, X_te, y_te):
        """Test model"""
        test_loader = self.mk_loader(X_te, y_te, shuffle=False)
        
        print("\nRunning standard test...")
        test_metrics = self.evaluate(test_loader, desc="Test", need_report=True)
        
        topk = min(self.config['TOPK'], self.num_classes)
        print("\nTest Results (Standard):")
        print(f"  Accuracy:     {test_metrics['acc']:.4f}")
        print(f"  Balanced Acc: {test_metrics['bal_acc']:.4f}")
        print(f"  F1 Macro:   {test_metrics['f1_macro']:.4f}")
        print(f"  F1 Weighted:{test_metrics['f1_weighted']:.4f}")
        print(f"  Top-{topk}:      {test_metrics[f'top{topk}_acc']:.4f}")
        
        # TTA test
        if self.config.get('USE_TTA', False):
            print("\nRunning test-time augmentation (TTA)...")
            tta_metrics = self.evaluate_tta(X_te, y_te)
            
            print("\nTest Results (TTA):")
            print(f"  Accuracy:     {tta_metrics['acc']:.4f} ({tta_metrics['acc']-test_metrics['acc']:+.4f})")
            print(f"  Balanced Acc: {tta_metrics['bal_acc']:.4f} ({tta_metrics['bal_acc']-test_metrics['bal_acc']:+.4f})")
            print(f"  F1 Macro:   {tta_metrics['f1_macro']:.4f} ({tta_metrics['f1_macro']-test_metrics['f1_macro']:+.4f})")
        
        print("\nDetailed Classification Report:")
        print(test_metrics["report"])
        
        return test_metrics
    
    @torch.no_grad()
    def evaluate_tta(self, X_test, y_test):
        """TTA evaluation"""
        self.model.eval()
        n_aug = self.config.get('TTA_NUM', 5)
        noise_std = self.config.get('TTA_NOISE', 0.01)
        all_preds = []
        
        for aug_idx in tqdm(range(n_aug), desc="TTA", leave=False):
            X_aug = X_test + _rng.standard_normal(X_test.shape).astype(np.float32) * noise_std
            X_aug = self.norm(X_aug)
            
            logits = []
            for i in range(0, len(X_aug), self.config['BATCH_SIZE']):
                batch = torch.from_numpy(X_aug[i:i+self.config['BATCH_SIZE']]).to(self.device)
                logits.append(self.model(batch).cpu().numpy())
            
            all_preds.append(np.concatenate(logits))
        
        avg_logits = np.mean(all_preds, axis=0)
        y_pred = avg_logits.argmax(1)
        
        topk = min(self.config['TOPK'], self.num_classes)
        return {
            "acc": accuracy_score(y_test, y_pred),
            "bal_acc": balanced_accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average='macro'),
            f"top{topk}_acc": top_k_accuracy_score(y_test, avg_logits, k=topk)
        }
    
    @torch.no_grad()
    def predict_proba(self, Xnp):
        """Predict probabilities"""
        self.model.eval()
        batch_size = max(8192, self.config['BATCH_SIZE'])
        proba_list = []
        
        for i in tqdm(range(0, len(Xnp), batch_size), desc="Predict [all]", unit="batch", leave=False):
            xb = torch.from_numpy(Xnp[i:i+batch_size]).to(device=self.device, dtype=torch.float32)
            logits = self.model(xb)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            proba_list.append(proba)
            
        return np.vstack(proba_list)
    
    def norm(self, X):
        """Normalization function"""
        if self.norm_meta["type"] == "scale":
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.mean_ = self.norm_meta["mean"]
            scaler.scale_ = self.norm_meta["scale"]
            scaler.var_ = scaler.scale_ ** 2
            return scaler.transform(X)
        elif self.norm_meta["type"] == "l2":
            def l2norm(a, eps=1e-12): 
                return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), eps)
            return l2norm(X)
        else:
            return X
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        print(f"✓ Loaded model from {checkpoint_path}")
        return checkpoint


# ========== Helper classes/functions needed from original code ==========

class PolyLoss(nn.Module):
    def __init__(self, weight=None, epsilon=2.0, label_smoothing=0.05):
        super().__init__()
        self.weight = weight
        self.epsilon = epsilon
        self.label_smoothing = label_smoothing
        
    def forward(self, logits, target):
        ce = F.cross_entropy(
            logits, target, weight=self.weight, 
            reduction='none', label_smoothing=self.label_smoothing
        )
        with torch.no_grad():
            p = torch.softmax(logits, dim=1)
            pt = p.gather(1, target.unsqueeze(1)).squeeze(1)
        poly = ce + self.epsilon * (1 - pt)
        return poly.mean()


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        
        features = F.normalize(features, dim=1)
        similarity = torch.matmul(features, features.T) / self.temperature
        
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask
        
        exp_logits = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        loss = -mean_log_prob_pos.mean()
        
        return loss


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["e_w"] = p.grad * scale
                p.add_(self.state[p]["e_w"])
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()
        
    def _grad_norm(self):
        return torch.norm(
            torch.stack([
                p.grad.norm(p=2)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ])
        )


def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = _rng.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = _rng.beta(alpha, alpha)
    else:
        return x, y, y, 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    feat_size = x.size(1)
    cut_size = int(feat_size * (1 - lam))

    if cut_size > 0:
        start_idx = _rng.integers(0, feat_size - cut_size + 1)
        x = x.clone()
        x[:, start_idx:start_idx + cut_size] = x[index, start_idx:start_idx + cut_size]

    return x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# Basic modules
class CosineClassifier(nn.Module):
    def __init__(self, in_dim, num_classes, scale=20.0):
        super().__init__()
        self.W = nn.Parameter(torch.empty(num_classes, in_dim))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, x):
        x = F.normalize(x, dim=1)
        Wn = F.normalize(self.W, dim=1)
        return self.scale * (x @ Wn.t())

class ResMLPBlock(nn.Module):
    def __init__(self, dim, expansion=2, dropout=0.2):
        super().__init__()
        hidden = dim * expansion
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.BatchNorm1d(dim)
        )
        
    def forward(self, x):
        return x + self.net(x)

class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x_in = x.unsqueeze(1)
        attn_out, _ = self.attention(x_in, x_in, x_in)
        x = self.norm(x + self.dropout(attn_out.squeeze(1)))
        return x

# Main model
class AttentionMLPHead(nn.Module):
    def __init__(self, in_dim=50, num_classes=10, hidden=512, depth=3, 
                 num_heads=4, p=0.2, cosine_head=True):
        super().__init__()
        
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(p)
        )
        
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(ResMLPBlock(hidden, expansion=2, dropout=p))
            if USE_ATTENTION and i % 2 == 0:
                self.blocks.append(SelfAttentionBlock(hidden, num_heads, p))
        
        self.final_norm = nn.LayerNorm(hidden)
        self.hidden_dim = hidden
        
        if cosine_head:
            self.head = CosineClassifier(hidden, num_classes)
        else:
            self.head = nn.Linear(hidden, num_classes)
    
    def get_features(self, x):
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)
    
    def forward(self, x, return_features=False):
        features = self.get_features(x)
        logits = self.head(features)
        if return_features:
            return logits, features
        return logits

# =============== Loss Functions ===============
class PolyLoss(nn.Module):
    def __init__(self, weight=None, epsilon=2.0, label_smoothing=0.05):
        super().__init__()
        self.weight = weight
        self.epsilon = epsilon
        self.label_smoothing = label_smoothing
        
    def forward(self, logits, target):
        ce = F.cross_entropy(
            logits, target, weight=self.weight, 
            reduction='none', label_smoothing=self.label_smoothing
        )
        with torch.no_grad():
            p = torch.softmax(logits, dim=1)
            pt = p.gather(1, target.unsqueeze(1)).squeeze(1)
        poly = ce + self.epsilon * (1 - pt)
        return poly.mean()

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        
        features = F.normalize(features, dim=1)
        similarity = torch.matmul(features, features.T) / self.temperature
        
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask
        
        exp_logits = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        loss = -mean_log_prob_pos.mean()
        
        return loss

# =============== SAM Optimizer ===============
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["e_w"] = p.grad * scale
                p.add_(self.state[p]["e_w"])
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()
        
    def _grad_norm(self):
        return torch.norm(
            torch.stack([
                p.grad.norm(p=2)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ])
        )

# =============== Data Augmentation ===============
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = _rng.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = _rng.beta(alpha, alpha)
    else:
        return x, y, y, 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    feat_size = x.size(1)
    cut_size = int(feat_size * (1 - lam))

    if cut_size > 0:
        start_idx = _rng.integers(0, feat_size - cut_size + 1)
        x = x.clone()
        x[:, start_idx:start_idx + cut_size] = x[index, start_idx:start_idx + cut_size]

    return x, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)