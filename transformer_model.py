import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset
import time
import math

class WeatherTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, dropout=0.2, max_len=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.sensor_proj = nn.Linear(3, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output = nn.Linear(d_model, vocab_size)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, token_ids, sensor_data):
        seq_len = token_ids.size(1)
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        pos_emb = self.pos_embed(positions)
        sensor_ctx = self.sensor_proj(sensor_data).unsqueeze(1)

        x = self.embed(token_ids) + pos_emb + sensor_ctx

        # Create padding mask (ignore padding tokens)
        pad_mask = (token_ids == 0)  # <PAD> token index is 0
        
        # Generate attention mask
        attn_mask = self.generate_square_subsequent_mask(seq_len, token_ids.device)
        
        x = self.transformer(x, mask=attn_mask, src_key_padding_mask=pad_mask)
        return self.output(x)

    def generate_square_subsequent_mask(self, sz, device):
        return torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)

class WeatherDataset(Dataset):
    def __init__(self, sensor_data, token_data):
        self.sensor_data = sensor_data
        self.token_data = token_data
        
    def __len__(self):
        return len(self.sensor_data)
    
    def __getitem__(self, idx):
        return self.sensor_data[idx], self.token_data[idx]

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-6
        )
        
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.lr,
            epochs=config.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy='linear'
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def train_epoch(self):
        self.model.train()
        total_loss, total_correct, total_tokens = 0, 0, 0

        for sensors, tokens in self.train_loader:
            sensors = sensors.float().to(self.device)
            inputs = tokens[:, :-1].to(self.device)
            targets = tokens[:, 1:].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs, sensors)

            loss = self.criterion(
                outputs.reshape(-1, outputs.size(-1)),
                targets.reshape(-1)
            )

            preds = outputs.argmax(dim=-1)
            non_pad_mask = targets != 0
            correct = (preds == targets) & non_pad_mask

            total_correct += correct.sum().item()
            total_tokens += non_pad_mask.sum().item()

            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.config.clip_norm)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader), total_correct / total_tokens if total_tokens > 0 else 0

    def validate(self):
        self.model.eval()
        total_loss, total_correct, total_tokens = 0, 0, 0

        with torch.no_grad():
            for sensors, tokens in self.val_loader:
                sensors = sensors.float().to(self.device)
                inputs = tokens[:, :-1].to(self.device)
                targets = tokens[:, 1:].to(self.device)

                outputs = self.model(inputs, sensors)
                loss = self.criterion(
                    outputs.reshape(-1, outputs.size(-1)),
                    targets.reshape(-1)
                )

                preds = outputs.argmax(dim=-1)
                non_pad_mask = targets != 0
                correct = (preds == targets) & non_pad_mask

                total_correct += correct.sum().item()
                total_tokens += non_pad_mask.sum().item()
                total_loss += loss.item()

        return total_loss / len(self.val_loader), total_correct / total_tokens if total_tokens > 0 else 0

    def train(self, epochs):
        best_loss = float('inf')
        best_acc = 0
        no_improve = 0

        print(f"{'Epoch':^6} | {'Train Loss':^10} | {'Val Loss':^10} | {'Val Acc':^8} | {'LR':^10} | {'Time':^8}")
        print("-" * 65)

        for epoch in range(epochs):
            start = time.time()
            train_loss, _ = self.train_epoch()
            val_loss, val_acc = self.validate()
            elapsed = time.time() - start

            save_flag = ""
            if val_loss < best_loss - self.config.min_delta:
                best_loss = val_loss
                best_acc = val_acc
                torch.save(self.model.state_dict(), self.config.save_path)
                no_improve = 0
                save_flag = "💾"
            else:
                no_improve += 1

            lr = self.optimizer.param_groups[0]['lr']
            print(f"{epoch+1:^6} | {train_loss:^10.4f} | {val_loss:^10.4f} | {val_acc:^8.2%} | {lr:^10.2e} | {elapsed:^8.1f}s {save_flag}")

            if no_improve >= self.config.patience:
                print(f"🚨 Early stopping at epoch {epoch+1}")
                break

        self.model.load_state_dict(torch.load(self.config.save_path))
        print("\n✅ Training complete! Loaded best model for final use")
        print(f"📊 Best Validation Loss: {best_loss:.4f}, Accuracy: {best_acc:.2%}")
        return best_loss, best_acc
