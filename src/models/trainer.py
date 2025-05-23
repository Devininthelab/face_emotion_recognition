import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from datetime import datetime
import numpy as np


class Trainer:
    def __init__(
        self,
        model,
        train_dataset,
        eval_dataset=None,
        collate_fn=None,
        batch_size=8,
        lr=1e-2,
        weight_decay=0.05,
        num_epochs=10,
        device=None,
        save_path=None,
        run_name="run",
        num_workers=2,
        export_to_onnx=False
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.save_path = save_path
        self.best_accuracy = 0.0
        self.run_name = run_name
        self.device = device
        self.best_model_path = None
        self.model.to(self.device)
        self.num_workers = num_workers
        self.export_to_onnx = export_to_onnx
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=self.num_workers
        )
        self.eval_loader = (
            DataLoader(self.eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=self.num_workers)
            if self.eval_dataset else None
        )

        wandb.init(
            project="video-classification",
            name=run_name,
            config={
                "epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "architecture": model.__class__.__name__,
                "dataset": train_dataset.__class__.__name__,
                "device": self.device,
                "weight_decay": weight_decay,
                "train_size": len(train_dataset),
                "eval_size": len(eval_dataset) if eval_dataset else 0
            }
        )
        wandb.watch(self.model, log="all", log_freq=50)

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            for videos, labels in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
                #Skip pushing videos to device # [B, T, C, H, W]
                labels = labels.to(self.device)              # [B]

                logits = self.model(videos)
                
                loss = self.criterion(logits, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                correct += (logits.argmax(dim=-1) == labels).sum().item()
                total += labels.size(0)

            accuracy = correct / total
            avg_loss = total_loss / len(self.train_loader)

            print(f"\nTrain Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
            wandb.log({
                "epoch": epoch,
                "train/loss": avg_loss,
                "train/accuracy": accuracy
            })

            if self.eval_loader:
                self.evaluate(epoch)
        
        if self.best_model_path and self.export_to_onnx:
            self.export_to_onnx()

        
    def evaluate(self, epoch):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for videos, labels in tqdm(self.eval_loader, desc="Evaluating"):
                # Skip videos without pushing to device
                labels = labels.to(self.device)

                logits = self.model(videos)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                correct += (logits.argmax(dim=-1) == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        avg_loss = total_loss / len(self.eval_loader)

        print(f"\nEval Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        wandb.log({
            "epoch": epoch,
            "eval/loss": avg_loss,
            "eval/accuracy": accuracy
        })

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            if self.save_path:
                os.makedirs(self.save_path, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.best_model_path = os.path.join(self.save_path, f"model_{self.run_name}_{timestamp}.pt")
                torch.save(self.model.state_dict(), self.best_model_path)
                wandb.save(self.best_model_path)
                print(f"New best model saved to {self.best_model_path}")

    def export_to_onnx(self, path=None):
        if path is not None:
            self.best_model_path = path
        print(f"\nExporting best model to ONNX...")
        dummy_input = list(torch.randint(0, 256,  (16, 3, 13, 57), dtype=torch.uint8))
        self.model.load_state_dict(torch.load(self.best_model_path))
        self.model.eval()
        onnx_path = self.best_model_path.replace(".pt", ".onnx")

        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

        wandb.save(onnx_path)
        print(f"ONNX model saved to {onnx_path} and logged to W&B.")