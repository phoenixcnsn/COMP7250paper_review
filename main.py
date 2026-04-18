import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_from_disk
import numpy as np
from tqdm import tqdm
import os
import sys
from transformers import get_cosine_schedule_with_warmup

# ------------------------------
# Configuration
# ------------------------------
MAX_SEQ_LEN = 448
SPECIAL_POS_ID = 511  # CodeBERT position encoding range 0~511
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 1e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "./checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)


# ------------------------------
# Helper functions for dynamic mask and position id generation
# ------------------------------
def get_var_token_length(var, tokenizer, cache={}):
    """Cache token length of variable names to avoid repeated tokenization"""
    if var not in cache:
        cache[var] = len(tokenizer.tokenize(var))
    return cache[var]


def build_var_name_to_pos(input_ids, vars_list, tokenizer, var_len_cache):
    """Build mapping from variable name to list of token positions based on input_ids"""
    seq_len = len(input_ids)
    # Find the first [SEP], after which is the variable region
    try:
        sep_idx = input_ids.index(tokenizer.sep_token_id)
    except ValueError:
        sep_idx = seq_len // 2
    pos = sep_idx + 1
    var_pos = {}
    for var in vars_list:
        var_len = var_len_cache.get(var, get_var_token_length(var, tokenizer, var_len_cache))
        positions = []
        for offset in range(var_len):
            token_pos = pos + offset
            if token_pos < seq_len:
                positions.append(token_pos)
        if positions:
            var_pos[var] = positions
        pos += var_len
        if pos >= seq_len:
            break
    return var_pos


def generate_attention_mask_and_position_ids(input_ids, vars_list, edges, tokenizer, var_len_cache):
    """
    Returns:
        attention_mask: torch.Tensor, shape (seq_len, seq_len), 0 for allowed attention, -inf for masked
        position_ids:   torch.Tensor, shape (seq_len,)
    """
    seq_len = len(input_ids)
    var_name_to_pos = build_var_name_to_pos(input_ids, vars_list, tokenizer, var_len_cache)

    # Initialize mask: all zeros (allow all attention)
    mask = np.zeros((seq_len, seq_len), dtype=np.float32)

    # Collect all variable positions
    all_var_pos = set()
    for pos_list in var_name_to_pos.values():
        all_var_pos.update(pos_list)
    all_var_pos = sorted(all_var_pos)

    if all_var_pos:
        # Build adjacency list for variables
        var_adj = {p: set() for p in all_var_pos}
        for u_name, v_name in edges:
            if u_name in var_name_to_pos and v_name in var_name_to_pos:
                for u_pos in var_name_to_pos[u_name]:
                    for v_pos in var_name_to_pos[v_name]:
                        if u_pos < seq_len and v_pos < seq_len:
                            var_adj[u_pos].add(v_pos)
                            var_adj[v_pos].add(u_pos)

        # Mask variable pairs without data flow edges
        for i in all_var_pos:
            for j in all_var_pos:
                if i != j and j not in var_adj[i]:
                    mask[i, j] = -1e9

    # Position ids: normal 0,1,2,... ; variable token positions set to SPECIAL_POS_ID
    pos_ids = list(range(seq_len))
    for pos_list in var_name_to_pos.values():
        for p in pos_list:
            if p < seq_len:
                pos_ids[p] = SPECIAL_POS_ID

    return torch.tensor(mask, dtype=torch.float32), torch.tensor(pos_ids, dtype=torch.long)


# ------------------------------
# Custom Dataset: dynamically generate masks and position ids
# ------------------------------
class CloneDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, tokenizer):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.var_len_cache = {}  # shared cache for variable name lengths

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids1 = item["input_ids1"]
        input_ids2 = item["input_ids2"]
        vars1 = item["vars1"]
        vars2 = item["vars2"]
        edges1 = item["edges1"]
        edges2 = item["edges2"]
        label = item["label"]

        # Dynamically generate attention masks and position ids
        attn_mask1, pos_ids1 = generate_attention_mask_and_position_ids(
            input_ids1, vars1, edges1, self.tokenizer, self.var_len_cache
        )
        attn_mask2, pos_ids2 = generate_attention_mask_and_position_ids(
            input_ids2, vars2, edges2, self.tokenizer, self.var_len_cache
        )

        return {
            "input_ids1": torch.tensor(input_ids1, dtype=torch.long),
            "input_ids2": torch.tensor(input_ids2, dtype=torch.long),
            "attention_mask1": attn_mask1,
            "attention_mask2": attn_mask2,
            "position_ids1": pos_ids1,
            "position_ids2": pos_ids2,
            "label": torch.tensor(label, dtype=torch.long),
        }


# ------------------------------
# CodeBERT is based on RoBERTa, so we use RoBERTa base classes
# ------------------------------
from transformers.models.roberta.modeling_roberta import (
    RobertaEmbeddings, RobertaEncoder, RobertaPooler, RobertaPreTrainedModel
)
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers import RobertaConfig


class CustomRobertaModel(RobertaPreTrainedModel):
    """RoBERTa model supporting full attention mask (3D mask), adapted for CodeBERT"""

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)
        self.pooler = RobertaPooler(config) if add_pooling_layer else None
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            token_type_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length, seq_length), device=device)
        # If attention_mask is 2D (batch, seq_len), expand to 3D
        if attention_mask.dim() == 2:
            extended_mask = attention_mask[:, None, None, :]
            extended_mask = extended_mask.to(dtype=next(self.parameters()).dtype)
            extended_mask = (1.0 - extended_mask) * torch.finfo(self.dtype).min
            attention_mask_3d = extended_mask
        else:
            # Already 3D mask, convert to 4D (batch, 1, seq_len, seq_len)
            attention_mask_3d = attention_mask[:, None, :, :]
            attention_mask_3d = attention_mask_3d.to(dtype=next(self.parameters()).dtype)

        # Get embeddings
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask_3d,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


# Clone detector based on the custom model
class CodeBERTCloneDetectorFixed(nn.Module):
    def __init__(self, model_name="microsoft/codebert-base", num_labels=2):
        super().__init__()
        config = RobertaConfig.from_pretrained(model_name)
        self.encoder = CustomRobertaModel.from_pretrained(model_name, config=config)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, num_labels)
        )

    def forward(self, input_ids1, attention_mask1, position_ids1,
                input_ids2, attention_mask2, position_ids2):
        out1 = self.encoder(
            input_ids=input_ids1,
            attention_mask=attention_mask1,
            position_ids=position_ids1,
        )
        out2 = self.encoder(
            input_ids=input_ids2,
            attention_mask=attention_mask2,
            position_ids=position_ids2,
        )
        cls1 = out1.pooler_output
        cls2 = out2.pooler_output
        concat = torch.cat([cls1, cls2], dim=-1)
        logits = self.classifier(concat)
        return logits


# ------------------------------
# Training and evaluation functions
# ------------------------------
def train_epoch(model, dataloader, optimizer, scheduler, device, scaler):
    model.train()
    total_loss = 0
    total_grad_norm = 0
    valid_batch_count = 0
    with tqdm(
            dataloader,
            desc="Training",
            leave=False,
            dynamic_ncols=True,
            mininterval=0.5,
            ascii=True,
            file=sys.stdout
    ) as progress:
        for batch in progress:
            input_ids1 = batch["input_ids1"].to(device)
            input_ids2 = batch["input_ids2"].to(device)
            attention_mask1 = batch["attention_mask1"].to(device)
            attention_mask2 = batch["attention_mask2"].to(device)
            position_ids1 = batch["position_ids1"].to(device)
            position_ids2 = batch["position_ids2"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(
                    input_ids1, attention_mask1, position_ids1,
                    input_ids2, attention_mask2, position_ids2
                )
                loss = nn.CrossEntropyLoss()(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # ----- Perform gradient clipping first, and then compute the norm -----
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Calculate the gradient norm after clipping.
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5

            # Detect abnormal gradients (inf or nan values)
            if torch.isinf(torch.tensor(grad_norm)) or torch.isnan(torch.tensor(grad_norm)):
                # Skip the current batch, without updating parameters, accumulating loss or gradient norm
                scaler.update()
                progress.set_postfix(loss=loss.item(), grad_norm="SKIP")
                continue

            total_grad_norm += grad_norm
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            valid_batch_count += 1
            progress.set_postfix(loss=loss.item(), grad_norm=f"{grad_norm:.2f}")

    sys.stdout.flush()
    avg_loss = total_loss / valid_batch_count if valid_batch_count > 0 else 0
    avg_grad_norm = total_grad_norm / valid_batch_count if valid_batch_count > 0 else 0
    return avg_loss, avg_grad_norm


def evaluate(model, dataloader, device):
    """
    Evaluate the model and return accuracy, precision, recall, f1.
    All metrics are for the positive class (label=1).
    """
    model.eval()
    all_preds = []
    all_labels = []
    with tqdm(
            dataloader,
            desc="Evaluating",
            leave=False,
            dynamic_ncols=True,
            mininterval=0.5,
            ascii=True,
            file=sys.stdout
    ) as progress:
        with torch.no_grad():
            for batch in progress:
                input_ids1 = batch["input_ids1"].to(device)
                input_ids2 = batch["input_ids2"].to(device)
                attention_mask1 = batch["attention_mask1"].to(device)
                attention_mask2 = batch["attention_mask2"].to(device)
                position_ids1 = batch["position_ids1"].to(device)
                position_ids2 = batch["position_ids2"].to(device)
                labels = batch["label"].to(device)

                logits = model(
                    input_ids1, attention_mask1, position_ids1,
                    input_ids2, attention_mask2, position_ids2
                )
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                # Update progress postfix with current accuracy
                if len(all_labels) > 0:
                    curr_acc = (np.array(all_preds) == np.array(all_labels)).mean()
                    progress.set_postfix(acc=f"{curr_acc:.4f}")

    sys.stdout.flush()
    # Convert to numpy arrays
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    # Calculate metrics
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return accuracy, precision, recall, f1


def collate_fn(batch):
    max_len1 = max(item["input_ids1"].size(0) for item in batch)
    max_len2 = max(item["input_ids2"].size(0) for item in batch)

    padded_batch = {}
    # 1D padding fields
    for key in ["input_ids1", "input_ids2", "position_ids1", "position_ids2"]:
        tensors = [item[key] for item in batch]
        padded_batch[key] = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0)

    # 2D attention mask padding
    for key in ["attention_mask1", "attention_mask2"]:
        max_len = max_len1 if "1" in key else max_len2
        padded_masks = []
        for item in batch:
            mask = item[key]                     # (seq_len, seq_len)
            seq_len = mask.size(0)
            if seq_len < max_len:
                # Pad columns to the right
                pad_right = torch.full((seq_len, max_len - seq_len), -1e9, dtype=mask.dtype)
                mask = torch.cat([mask, pad_right], dim=1)
                # Pad rows downwards
                pad_down = torch.full((max_len - seq_len, max_len), -1e9, dtype=mask.dtype)
                mask = torch.cat([mask, pad_down], dim=0)
            padded_masks.append(mask)
        padded_batch[key] = torch.stack(padded_masks, dim=0)

    # Labels
    padded_batch["label"] = torch.stack([item["label"] for item in batch], dim=0)
    return padded_batch


# ------------------------------
# Main program
# ------------------------------
def main():
    print("Loading tokenizer and datasets...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    train_data = load_from_disk("./processed_train")
    val_data = load_from_disk("./processed_val")
    test_data = load_from_disk("./processed_test")

    print(f"Train set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    print(f"Test set size: {len(test_data)}")

    train_dataset = CloneDataset(train_data, tokenizer)
    val_dataset = CloneDataset(val_data, tokenizer)
    test_dataset = CloneDataset(test_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    model = CodeBERTCloneDetectorFixed(num_labels=2).to(DEVICE)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler('cuda')

    # For recording training logs
    records = {
        "epoch": [],
        "train_loss": [],
        "train_grad_norm": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "learning_rate": []
    }

    best_val_f1 = 0.0  # Use F1 as the criterion for saving best model (can also use accuracy)
    for epoch in range(EPOCHS):
        print("\n" + "=" * 50)
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        train_loss, train_grad_norm = train_epoch(model, train_loader, optimizer, scheduler, DEVICE, scaler)
        val_acc, val_precision, val_recall, val_f1 = evaluate(model, val_loader, DEVICE)
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Train Loss: {train_loss:.4f} | Train Grad Norm: {train_grad_norm:.4f}"
        )
        print(
            f"Val Acc: {val_acc:.4f} | Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f} | Val F1: {val_f1:.4f} | LR: {current_lr:.2e}"
        )

        # Record
        records["epoch"].append(epoch + 1)
        records["train_loss"].append(train_loss)
        records["train_grad_norm"].append(train_grad_norm)
        records["val_acc"].append(val_acc)
        records["val_precision"].append(val_precision)
        records["val_recall"].append(val_recall)
        records["val_f1"].append(val_f1)
        records["learning_rate"].append(current_lr)

        # Save best model based on validation F1 score (or accuracy, change as needed)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
            print("Saved best model (based on validation F1).")

    # Save CSV log
    import pandas as pd
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(SAVE_DIR, "training_log.csv"), index=False)
    print(f"Training log saved to {os.path.join(SAVE_DIR, 'training_log.csv')}")

    # Evaluate on test set using best model
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model.pth")))
    test_acc, test_precision, test_recall, test_f1 = evaluate(model, test_loader, DEVICE)
    print("\n" + "=" * 50)
    print("Test Set Results:")
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"F1 Score:  {test_f1:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()