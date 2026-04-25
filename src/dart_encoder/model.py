from __future__ import annotations
from typing import List, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .data_types import OntologyType, TrainSample
from .input_format import format_query, format_type


class BiEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.05,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def mean_pool(self, model_output, attention_mask) -> torch.Tensor:
        token_embeddings = model_output.last_hidden_state          # (B, T, H)
        mask = attention_mask.unsqueeze(-1).float()                 # (B, T, 1)
        summed = (token_embeddings * mask).sum(dim=1)               # (B, H)
        counts = mask.sum(dim=1).clamp(min=1e-9)                    # (B, 1)
        return summed / counts                                       # (B, H)

    def encode(self, encoding: dict) -> torch.Tensor:
        out = self.encoder(**encoding)
        emb = self.mean_pool(out, encoding["attention_mask"])
        return F.normalize(emb, dim=-1)

    def forward(
        self,
        query_enc: dict,
        pos_enc: dict,
        neg_enc: Optional[dict],
        neg_counts: List[int],
    ) -> torch.Tensor:
        q_emb   = self.encode(query_enc)                            # (B, H)
        pos_emb = self.encode(pos_enc)                              # (B, H)

        # In-batch negatives: all positive embeddings serve as negatives
        # for every other query in the batch.
        # Shape: (B, B) — diagonal is positive
        sim_matrix = torch.matmul(q_emb, pos_emb.T) / self.temperature.clamp(min=1e-4)
        labels = torch.arange(len(q_emb), device=q_emb.device)
        loss_inbatch = F.cross_entropy(sim_matrix, labels)

        # Static hard negatives loss
        loss_hard = torch.tensor(0.0, device=q_emb.device)
        if neg_enc is not None and sum(neg_counts) > 0:
            neg_emb = self.encode(neg_enc)                          # (N_total, H)

            # Split per sample
            neg_emb_list = torch.split(neg_emb, neg_counts, dim=0)

            hard_losses = []
            for i, (q, p, negs) in enumerate(
                zip(q_emb, pos_emb, neg_emb_list)
            ):
                if negs.shape[0] == 0:
                    continue
                # Concatenate positive and negatives: [pos, neg1, neg2, ...]
                candidates = torch.cat([p.unsqueeze(0), negs], dim=0)  # (1+K, H)
                scores = (q.unsqueeze(0) @ candidates.T) / self.temperature.clamp(min=1e-4)
                # Label 0 = positive
                target = torch.zeros(1, dtype=torch.long, device=q_emb.device)
                hard_losses.append(F.cross_entropy(scores, target))

            if hard_losses:
                loss_hard = torch.stack(hard_losses).mean()

        # Combine losses with equal weight
        loss = loss_inbatch + loss_hard
        return loss



## evaluation helpers

@torch.no_grad()
def evaluate_recall(
    model: BiEncoder,
    val_samples: List[TrainSample],
    ontology: Dict[str, OntologyType],
    tokenizer,
    device: torch.device,
    batch_size: int = 64,
    max_length: int = 256,
    max_cells: int = 10,
    max_parents: int = 3,
    k_values: List[int] = [1, 5, 20],
) -> Dict[str, float]:
    model.eval()

    def tokenize(texts):
        return {
            k: v.to(device)
            for k, v in tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).items()
        }

    # Build type pool embeddings
    all_qids   = list(ontology.keys())
    all_texts  = [format_type(ontology[q], max_parents) for q in all_qids]
    type_embs  = []

    for i in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[i : i + batch_size]
        enc = tokenize(batch_texts)
        emb = model.encode(enc)
        type_embs.append(emb.cpu())

    type_embs  = torch.cat(type_embs, dim=0)                       # (N_types, H)
    qid_to_idx = {q: i for i, q in enumerate(all_qids)}

    # Encode queries and compute recall
    hits = {k: 0 for k in k_values}
    total = 0

    for i in range(0, len(val_samples), batch_size):
        batch = val_samples[i : i + batch_size]
        query_texts = [
            format_query(s.anchor_header, s.anchor_cells, max_cells)
            for s in batch
        ]
        enc    = tokenize(query_texts)
        q_embs = model.encode(enc).cpu()                            # (B, H)

        scores = q_embs @ type_embs.T                               # (B, N_types)

        for j, s in enumerate(batch):
            pos_idx = qid_to_idx.get(s.positive_qid)
            if pos_idx is None:
                continue
            ranked = scores[j].argsort(descending=True).tolist()
            for k in k_values:
                if pos_idx in ranked[:k]:
                    hits[k] += 1
            total += 1

    model.train()
    return {f"Recall@{k}": hits[k] / max(total, 1) for k in k_values}
