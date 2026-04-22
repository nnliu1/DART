from __future__ import annotations
import json, logging
from typing import List, Tuple
from pathlib import Path
import logging

import torch
from transformers import AutoModel, AutoTokenizer

from input_format import format_query, format_type


logger = logging.getLogger(__name__)


## ontology loader
def load_ontology(
    ontology_path: str,
    max_parents: int = 3,
) -> Tuple[List[str], List[str], List[str]]:
    with open(ontology_path) as f:
        data = json.load(f)
 
    concept_ids, labels, type_texts = [], [], []
    for uri, info in data["concepts"].items():
        label       = info.get("label", "")
        description = info.get("description", "")
        parents     = info.get("parents") or info.get("ancestor_chain", [])
 
        concept_ids.append(uri)
        labels.append(label)
        type_texts.append(format_type(label, description, parents, max_parents))
 
    logger.info("Loaded %d ontology types", len(concept_ids))
    return concept_ids, labels, type_texts
 
 
def load_dataset(data_dir: str) -> List[dict]:
    records = []
    for f in sorted(Path(data_dir).glob("*.json")):
        with open(f) as fp:
            records.append(json.load(fp))
    logger.info("Loaded %d columns", len(records))
    return records

## DART encoder
class BiEncoderInference:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device    = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model     = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        logger.info("Model loaded from %s on %s", model_path, self.device)
 
    def mean_pool(self, model_output, attention_mask) -> torch.Tensor:
        token_emb = model_output.last_hidden_state
        mask      = attention_mask.unsqueeze(-1).float()
        return (token_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
 
    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 128,
               max_length: int = 256) -> torch.Tensor:
        all_embs = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(
                texts[i : i + batch_size], padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc)
            emb = self.mean_pool(out, enc["attention_mask"])
            emb = F.normalize(emb, dim=-1)
            all_embs.append(emb.cpu())
        return torch.cat(all_embs, dim=0)


## retriever
    def __init__(
        self,
        model_path: str,
        concept_ids: List[str],
        labels: List[str],
        type_texts: List[str],
        batch_size: int = 128,
        max_length: int = 256,
        device: str = "cuda",
    ):
        self.concept_ids = concept_ids
        self.labels      = labels
        self.batch_size  = batch_size
        self.max_length  = max_length
        self.model       = BiEncoderInference(model_path, device=device)
 
        logger.info("Building type index (%d types)...", len(type_texts))
        self.type_embs = self.model.encode(
            type_texts, batch_size=batch_size, max_length=max_length
        )                                                        # (N, H)
        logger.info("Type index built.")
 
    def retrieve(
        self,
        header: str,
        cells: List[str],
        top_k: int = 20,
        max_cells: int = 10,
    ) -> List[Tuple[str, str, float]]:
        """
        Retrieve top-K types for a single column.
 
        Returns:
            List of (concept_id, label, score) sorted by score descending.
        """
        q_emb  = self.model.encode(
            [format_query(header, cells, max_cells)],
            max_length=self.max_length,
        )                                                        # (1, H)
        scores = (q_emb @ self.type_embs.T).squeeze(0)          # (N,)
        top_idx = scores.argsort(descending=True)[:top_k].tolist()
 
        return [
            (self.concept_ids[i], self.labels[i], scores[i].item())
            for i in top_idx
        ]
