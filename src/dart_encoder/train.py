from __future__ import annotations
import argparse, json, logging, random
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from .data_types import TrainSample
from .dataset import CTADataset, CTACollator
from .model import BiEncoder, evaluate_recall


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("train")



## train DART Encoder
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model     = BiEncoder(
        args.model_name,
        temperature=args.temperature,
        gradient_checkpointing=args.gradient_checkpointing,
    ).to(device)

    logger.info("grad_ckpt=%s", args.gradient_checkpointing)

    # Datasets
    train_dataset = CTADataset(
        args.train_path,
        args.ontology_path,
        max_cells=args.max_cells,
        max_parents=args.max_parents,
        seed=args.seed,
        mask_header_prob=args.mask_header_prob,
        random_cell_sample=args.random_cell_sample,
        min_majority_ratio=args.min_majority_ratio,
        min_type_count=args.min_type_count,
        hard_neg_path=args.hard_neg_path,
    )
    collator = CTACollator(tokenizer, max_length=args.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Validation samples (load separately, no collation needed)
    val_samples: List[TrainSample] = []
    if args.val_path:
        with open(args.val_path) as f:
            for line in f:
                d = json.loads(line)
                val_samples.append(TrainSample(
                    anchor_header=d["anchor_header"],
                    anchor_cells=d["anchor_cells"],
                    positive_qid=d["positive_type_qid"],
                    hard_negative_qids=d.get("hard_negative_type_qids", []),
                ))
        logger.info("Loaded %d validation samples", len(val_samples))

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps   = len(train_loader) * args.epochs
    warmup_steps  = int(total_steps * args.warmup_ratio)
    scheduler     = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    # Output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_recall20 = 0.0
    global_step   = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            query_enc = {k: v.to(device) for k, v in batch["query_enc"].items()}
            pos_enc   = {k: v.to(device) for k, v in batch["pos_enc"].items()}
            neg_enc   = (
                {k: v.to(device) for k, v in batch["neg_enc"].items()}
                if batch["neg_enc"] is not None
                else None
            )

            loss = model(query_enc, pos_enc, neg_enc, batch["neg_counts"])
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss  += loss.item()
            global_step += 1

            if step % args.log_every == 0:
                avg = epoch_loss / step
                lr  = scheduler.get_last_lr()[0]
                logger.info(
                    "Epoch %d | Step %d/%d | Loss %.4f | LR %.2e",
                    epoch, step, len(train_loader), avg, lr,
                )

        # End-of-epoch evaluation
        if val_samples:
            train_dataset.training = False   # disable augmentation during eval
            metrics = evaluate_recall(
                model,
                val_samples,
                train_dataset.ontology,
                tokenizer,
                device,
                batch_size=args.batch_size,
                max_length=args.max_length,
                max_cells=args.max_cells,
                max_parents=args.max_parents,
            )
            train_dataset.training = True    # re-enable for next epoch
            logger.info("Epoch %d | %s", epoch, metrics)

            recall20 = metrics.get("Recall@20", 0.0)
            if recall20 > best_recall20:
                best_recall20 = recall20
                save_path = out_dir / "best_model"
                model.encoder.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                logger.info(
                    "New best Recall@20=%.4f  → saved to %s", recall20, save_path
                )

        # Save checkpoint every epoch
        ckpt_path = out_dir / f"epoch_{epoch}"
        model.encoder.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)

    logger.info("Training complete. Best Recall@20: %.4f", best_recall20)



def parse_args():
    p = argparse.ArgumentParser(description="Train DART bi-encoder retrieval model")
    p.add_argument("--train_path",    required=True,  help="Path to train.jsonl")
    p.add_argument("--val_path",      default=None,   help="Path to val.jsonl")
    p.add_argument("--ontology_path", required=True,  help="Path to ontology_types.jsonl")
    p.add_argument("--output_dir",    required=True,  help="Output directory")
    p.add_argument("--model_name",    default="intfloat/multilingual-e5-base")
    p.add_argument("--epochs",        type=int,   default=6)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=2e-5)
    p.add_argument("--weight_decay",  type=float, default=0.01)
    p.add_argument("--warmup_ratio",  type=float, default=0.1)
    p.add_argument("--temperature",   type=float, default=0.05)
    p.add_argument("--max_length",    type=int,   default=192)
    p.add_argument("--max_cells",     type=int,   default=10)
    p.add_argument("--max_parents",   type=int,   default=3)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--log_every",     type=int,   default=50)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--mask_header_prob",  type=float, default=0.3,
                   help="Probability of masking header during training (0=disabled)")
    p.add_argument("--random_cell_sample", action="store_true",
                   help="Randomly sample cells instead of taking first N")
    p.add_argument("--min_majority_ratio", type=float, default=0.7,
                   help="Filter samples with majority_ratio below this threshold")
    p.add_argument("--min_type_count", type=int, default=3,
                   help="Filter types with fewer than this many training samples")
    # Hard negative mining
    p.add_argument("--hard_neg_path", default=None,
                   help="Path to mined hard negatives JSON (from mine_hard_negatives.py)")
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Enable gradient checkpointing to save GPU memory")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train(args)