
"""
Bước 3: GPT-2 XL Pipeline Parallelism trên 2 GPU
- torch.distributed.pipelining, manual stage split
- bf16, AdamW8bit, BATCH_SIZE=16, SEQ_LEN=256
- Thử chunks = 2, 4, 8, 16
"""

import os, time, json
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
from torch.distributed.pipelining.microbatch import TensorChunkSpec
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
import bitsandbytes as bnb
from torch.utils.checkpoint import checkpoint

# ── Config ─────────────────────────────────────────────────────
MODEL_NAME  = "gpt2-xl"
SEQ_LEN     = 256
BATCH_SIZE  = 16
GRAD_ACCUM  = 2
MAX_STEPS   = 100
DTYPE       = torch.bfloat16
LOG_FILE    = "step3_metrics.json"
CHUNKS_LIST = [2, 4, 8, 16]
SPLIT_LAYER = 24

# ── Stage 0 ────────────────────────────────────────────────────
class Stage0(nn.Module):
    def __init__(self, model):
        super().__init__()
        t = model.transformer
        self.wte    = t.wte
        self.wpe    = t.wpe
        self.drop   = t.drop
        self.blocks = nn.ModuleList(t.h[:SPLIT_LAYER])

    def forward(self, input_ids, attention_mask):
        B, T  = input_ids.shape
        pos   = torch.arange(T, device=input_ids.device)
        hidden = self.drop(self.wte(input_ids) + self.wpe(pos))

        causal = torch.tril(torch.ones(T, T, device=hidden.device, dtype=torch.bool))[None, None]
        pad    = attention_mask[:, None, None, :].bool()
        mask   = causal & pad

        for block in self.blocks:
            hidden = checkpoint(block, hidden, use_reentrant=False,
                                attention_mask=mask)[0]
        # Trả về hidden + mask để Stage1 dùng (mask cần để backprop qua cả 2 stage)
        return hidden, mask.to(dtype=hidden.dtype)

# ── Stage 1 ────────────────────────────────────────────────────
class Stage1(nn.Module):
    def __init__(self, model):
        super().__init__()
        t = model.transformer
        self.blocks  = nn.ModuleList(t.h[SPLIT_LAYER:])
        self.ln_f    = t.ln_f
        self.lm_head = model.lm_head

    def forward(self, hidden, mask):
        # Dummy op để giữ gradient flow qua mask
        hidden = hidden + mask.sum() * 0.0
        bool_mask = mask.bool()
        for block in self.blocks:
            hidden = checkpoint(block, hidden, use_reentrant=False,
                                attention_mask=bool_mask)[0]
        return self.lm_head(self.ln_f(hidden))

# ── Loss ───────────────────────────────────────────────────────
def compute_loss(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return nn.CrossEntropyLoss()(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )

# ── Helpers ────────────────────────────────────────────────────
def get_memory_stats(device):
    return {
        "allocated_gb": round(torch.cuda.memory_allocated(device)    / 1024**3, 3),
        "peak_gb"     : round(torch.cuda.max_memory_allocated(device) / 1024**3, 3),
    }

def get_dataloader(tokenizer, rank, world_size):
    # 1. Cho GPU 0 tải và xử lý data trước
    if rank == 0:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        def tokenize(examples):
            return tokenizer(examples["text"], truncation=True, max_length=SEQ_LEN, padding="max_length")
        tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
        tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
        tokenized = tokenized.filter(lambda x: x["input_ids"].sum() > 0)
        
    # 2. Ép TẤT CẢ GPU gặp nhau ở đây (GPU 1 sẽ đợi GPU 0 làm xong việc trên)
    dist.barrier()

    # 3. Bây giờ GPU 1 mới vào việc (tốc độ sẽ tính bằng mili-giây vì đọc thẳng từ cache)
    if rank != 0:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        def tokenize(examples):
            return tokenizer(examples["text"], truncation=True, max_length=SEQ_LEN, padding="max_length")
        tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
        tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
        tokenized = tokenized.filter(lambda x: x["input_ids"].sum() > 0)

    # 4. Gặp nhau lần nữa cho chắc cú trước khi chia data
    dist.barrier()

    sampler = DistributedSampler(tokenized, num_replicas=world_size, rank=rank, shuffle=True)
    return DataLoader(tokenized, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)

# ── Worker ─────────────────────────────────────────────────────
def train_worker(rank, world_size, chunks, result_queue):
    dist.init_process_group(backend="nccl", init_method="env://",
                            world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    torch.cuda.reset_peak_memory_stats(rank)
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"\n{'='*60}\nPipeline | chunks={chunks} | 2 GPU\n{'='*60}")

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    full_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, torch_dtype=DTYPE)
    full_model.eval()

    stage_mod = Stage0(full_model).to(dtype=DTYPE, device=device) if rank == 0 \
                else Stage1(full_model).to(dtype=DTYPE, device=device)
    del full_model
    torch.cuda.empty_cache()

    print(f"[GPU{rank}] loaded — {get_memory_stats(rank)['allocated_gb']:.2f}GB")

    optimizer = bnb.optim.AdamW8bit(stage_mod.parameters(), lr=5e-5)

    # Build stage & schedule (1 lần duy nhất, không tạo lại trong loop)
    stage = PipelineStage(stage_mod, stage_index=rank,
                          num_stages=world_size, device=device)

    # Stage0 output: (hidden, mask) → 2 tensors → TensorChunkSpec cho cả 2
    # Stage1 nhận (hidden, mask) → labels truyền qua target
    schedule = ScheduleGPipe(
        stage,
        n_microbatches=chunks,
        loss_fn=compute_loss,
        args_chunk_spec=(TensorChunkSpec(0), TensorChunkSpec(0)),
    )

    dataloader = get_dataloader(tokenizer, rank, world_size)
    stage_mod.train()

    optimizer_step = 0
    accum_step     = 0        # đếm số lần schedule.step() để gộp GRAD_ACCUM
    all_metrics    = []
    total_start    = time.time()
    accum_loss     = 0.0

    try:
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = input_ids.clone()

            # ── Đo thời gian bắt đầu từ ĐÂY ──────────────────
            step_start = time.time()

            if rank == 0:
                # Stage0: feed input, nhận gradient từ Stage1
                schedule.step(input_ids, attention_mask)
            else:
                # Stage1: tính loss, backward tự động bởi schedule
                losses = []
                schedule.step(target=labels, losses=losses)
                if losses:
                    accum_loss += sum(l.item() for l in losses) / len(losses)

            accum_step += 1

            # Optimizer step sau GRAD_ACCUM lần schedule.step()
            if accum_step % GRAD_ACCUM == 0:
                optimizer.step()
                optimizer.zero_grad()
                dist.barrier()   # sync 2 GPU trước khi đo thời gian

                step_time = time.time() - step_start
                optimizer_step += 1

                if rank == world_size - 1:
                    # tokens = BATCH_SIZE * GRAD_ACCUM * SEQ_LEN (pipeline ≠ data parallel)
                    tokens_sec = (BATCH_SIZE * GRAD_ACCUM * SEQ_LEN) / step_time
                    mem        = get_memory_stats(rank)
                    p          = world_size
                    bubble     = (p - 1) / (chunks + p - 1)

                    step_metrics = {
                        "step"          : optimizer_step,
                        "loss"          : round(accum_loss, 4),
                        "sec_per_step"  : round(step_time, 3),
                        "tokens_per_sec": round(tokens_sec, 1),
                        "peak_vram_gb"  : mem["peak_gb"],
                        "chunks"        : chunks,
                        "bubble_ratio"  : round(bubble, 4),
                    }
                    all_metrics.append(step_metrics)

                    if optimizer_step % 10 == 0:
                        print(f"  step={optimizer_step:4d}  loss={accum_loss:.4f}  "
                              f"tok/s={tokens_sec:8.1f}  "
                              f"sec/step={step_time:.3f}s  "
                              f"bubble={bubble:.2%}")

                accum_loss = 0.0

                if optimizer_step >= MAX_STEPS:
                    break

    except torch.cuda.OutOfMemoryError as e:
        print(f"[GPU{rank}] 💥 OOM: {e}")

    if rank == world_size - 1 and all_metrics:
        total_time     = time.time() - total_start
        avg_tokens_sec = sum(m["tokens_per_sec"] for m in all_metrics) / len(all_metrics)
        avg_sec_step   = sum(m["sec_per_step"]   for m in all_metrics) / len(all_metrics)
        bubble_ratio   = round((world_size - 1) / (chunks + world_size - 1), 4)

        summary = {
            "method"          : "2GPU_pytorch_pipeline",
            "chunks"          : chunks,
            "bubble_ratio"    : bubble_ratio,
            "config"          : {"dtype": "bfloat16", "seq_len": SEQ_LEN,
                                 "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
                                 "effective_bs": BATCH_SIZE * GRAD_ACCUM,
                                 "split_layer": SPLIT_LAYER},
            "steps"           : optimizer_step,
            "avg_tokens_sec"  : round(avg_tokens_sec, 1),
            "avg_sec_per_step": round(avg_sec_step, 3),
            "peak_vram_gb"    : max(m["peak_vram_gb"] for m in all_metrics),
            "final_loss"      : all_metrics[-1]["loss"],
            "total_time_sec"  : round(total_time, 1),
            "per_step_metrics": all_metrics,
        }
        result_queue.put(summary)
        print(f"\n{'='*60}")
        print(f"📊 chunks={chunks} | tok/s={avg_tokens_sec:.1f} | "
              f"bubble={bubble_ratio:.2%} | peak_vram={summary['peak_vram_gb']:.2f}GB")
        print(f"{'='*60}")

    dist.destroy_process_group()

# ── Main ───────────────────────────────────────────────────────
if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    world_size  = 2
    all_results = []
    ctx         = mp.get_context("spawn")

    for chunks in CHUNKS_LIST:
        print(f"\n{'#'*60}\n# Thử nghiệm chunks = {chunks}\n{'#'*60}")
        result_queue = ctx.Queue()
        mp.spawn(train_worker, args=(world_size, chunks, result_queue),
                 nprocs=world_size, join=True)
        if not result_queue.empty():
            all_results.append(result_queue.get())
        torch.cuda.empty_cache()
        time.sleep(2)

    with open(LOG_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}\n✅ Saved to {LOG_FILE}")
    print(f"\n{'chunks':>8} | {'tok/s':>12} | {'bubble':>8} | {'sec/step':>10} | {'loss':>8}")
    print("-" * 58)
    for r in all_results:
        print(f"{r['chunks']:>8} | {r['avg_tokens_sec']:>12.1f} | "
              f"{r['bubble_ratio']:>8.2%} | {r['avg_sec_per_step']:>10.3f}s | "
              f"{r['final_loss']:>8.4f}")
