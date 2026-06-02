
import os, time, json
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
import bitsandbytes as bnb
from torch.utils.checkpoint import checkpoint
import deepspeed

deepspeed.utils.nvtx._range_push = lambda *args, **kwargs: None
deepspeed.utils.nvtx._range_pop  = lambda *args, **kwargs: None
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_NAME  = "gpt2-xl"
SEQ_LEN     = 256
EFFECTIVE_BS = 16
MAX_STEPS   = 100
DTYPE       = torch.bfloat16
LOG_FILE    = "step3_deepspeed_metrics.json"
CHUNKS_LIST = [4, 8, 16]   # DeepSpeed gọi đây là gradient_accumulation_steps
SPLIT_LAYER = 24              

class Stage0(nn.Module):
    """Embedding + transformer blocks 0..SPLIT_LAYER-1"""
    def __init__(self, model):
        super().__init__()
        t = model.transformer
        self.wte    = t.wte
        self.wpe    = t.wpe
        self.drop   = t.drop
        self.blocks = nn.ModuleList(t.h[:SPLIT_LAYER])

    def forward(self, inputs):
        input_ids, attention_mask = inputs
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)
        hidden = self.drop(self.wte(input_ids) + self.wpe(pos))
        
        causal_mask = torch.tril(torch.ones((T, T), device=input_ids.device, dtype=torch.bool))
        causal_mask = causal_mask[None, None, :, :] 
        pad_mask = attention_mask[:, None, None, :].bool() 
        extended_attn_mask = causal_mask & pad_mask  
        
        for block in self.blocks:
            hidden = checkpoint(block, hidden, use_reentrant=False, attention_mask=extended_attn_mask)[0]
            
        return (hidden, extended_attn_mask)

class Stage1(nn.Module):
    """Transformer blocks SPLIT_LAYER..47 + ln_f + lm_head"""
    def __init__(self, model):
        super().__init__()
        t = model.transformer
        self.blocks  = nn.ModuleList(t.h[SPLIT_LAYER:])
        self.ln_f    = t.ln_f
        self.lm_head = model.lm_head

    def forward(self, inputs):
        hidden, extended_attn_mask = inputs
        
        for block in self.blocks:
            hidden = checkpoint(block, hidden, use_reentrant=False, attention_mask=extended_attn_mask)[0]
            
        hidden = self.ln_f(hidden)
        logits = self.lm_head(hidden) 
        return logits

def compute_loss(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return nn.CrossEntropyLoss()(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )

class DeepSpeedPipelineIterator:
    """DeepSpeed Pipeline yêu cầu Iterator trả về đúng chuẩn ((inputs), labels)"""
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iter = iter(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.iter)
        except StopIteration:
            self.iter = iter(self.dataloader)
            batch = next(self.iter)
            
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = input_ids.clone()
        
        return ((input_ids, attention_mask), labels)

def get_memory_stats(device):
    return {
        "allocated_gb": round(torch.cuda.memory_allocated(device)    / 1024**3, 3),
        "peak_gb"     : round(torch.cuda.max_memory_allocated(device) / 1024**3, 3),
    }

def get_dataloader(tokenizer, rank, world_size, micro_batch_size):
    if rank == 0:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        def tokenize(examples):
            return tokenizer(examples["text"], truncation=True, max_length=SEQ_LEN, padding="max_length")
        tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
        tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
        tokenized = tokenized.filter(lambda x: x["input_ids"].sum() > 0)
        
    dist.barrier()

    if rank != 0:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        def tokenize(examples):
            return tokenizer(examples["text"], truncation=True, max_length=SEQ_LEN, padding="max_length")
        tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
        tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
        tokenized = tokenized.filter(lambda x: x["input_ids"].sum() > 0)

    dist.barrier()
    sampler = DistributedSampler(tokenized, num_replicas=world_size, rank=rank, shuffle=True)
    return DataLoader(tokenized, batch_size=micro_batch_size, sampler=sampler, drop_last=True)

def train_worker(rank, world_size, chunks, result_queue):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    
    deepspeed.init_distributed(dist_backend="nccl")
    torch.cuda.set_device(rank)
    torch.cuda.reset_peak_memory_stats(rank)

    if rank == 0:
        print(f"\n{'='*60}\nDeepSpeed Pipeline | chunks={chunks} | 2 GPU\n{'='*60}")

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    full_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, torch_dtype=DTYPE)
    full_model.eval()

    layers = [
        Stage0(full_model),
        Stage1(full_model)
    ]

    model = deepspeed.PipelineModule(
        layers=layers,
        num_stages=world_size,
        loss_fn=compute_loss,
        partition_method="parameters",
        activation_checkpoint_interval=0 
    )

    del full_model
    torch.cuda.empty_cache()

    mem = get_memory_stats(rank)
    print(f"[GPU{rank}] loaded — {mem['allocated_gb']:.2f}GB")

    micro_batch_size = EFFECTIVE_BS // chunks

    ds_config = {
        "train_batch_size": EFFECTIVE_BS,
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": chunks,
        "bf16": {"enabled": True},
        "zero_allow_untested_optimizer": True,
        "zero_optimization": {
            "stage": 1,                   
            "reduce_bucket_size": 5e8,    
            "allgather_bucket_size": 5e8
        }
    }

    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=5e-5)

    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config
    )

    dataloader = get_dataloader(tokenizer, rank, world_size, micro_batch_size)
    data_iter = iter(DeepSpeedPipelineIterator(dataloader))

    all_metrics = []
    start_time = time.time()

    try:
        for step in range(1, MAX_STEPS + 1):
            step_start = time.time()
            
            loss = engine.train_batch(data_iter=data_iter)

            if rank == world_size - 1:
                step_time = time.time() - step_start
                tokens_sec = (EFFECTIVE_BS * SEQ_LEN) / step_time
                mem = get_memory_stats(rank)
                bubble = (world_size - 1) / (chunks + world_size - 1)

                step_metrics = {
                    "step"          : step,
                    "loss"          : round(loss.item(), 4),
                    "sec_per_step"  : round(step_time, 3),
                    "tokens_per_sec": round(tokens_sec, 1),
                    "peak_vram_gb"  : mem["peak_gb"],
                    "chunks"        : chunks,
                    "bubble_ratio"  : round(bubble, 4),
                }
                all_metrics.append(step_metrics)

                if step % 10 == 0:
                    print(f"  step={step:4d}  loss={step_metrics['loss']:.4f}  "
                          f"tok/s={tokens_sec:8.1f}  "
                          f"sec/step={step_time:.3f}s  "
                          f"bubble={bubble:.2%}")

    except torch.cuda.OutOfMemoryError as e:
        print(f"\n[GPU{rank}] 💥 OOM: {e}")

    if rank == world_size - 1 and all_metrics:
        total_time = time.time() - start_time
        avg_tokens_sec = sum(m["tokens_per_sec"] for m in all_metrics) / len(all_metrics)
        avg_sec_step = sum(m["sec_per_step"] for m in all_metrics) / len(all_metrics)
        bubble_ratio = round((world_size - 1) / (chunks + world_size - 1), 4)

        summary = {
            "method"          : "2GPU_deepspeed_pipeline",
            "chunks"          : chunks,
            "bubble_ratio"    : bubble_ratio,
            "config"          : {"dtype": "bfloat16", "seq_len": SEQ_LEN,
                                 "batch_size": EFFECTIVE_BS, 
                                 "effective_bs": EFFECTIVE_BS,
                                 "split_layer": SPLIT_LAYER},
            "steps"           : step,
            "avg_tokens_sec"  : round(avg_tokens_sec, 1),
            "avg_sec_per_step": round(avg_sec_step, 3),
            "peak_vram_gb"    : max(m["peak_vram_gb"] for m in all_metrics),
            "final_loss"      : all_metrics[-1]["loss"],
            "total_time_sec"  : round(total_time, 1),
        }
        result_queue.put(summary)
        print(f"\n{'='*60}")
        print(f"📊 chunks={chunks} | tok/s={avg_tokens_sec:.1f} | "
              f"bubble={bubble_ratio:.2%} | peak_vram={summary['peak_vram_gb']:.2f}GB")
        print(f"{'='*60}")

    dist.destroy_process_group()

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    world_size  = 2
    all_results = []
    ctx         = mp.get_context("spawn")

    for chunks in CHUNKS_LIST:
        print(f"\n{'#'*60}\n# Thử nghiệm chunks = {chunks}\n{'#'*60}")
        result_queue = ctx.Queue()

        mp.spawn(
            train_worker,
            args=(world_size, chunks, result_queue),
            nprocs=world_size,
            join=True,
        )

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
