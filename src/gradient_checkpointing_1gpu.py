
import torch
import time
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import bitsandbytes as bnb

MODEL_NAME  = "gpt2-xl"
SEQ_LEN     = 256
BATCH_SIZE  = 3
GRAD_ACCUM  = 2
MAX_STEPS   = 100
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE       = torch.bfloat16
LOG_FILE    = "step2_metrics.json"

def get_memory_stats():
    return {
        "allocated_gb": round(torch.cuda.memory_allocated()    / 1024**3, 3),
        "reserved_gb" : round(torch.cuda.memory_reserved()     / 1024**3, 3),
        "peak_gb"     : round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }

def log_memory(tag=""):
    s = get_memory_stats()
    print(f"[MEM]{' '+tag+' ' if tag else ' '}"
          f"allocated={s['allocated_gb']:.2f}GB  "
          f"reserved={s['reserved_gb']:.2f}GB  "
          f"peak={s['peak_gb']:.2f}GB")

print("=" * 60)
print("Bước 2 – Baseline: GPT-2 XL | 1 GPU | bf16 | Gradient Checkpointing")
print("=" * 60)

print("\n[1/5] Loading tokenizer & dataset ...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=SEQ_LEN,
        padding="max_length",
    )

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
tokenized = tokenized.filter(lambda x: x["input_ids"].sum() > 0)
dataloader = DataLoader(tokenized, batch_size=BATCH_SIZE, shuffle=True)

print(f"  Dataset size    : {len(tokenized)} samples")
print(f"  Batch size      : {BATCH_SIZE}  (grad_accum={GRAD_ACCUM} → effective bs={BATCH_SIZE*GRAD_ACCUM})")
print(f"  Seq length      : {SEQ_LEN}")

print(f"\n[2/5] Loading {MODEL_NAME} in bf16 ...")
torch.cuda.reset_peak_memory_stats()
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, torch_dtype=DTYPE)
model.to(DEVICE)
log_memory("after model load")

print("\n[3/5] Enabling Gradient Checkpointing ...")
model.gradient_checkpointing_enable()
print("  ✅ gradient_checkpointing = True")
print("  ℹ️  Recompute activations on backward → giảm VRAM, tăng compute ~20-30%")
log_memory("after GC enable")

print("\n[4/5] Setting up AdamW 8-bit optimizer ...")
print("  ℹ️  bf16 không cần GradScaler — dynamic range đủ rộng")
print("  ℹ️  AdamW8bit: optimizer states ở INT8 → giảm ~4x VRAM so FP32 AdamW")
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=5e-5)
log_memory("after optimizer init")

print(f"\n[5/5] Training for {MAX_STEPS} optimizer steps ...")
print("-" * 60)

model.train()
optimizer_step = 0
micro_step     = 0
all_metrics    = []
start_time     = time.time()
accum_loss     = 0.0
step_start     = time.time()

try:
    for batch in dataloader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = input_ids.clone()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss / GRAD_ACCUM
        loss.backward()

        accum_loss += loss.item()
        micro_step += 1

        if micro_step % GRAD_ACCUM == 0:
            optimizer.step()
            optimizer.zero_grad()

            step_time  = time.time() - step_start
            tokens_sec = (BATCH_SIZE * GRAD_ACCUM * SEQ_LEN) / step_time
            mem        = get_memory_stats()

            optimizer_step += 1
            step_metrics = {
                "step"          : optimizer_step,
                "loss"          : round(accum_loss * GRAD_ACCUM, 4),
                "sec_per_step"  : round(step_time, 3),
                "tokens_per_sec": round(tokens_sec, 1),
                "peak_vram_gb"  : mem["peak_gb"],
            }
            all_metrics.append(step_metrics)

            if optimizer_step % 10 == 0:
                print(f"  step={optimizer_step:4d}  loss={step_metrics['loss']:.4f}  "
                      f"tokens/sec={tokens_sec:7.1f}  "
                      f"sec/step={step_time:.3f}s  "
                      f"peak_vram={mem['peak_gb']:.2f}GB")

            accum_loss = 0.0
            step_start = time.time()

            if optimizer_step >= MAX_STEPS:
                break

except torch.cuda.OutOfMemoryError as e:
    print(f"\n💥 OOM at optimizer_step={optimizer_step}, micro_step={micro_step}")
    print(f"   Error: {e}")
    log_memory("OOM point")

total_time = time.time() - start_time

if all_metrics:
    avg_tokens_sec = sum(m["tokens_per_sec"] for m in all_metrics) / len(all_metrics)
    avg_sec_step   = sum(m["sec_per_step"]   for m in all_metrics) / len(all_metrics)
    final_loss     = all_metrics[-1]["loss"]
    peak_vram      = max(m["peak_vram_gb"]   for m in all_metrics)

    print("\n" + "=" * 60)
    print("📊 SUMMARY — Bước 2 (1 GPU + GC + bf16)")
    print("=" * 60)
    print(f"  Steps completed  : {optimizer_step}")
    print(f"  Avg tokens/sec   : {avg_tokens_sec:.1f}")
    print(f"  Avg sec/step     : {avg_sec_step:.3f}s")
    print(f"  Peak VRAM        : {peak_vram:.2f} GB")
    print(f"  Final loss       : {final_loss:.4f}")
    print(f"  Total time       : {total_time:.1f}s")
    print("=" * 60)

    summary = {
        "method"          : "1GPU_GC_bf16_adamw8bit",
        "config"          : {
            "dtype"        : "bfloat16",
            "seq_len"      : SEQ_LEN,
            "batch_size"   : BATCH_SIZE,
            "grad_accum"   : GRAD_ACCUM,
            "effective_bs" : BATCH_SIZE * GRAD_ACCUM,
        },
        "steps"           : optimizer_step,
        "avg_tokens_sec"  : round(avg_tokens_sec, 1),
        "avg_sec_per_step": round(avg_sec_step, 3),
        "peak_vram_gb"    : peak_vram,
        "final_loss"      : final_loss,
        "total_time_sec"  : round(total_time, 1),
        "per_step_metrics": all_metrics,
    }
    with open(LOG_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Metrics saved to {LOG_FILE}")
