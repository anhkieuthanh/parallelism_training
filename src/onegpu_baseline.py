
import torch
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import bitsandbytes as bnb

# ── Config (đồng bộ với Bước 2) ───────────────────────────────
MODEL_NAME  = "gpt2-xl"
SEQ_LEN     = 256
BATCH_SIZE  = 3
GRAD_ACCUM  = 2
MAX_STEPS   = 100
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE       = torch.bfloat16

# ── Helper ─────────────────────────────────────────────────────
def log_memory(tag=""):
    alloc  = torch.cuda.memory_allocated()    / 1024**3
    resv   = torch.cuda.memory_reserved()     / 1024**3
    peak   = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[MEM]{' '+tag+' ' if tag else ' '}"
          f"allocated={alloc:.2f}GB  reserved={resv:.2f}GB  peak={peak:.2f}GB")

# ── 1. Load tokenizer & dataset ────────────────────────────────
print("=" * 60)
print("Bước 1 – OOM Demo: GPT-2 XL | 1 GPU | bf16 | NO Gradient Checkpointing")
print("=" * 60)

print("\n[1/4] Loading tokenizer & dataset ...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

dataset   = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

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

# ── 2. Load model in bf16 ──────────────────────────────────────
print(f"\n[2/4] Loading {MODEL_NAME} in bf16 ...")
torch.cuda.reset_peak_memory_stats()
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, torch_dtype=DTYPE)
model.to(DEVICE)
log_memory("after model load")

# ── 3. Setup optimizer ─────────────────────────────────────────
print("\n[3/4] Setting up AdamW 8-bit optimizer ...")
print("  ⚠️  Gradient Checkpointing: DISABLED")
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=5e-5)
log_memory("after optimizer init")

# ── 4. Training loop ───────────────────────────────────────────
print(f"\n[4/4] Starting training — expecting OOM ...")
print("-" * 60)

model.train()
optimizer_step = 0
micro_step     = 0
start_time     = time.time()
accum_loss     = 0.0

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
            optimizer_step += 1

            if optimizer_step % 10 == 0:
                log_memory(f"step {optimizer_step}")
                print(f"  step={optimizer_step:4d}  loss={accum_loss*GRAD_ACCUM:.4f}")

            accum_loss = 0.0

            if optimizer_step >= MAX_STEPS:
                print(f"\n✅ Hoàn thành {MAX_STEPS} steps mà không OOM.")
                print("   → Activations không đủ lớn để trigger OOM với config này.")
                break

except torch.cuda.OutOfMemoryError as e:
    print("\n" + "=" * 60)
    print("💥 CUDA OUT OF MEMORY — OOM confirmed!")
    print("=" * 60)
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"   Peak VRAM allocated : {peak:.2f} GB")
    print(f"   T4 VRAM available   : 16.00 GB")
    print(f"   Overflow            : {max(peak-16, 0):.2f} GB")
    print(f"   Optimizer steps     : {optimizer_step}")
    print(f"   Micro steps         : {micro_step}")
    print("=" * 60)
    print("\nKết luận: Không có Gradient Checkpointing → activations tích lũy")
    print("          → OOM dù đã dùng bf16 + AdamW8bit.")

finally:
    log_memory("final")
    print(f"\nTotal time: {time.time() - start_time:.1f}s")
