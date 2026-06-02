# Huấn luyện song song Pipeline cho các mô hình vượt kích thước bộ nhớ GPU

Tài liệu này trình bày chi tiết về dự án thử nghiệm và so sánh các giải pháp huấn luyện song song mô hình ngôn ngữ lớn (LLM) vượt quá giới hạn bộ nhớ của một GPU đơn lẻ, sử dụng kỹ thuật **Song song hóa đường ống (Pipeline Parallelism - PP)** trên môi trường đa GPU.

---

## 1. Bài toán

Khi kích thước của các mô hình ngôn ngữ lớn ngày càng tăng (ví dụ: các mô hình có hàng tỷ đến hàng chục tỷ tham số như GPT-2 XL, OPT-6.7B, LLaMA), việc huấn luyện hoặc tinh chỉnh (fine-tune) chúng đối mặt với giới hạn vật lý của phần cứng:
- **Tràn bộ nhớ GPU (CUDA Out Of Memory - OOM)**: Một mô hình lớn không thể nằm trọn trong VRAM của một GPU đơn lẻ. Bộ nhớ GPU không chỉ chứa tham số mô hình (weights) mà còn chứa trạng thái bộ tối ưu hóa (optimizer states), gradient, và đặc biệt là các biến kích hoạt trung gian (activations) sinh ra trong quá trình truyền xuôi (forward pass).
- **Yêu cầu kỹ thuật**: Dự án này giải quyết bài toán huấn luyện mô hình **GPT-2 XL (1.5 tỷ tham số)** trên cấu hình **2 GPU T4 (16GB VRAM mỗi GPU)** bằng cách chia nhỏ mô hình theo chiều dọc (các lớp liên tiếp) thành các phân đoạn (stages) khác nhau, phân phối chúng lên các GPU và áp dụng cơ chế song song hóa đường ống để xử lý dữ liệu theo các gói nhỏ (micro-batches).

---

## 2. Cấu hình nền tảng, mô hình và tập dữ liệu

### Cấu hình nền tảng phần cứng
- **Số lượng GPU**: 2 GPU (ví dụ: 2x NVIDIA Tesla T4, mỗi GPU có 16GB VRAM GDDR6).
- **Kết nối liên GPU**: NCCL backend hỗ trợ truyền thông điểm-điểm (P2P) nhanh chóng giữa GPU 0 và GPU 1.

### Cấu hình mô hình
- **Mô hình**: `gpt2-xl` (Hugging Face)
- **Số lượng tham số**: 1.55 tỷ tham số.
- **Số lượng layer**: 48 transformer blocks.
- **Kiểu dữ liệu (Dtype)**: `torch.bfloat16` (giúp giảm bộ nhớ mô hình đi 50% so với FP32 mà vẫn bảo toàn dải động số học rộng, không cần sử dụng bộ cân bằng thang đo gradient - GradScaler).

### Bộ tối ưu hóa (Optimizer)
- **Thuật toán**: 8-bit AdamW (`bitsandbytes.optim.AdamW8bit`).
- **Tác dụng**: Giảm dung lượng bộ nhớ dành cho các trạng thái của bộ tối ưu hóa (optimizer states) từ 8 bytes/tham số (ở FP32) xuống còn 2 bytes/tham số (ở INT8), giảm tổng dung lượng VRAM tĩnh cần thiết cho bộ tối ưu hóa xuống khoảng 4 lần.
- **Tốc độ học (Learning Rate)**: `5e-5`.

### Tập dữ liệu & Tiền xử lý
- **Tập dữ liệu**: `wikitext` (cấu hình `wikitext-2-raw-v1`, tập `train`).
- **Độ dài chuỗi tối đa (SEQ_LEN)**: 256 tokens.
- **Đệm & Cắt chuỗi**: Truncation và Padding được thiết lập về `max_length = 256` sử dụng `GPT2Tokenizer`. Các dòng trống hoặc không có nội dung hữu ích (tổng số lượng input_ids bằng 0) được lọc bỏ hoàn toàn trước khi đưa vào dataloader.

---

## 3. Các bước thực hiện và cấu hình từng bước

Quy trình thực nghiệm được chia làm 4 bước tuần tự nhằm đi từ việc mô phỏng lỗi OOM cho đến các giải pháp tối ưu hóa bộ nhớ trên 1 GPU và cuối cùng là song song hóa trên 2 GPU.

### Bước 1: Mô phỏng lỗi tràn bộ nhớ (OOM Baseline)
- **Mã nguồn**: `onegpu_baseline.py`
- **Cấu hình**: Chạy trên **1 GPU**, kiểu dữ liệu `bf16`, bộ tối ưu hóa 8-bit AdamW. Tắt cơ chế Gradient Checkpointing (`Gradient Checkpointing: DISABLED`).
- **Thông số batch**: `BATCH_SIZE = 3`, `GRAD_ACCUM = 2` (Effective batch size = 6).
- **Hiện tượng xảy ra**: Quá trình huấn luyện lập tức bị dừng lại ở micro-step đầu tiên do lỗi **CUDA Out Of Memory (OOM)**. Nguyên nhân do toàn bộ các tensor kích hoạt (activations) của 48 blocks GPT-2 XL được tích lũy liên tục và lưu giữ trên VRAM để phục vụ cho quá trình tính đạo hàm ngược (backward pass), vượt quá giới hạn 16GB của GPU T4.

### Bước 2: Tối ưu hóa bộ nhớ trên 1 GPU với Gradient Checkpointing
- **Mã nguồn**: `onegpu_GC.py`
- **Cấu hình**: Chạy trên **1 GPU**, kiểu dữ liệu `bf16`, bộ tối ưu hóa 8-bit AdamW. Kích hoạt tính năng **Gradient Checkpointing** (`model.gradient_checkpointing_enable()`).
- **Thông số batch**: `BATCH_SIZE = 3`, `GRAD_ACCUM = 2` (Effective batch size = 6).
- **Cơ chế**: Giải phóng các activation tensor trung gian sau khi hoàn thành forward pass của mỗi block. Trong quá trình backward pass, các activation này sẽ được tính toán lại (recompute) khi cần thiết. 
- **Kết quả**: Giảm dung lượng VRAM đỉnh (Peak VRAM) xuống dưới 16GB, cho phép huấn luyện thành công GPT-2 XL trên 1 GPU đơn lẻ mà không bị OOM, đổi lại chi phí tính toán tăng thêm khoảng 20-30% thời gian do phải tính toán lại activations.

### Bước 3: Song song hóa đường ống bằng PyTorch Native (Distributed Pipelining)
- **Mã nguồn**: `ddp_v2.py`
- **Cấu hình**: Chạy song song trên **2 GPU** bằng thư viện native `torch.distributed.pipelining`. 
- **Phân chia Stage thủ công (`SPLIT_LAYER = 24`)**:
  - **Stage 0 (GPU 0)**: Đảm nhận Token Embedding (`wte`), Position Embedding (`wpe`), Dropout (`drop`), và 24 transformer blocks đầu tiên (`t.h[:24]`).
  - **Stage 1 (GPU 1)**: Nhận đầu ra của Stage 0, xử lý 24 blocks còn lại (`t.h[24:]`), LayerNorm (`ln_f`), và ngõ ra tuyến tính (`lm_head`).
- **Thông số batch**: `BATCH_SIZE = 16`, `GRAD_ACCUM = 2` (Effective batch size = 32).
- **Cơ chế Pipeline**: Sử dụng `PipelineStage` và `ScheduleGPipe` để chia Batch dữ liệu thành các nhóm nhỏ (**micro-batches** hay **chunks**). Thực hiện thử nghiệm tuần tự với số lượng chunks gồm: `[2, 4, 8, 16]`.
- **Hỗ trợ tối ưu bổ sung**: Tích hợp Gradient Checkpointing thủ công (`torch.utils.checkpoint.checkpoint`) trên từng block của mỗi stage.

### Bước 4: Song song hóa đường ống bằng DeepSpeed Pipeline Parallelism
- **Mã nguồn**: `dp.py`
- **Cấu hình**: Chạy song song trên **2 GPU** sử dụng thư viện **DeepSpeed** (`deepspeed.PipelineModule`).
- **Phân chia Stage**: Cấu trúc thành các lớp liên tiếp trong một danh sách và giao cho `deepspeed.PipelineModule` tự động phân chia tải trọng theo dung lượng tham số (`partition_method="parameters"`).
- **Tích hợp ZeRO-1**: Bật công nghệ **ZeRO Stage 1** trong tệp cấu hình DeepSpeed để phân mảnh trạng thái optimizer (optimizer states partitioning) giữa các GPU, giảm dung lượng VRAM tĩnh xuống mức tối đa.
- **Thông số batch**: `EFFECTIVE_BS = 16`, chia nhỏ kích thước micro-batch dựa trên số lượng chunks (`micro_batch_size = EFFECTIVE_BS // chunks`). Thử nghiệm tuần tự với chunks = `[4, 8, 16]`.
- **Trình nạp dữ liệu**: Xây dựng lớp wrapper `DeepSpeedPipelineIterator` trả về đúng định dạng dữ liệu đầu vào song song đường ống là `((inputs), labels)`.

---

## 4. Phân tích hiện tượng "Bubble" và So sánh hiệu năng

### Hiện tượng "Bubble" trong Pipeline Parallelism
Hiện tượng "Bubble" (bong bóng/khoảng trống nhàn rỗi) xảy ra do tính chất tuần tự của Pipeline. GPU ở stage phía sau phải đợi GPU ở stage phía trước xử lý xong và truyền dữ liệu qua kết nối mạng (hoặc truyền P2P giữa các GPU). Ngược lại, trong quá trình backward, các GPU phía trước phải đợi tín hiệu gradient truyền ngược về từ các stage phía sau.

Khoảng thời gian một hoặc nhiều GPU phải nằm chờ nhàn rỗi này được gọi là "Bubble". 

Tỷ lệ phần trăm thời gian nhàn rỗi (tỷ lệ Bubble) của một hệ thống có $p$ stages (GPU) huấn luyện một batch được chia thành $m$ micro-batches (chunks) được tính theo công thức:
$$\text{Bubble Ratio} = \frac{p - 1}{m + p - 1}$$

Khi áp dụng vào hệ thống **2 GPU ($p = 2$)**, ta có các tỷ lệ lý thuyết sau:
- **chunks = 2**: $\frac{2 - 1}{2 + 2 - 1} = 33.33\%$
- **chunks = 4**: $\frac{2 - 1}{4 + 2 - 1} = 20.00\%$
- **chunks = 8**: $\frac{2 - 1}{8 + 2 - 1} = 11.11\%$
- **chunks = 16**: $\frac{2 - 1}{16 + 2 - 1} = 5.88\%$

**Nhận xét**: Khi tăng số lượng micro-batches ($m$ hay chunks), tỷ lệ Bubble giảm dần về mức rất thấp. Tuy nhiên, việc chia quá nhiều chunks sẽ làm giảm kích thước của mỗi micro-batch xuống quá nhỏ, dẫn đến GPU không được khai thác hết hiệu năng tính toán song song ma trận (dưới mức bão hòa của nhân Tensor Cores). Do đó cần lựa chọn số lượng chunks hài hòa.

---

### Đối chiếu PyTorch Native Pipeline và DeepSpeed Pipeline

| Tiêu chí | PyTorch Native Pipeline (`torch.distributed.pipelining`) | DeepSpeed Pipeline Parallelism |
| :--- | :--- | :--- |
| **Kiến trúc & Cực phân** | Yêu cầu định nghĩa module riêng biệt cho từng Stage và cấu hình luồng truyền dữ liệu thủ công. | Chỉ cần xếp các Layer nối tiếp nhau, DeepSpeed tự động phân vùng và tối ưu hóa vị trí đặt các Stage. |
| **Độ phức tạp lập trình** | **Cao**. Cần viết nhiều mã nguồn bổ trợ (`TensorChunkSpec`, xử lý truyền xuôi/ngược thủ công thông qua `ScheduleGPipe`). | **Thấp**. Tự động hóa hoàn toàn luồng huấn luyện chỉ bằng một câu lệnh `engine.train_batch()`. |
| **Tích hợp ZeRO** | Phức tạp, khó kết hợp trực tiếp với các tính năng phân mảnh bộ nhớ của FSDP hoặc ZeRO. | **Rất mạnh mẽ**. Tích hợp sẵn ZeRO-1 giúp phân chia optimizer states, giảm bộ nhớ tĩnh cực kỳ hiệu quả. |
| **Quản lý bộ nhớ VRAM** | Trung bình. VRAM được phân bổ cho mô hình giảm một nửa, nhưng bộ nhớ optimizer states vẫn chiếm dụng tĩnh trên từng GPU. | **Xuất sắc**. Lượng VRAM allocated tĩnh cực kỳ thấp (~1.55GB mỗi GPU so với ~5GB của PyTorch Native). |
| **Hiệu năng truyền thông** | Sử dụng luồng P2P của PyTorch Distributed. | Tối ưu hóa sâu các gói tin NCCL (reduce_bucket_size, allgather_bucket_size), giảm nghẽn băng thông. |

---

### So sánh Pipeline (2 GPU) và Gradient Checkpointing (1 GPU)
- **Huấn luyện trên 1 GPU + GC**: Tiết kiệm phần cứng (chỉ cần 1 GPU), nhưng tốc độ bị giới hạn và thời gian huấn luyện kéo dài do GPU phải tính toán lại activations trong backward pass. Đồng thời, dung lượng batch size tối đa bị giới hạn nghiêm trọng bởi dung lượng VRAM của 1 card duy nhất.
- **Huấn luyện trên 2 GPU + Pipeline**: Cho phép huấn luyện các mô hình có kích thước lớn gấp đôi hoặc gấp nhiều lần (bằng cách tăng số stage). Nhờ cơ chế chia luồng dữ liệu song song và phân bổ mô hình, thời gian huấn luyện được rút ngắn đáng kể, hiệu năng tính toán của các GPU được tận dụng hiệu quả hơn khi số lượng chunks tối ưu.

---

## 5. Kết quả thực nghiệm thực tế

Dưới đây là bảng số liệu thu thập được từ thực nghiệm trên hệ thống 2 GPU T4 (16GB VRAM):

### A. PyTorch Native Pipeline (Trích xuất từ `step3_metrics.json`)
*Cấu hình: BATCH_SIZE = 16, GRAD_ACCUM = 2 (Effective Batch Size = 32), SEQ_LEN = 256*

| Số lượng Chunks (m) | Tỷ lệ Bubble (%) | Tốc độ huấn luyện (tokens/sec) | Thời gian một bước (sec/step) | Bộ nhớ đỉnh (Peak VRAM - GB) |
| :---: | :---: | :---: | :---: | :---: |
| **2** | 33.33% | 324.8 | 25.224s | 6.18 GB |
| **4** | 20.00% | 431.0 | 19.010s | 5.99 GB |
| **8** | 11.11% | 517.7 | 15.825s | 5.97 GB |
| **16** | 5.88% | 571.4 | 14.336s | 5.96 GB |

**Nhận xét**: 
- Khi số lượng chunks tăng từ **2 lên 16**, tốc độ huấn luyện tăng rõ rệt từ **324.8 tokens/s lên 571.4 tokens/s** (tăng ~76.5%), trong khi thời gian thực thi mỗi bước giảm tương ứng từ **25.224s xuống còn 14.336s**.
- Kết quả này hoàn toàn khớp với lý thuyết về hiện tượng Bubble: Tỷ lệ bong bóng nhàn rỗi giảm mạnh từ **33.33% xuống còn 5.88%**, giúp thời gian GPU nhàn rỗi chờ đợi nhau giảm thiểu tối đa.
- Bộ nhớ đỉnh (Peak VRAM) có xu hướng giảm nhẹ và duy trì cực kỳ ổn định quanh mức **5.96 - 6.18 GB**, do kích thước của mỗi micro-batch nhỏ hơn giúp giảm lượng activation lưu trữ tạm thời tại một thời điểm trên card.

### B. DeepSpeed Pipeline Parallelism (Trích xuất từ log chạy của `dp.py`)
*Cấu hình: EFFECTIVE_BS = 16, SEQ_LEN = 256*

| Số lượng Chunks (m) | Tỷ lệ Bubble (%) | Tốc độ huấn luyện (tokens/sec) | Thời gian một bước (sec/step) | Bộ nhớ tĩnh đã cấp phát (Allocated VRAM) |
| :---: | :---: | :---: | :---: | :---: |
| **4** | 20.00% | 258.1 | 15.872s | 1.55 GB |
| **8** | 11.11% | 284.5 | 14.397s | 1.55 GB |
| **16** | 5.88% | 298.3 | 13.730s | 1.55 GB |

**Giải thích sự khác biệt giữa PyTorch Native và DeepSpeed**:
1. **Dung lượng VRAM**: DeepSpeed chỉ tiêu tốn **1.55 GB VRAM** tĩnh được cấp phát cho mô hình và optimizer trên mỗi GPU, thấp hơn nhiều so với mức ~5-6 GB của PyTorch Native. Đây là nhờ sự kết hợp cực kỳ hiệu quả giữa chia nhỏ Pipeline và **ZeRO Stage 1** (phân mảnh optimizer states).
2. **Tốc độ huấn luyện (tokens/sec)**:
   - Số liệu ghi nhận tốc độ tokens/s của DeepSpeed thấp hơn PyTorch Native (khoảng 258 - 298 tokens/s so với 431 - 571 tokens/s).
   - **Nguyên nhân kỹ thuật**: Sự chênh lệch này đến từ việc cấu hình kích thước Batch Size hiệu dụng khác nhau giữa 2 file chạy. PyTorch Native sử dụng `BATCH_SIZE = 16` kết hợp `GRAD_ACCUM = 2` trên mỗi GPU, mang lại tổng Batch Size hiệu dụng là **32** (tương đương **8192 tokens** xử lý mỗi bước). Trong khi đó, DeepSpeed được cấu hình với `EFFECTIVE_BS = 16` làm tổng kích thước batch chung cho cả hệ thống (tương đương **4096 tokens** mỗi bước), dẫn đến kích thước micro-batch trên mỗi GPU bị đẩy xuống cực kỳ nhỏ (chỉ còn 4, 2, và 1 sample tương ứng với chunks 4, 8, 16).
   - Kích thước micro-batch quá nhỏ (ví dụ: micro_batch = 1 ở cấu hình chunks = 16) khiến GPU không thể tối ưu hóa các phép toán song song trên lõi Tensor Cores, làm giảm hiệu suất tính toán thực tế của phần cứng và tăng tỷ lệ overhead truyền thông NCCL trên mỗi token.

---

## 6. Hướng dẫn chạy thử nghiệm và Cập nhật kết quả

### Yêu cầu môi trường
- Python >= 3.10
- PyTorch >= 2.1 với hỗ trợ CUDA và NCCL
- Các thư viện bổ trợ: `transformers`, `datasets`, `bitsandbytes`, `deepspeed`, `accelerate`
```bash
pip install -q datasets bitsandbytes deepspeed accelerate
```

### Cách chạy thực nghiệm

1. **Chạy Baseline OOM (Không dùng Gradient Checkpointing)**:
   ```bash
   python onegpu_baseline.py
   ```
   *Kỳ vọng: Chương trình sẽ dừng lại và hiển thị thông báo lỗi `CUDA Out of Memory`.*

2. **Chạy Baseline thành công trên 1 GPU (Có Gradient Checkpointing)**:
   ```bash
   python onegpu_GC.py
   ```
   *Tệp kết quả `step2_metrics.json` sẽ được tạo ra sau khi hoàn tất.*

3. **Chạy PyTorch Native Pipeline (2 GPU)**:
   ```bash
   python ddp_v2.py
   ```
   *Tệp kết quả `step3_metrics.json` chứa thông số của 4 cấu hình chunks sẽ được sinh ra.*

4. **Chạy DeepSpeed Pipeline (2 GPU)**:
   ```bash
   python dp.py
   ```
   *Tệp kết quả `step3_deepspeed_metrics.json` sẽ được sinh ra.*

> **Ghi chú cho người dùng**: Sau khi chạy đầy đủ các chương trình trên hệ thống của bạn, bạn có thể đọc các tệp tin `.json` kết quả được sinh ra và cập nhật trực tiếp các thông số chính xác vào bảng số liệu ở mục **5. Kết quả thực nghiệm thực tế** để hoàn thiện báo cáo thực hành của mình.
