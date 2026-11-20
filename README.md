# EDA Agent: ReAct + Code Interpreter + Visualization

Agent tự động phân tích dữ liệu (EDA): viết code Python, chạy thật trong sandbox, đọc kết quả, tự sửa lỗi,
sinh báo cáo có biểu đồ, cho dataset csv / xlsx / tsv / json / parquet.
LLM: **Qwen2.5-Coder 14B** qua Ollama; vòng lặp **ReAct tự viết** (không dùng AgentExecutor).

> Trạng thái: GĐ0–GĐ3 xong (đặc tả, dữ liệu + ground truth, 36 task + checker, hạ tầng). GĐ4 (agent) đang làm.
> **Chưa có kết quả đánh giá nào**: mọi chỉ tiêu (≥80% task completion, ≥70% self-debug) chỉ được công bố
> sau khi chạy eval thật (GĐ5). README đầy đủ (kết quả, so sánh ChatGPT, demo) hoàn thiện ở GĐ7.

## Thay đổi so với đề gốc

| Đề gốc | Dự án dùng | Lý do |
|---|---|---|
| Sandbox E2B (trả phí) | **Docker** local (Docker Desktop + WSL2); dự phòng **subprocess có timeout** | Miễn phí; cách ly mạng, RAM, CPU, ghi file. Kết quả eval luôn ghi `sandbox=docker\|subprocess`; chế độ subprocess không cách ly. |
| LLM chạy cùng máy | Qwen2.5-Coder 14B chạy trên **Kaggle GPU**, máy local gọi qua tunnel Cloudflare | GTX 1650 4GB không chạy được 14B |
| LangChain Python REPL tool | Tool `python_repl` **tự bọc bằng `@tool`** (langchain-core, đã pin) gọi vào sandbox | `PythonREPLTool` của langchain_experimental chạy code ngay trên máy host, không an toàn |

## Ghi chú thiết kế ảnh hưởng tới cách đọc kết quả

- **Sheet Excel mặc định.** Chỉ có một `load_data()`. Khi không truyền `sheet=`: UI và báo cáo đọc sheet
  **lớn nhất**; sandbox của agent (mọi cấu hình eval) đọc sheet **đầu tiên**, giống `pd.read_excel`, và không trả
  danh sách sheet. Vì vậy gợi ý về sheet chỉ đến từ **pre-profiling** (dòng `SHEETS:` liệt kê mọi sheet).
  Ở cấu hình **baseline** (profiling tắt) agent **không nhận gợi ý sheet nào** từ `load_data` hay profiling.
  Lợi thế của agent ở **G1_02** (và các task trên file Excel nhiều sheet), so với baseline hay với ChatGPT,
  đến từ pre-profiling. Kiểm chứng: `tests/test_sandbox.py::test_sandbox_load_data_gives_no_sheet_hint`.
- **JSON giữ kiểu thô**: `load_data` đọc JSON bằng `json.load` + DataFrame; `pd.read_json` sẽ âm thầm đổi tuổi
  lưu dạng chuỗi `"34"` thành số và xóa mất bẫy dữ liệu.
- **pandas 2.2.3** (không phải 3.x): pandas 3 đổi kiểu chuỗi mặc định và copy-on-write; Qwen2.5-Coder được
  huấn luyện chủ yếu trên code pandas 2.x. Image Docker, venv và bộ sinh ground truth dùng cùng phiên bản.

## Cài đặt (Windows, PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
docker build -t eda-sandbox:1.1 -f sandbox/Dockerfile .
.\.venv\Scripts\python.exe datasets\fetch_real.py --verify
.\.venv\Scripts\python.exe -m pytest -rs
```

- `requirements-lock.txt` cố định **mọi** gói (kể cả gián tiếp như pydantic). `requirements.txt` +
  `sandbox/requirements-sandbox.txt` là các pin trực tiếp; image được build với lock làm constraints.
- Bộ task (`eval/tasks.json`) và dữ liệu (`datasets/`) đã **đóng băng**: không chạy lại `make_datasets.py`.
- Các file requirements phải giữ ASCII (pip trên Windows đọc bằng cp1252).

## LLM trên Kaggle

1. Mở `kaggle/ollama_kaggle.ipynb` trên Kaggle (GPU T4, Internet ON), chạy từ trên xuống, để ô cuối chạy.
2. Trên máy local:
   ```powershell
   $env:EDA_OLLAMA_URL = "https://<url>.trycloudflare.com"
   .\.venv\Scripts\python.exe kaggle\health_check.py --judge
   ```
3. URL tunnel là công khai khi tunnel chạy: không chia sẻ, tắt session khi xong.

## Cấu hình

Mọi cấu hình nằm trong `config.py`; ghi đè bằng biến môi trường `EDA_<TÊN>` (ví dụ `EDA_SANDBOX_MODE=subprocess`
cho chế độ dự phòng chạy toàn bộ trên Kaggle).

## Cấu trúc

```
config.py  llm.py  data.py  agent.py (GĐ4)  report.py, app.py (GĐ7)
sandbox/   base, docker_sandbox, subprocess_sandbox, worker, tool (@tool python_repl), Dockerfile
kaggle/    ollama_kaggle.ipynb, health_check.py
datasets/  make_datasets.py, fetch_real.py, synthetic/, real/, ground_truth.json
eval/      tasks.json, checkers.py, make_chatgpt_template.py, chatgpt_results.xlsx
tests/     pytest cho từng module
SPEC_GD0.md  đặc tả và các quyết định đã chốt
```
