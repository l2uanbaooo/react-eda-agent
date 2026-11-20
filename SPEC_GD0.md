# GĐ0 — Đặc tả: ReAct Agent tự động hóa phân tích dữ liệu

> Phiên bản đặc tả: v0.2 (2026-11-20). Dự án dựng lại từ đầu, không dùng lại code cũ. Thư mục dự án: `C:\projects\eda_agent`.
> Trạng thái: ĐÃ CHỐT (chờ người dùng gõ "xong"). Chưa có chỉ số nào được đo; mọi con số mục tiêu dưới đây là chỉ tiêu, không phải kết quả.
>
> Thay đổi v0.1 → v0.2: so sánh ChatGPT 12 task + mẫu bảng (mục 11); judge = `qwen2.5:14b-instruct` + chấm tay đối chiếu bằng ChatGPT (mục 12); dự phòng đáp án từ dòng `FINAL:` và phân loại lỗi (mục 3); tool `python_repl` tự bọc bằng `@tool` (mục 1, R9).

---

## 1. Hai thay đổi so với đề gốc (ghi lại nguyên văn vào README ở GĐ7)

| Đề gốc | Thay bằng | Lý do | Hệ quả cần khai báo |
|---|---|---|---|
| Sandbox E2B (cloud, trả phí) | **Docker** chạy ở máy local (Docker Desktop + WSL2); dự phòng: **subprocess có timeout** | E2B trả phí; Docker miễn phí và cách ly được mạng, RAM, CPU, ghi file | Mỗi kết quả eval ghi trường `sandbox = docker \| subprocess`. Chế độ subprocess KHÔNG cách ly mạng/FS, chỉ dùng khi bắt buộc |
| LLM chạy cùng máy | **Qwen2.5-Coder 14B chạy trên Kaggle GPU** bằng Ollama, máy local gọi qua tunnel HTTP (cloudflared) | GTX 1650 4GB không chạy được 14B (bản Q4_K_M cần ~9GB VRAM) | Độ trễ mạng; tunnel có thể đứt → llm.py có health check + retry/backoff, runner có resume |

Thêm một điểm phải nói thẳng (quy tắc 8): `PythonREPLTool` của `langchain_experimental` **chạy code ngay trên máy host**, không an toàn. Vì vậy dự án KHÔNG dùng bất kỳ tool REPL dựng sẵn nào. Tool `python_repl` là một hàm do dự án tự viết, bọc bằng decorator `@tool` (từ `langchain_core.tools`), bên trong chỉ gọi `sandbox.run(code)`. Vòng lặp ReAct gọi thẳng `python_repl.invoke({"code": code})`; LangChain không điều khiển agent. Phiên bản `langchain-core` được pin trong `requirements.txt` ở GĐ3 và `tests/test_tool.py` chạy trên đúng phiên bản đã pin (kiểm tra `name`, `args` schema, `.invoke()` trả output của sandbox). README sẽ nêu rõ điều này.

---

## 2. Bảng đối chiếu: yêu cầu đề → hạng mục → minh chứng

Ký hiệu cột GĐ: giai đoạn thực hiện. Cột "Minh chứng" là thứ kiểm tra được (file, test, lệnh, số liệu).

### 2.1 Mục đích và công nghệ

| # | Yêu cầu đề | Hạng mục / file | GĐ | Minh chứng |
|---|---|---|---|---|
| R1 | Agent viết code Python | `agent.py`: mỗi bước model trả đúng 1 khối ```python``` hoặc 1 dòng `FINAL:` | 4 | Log JSONL có trường `code` ở từng bước; test parser `tests/test_agent.py` |
| R2 | Chạy code thật | `sandbox/` → DockerSandbox (chính), SubprocessSandbox (dự phòng) | 3 | `tests/test_sandbox.py` chạy code thật, đọc stdout |
| R3 | Đọc kết quả | Observation = stdout/stderr đã cắt độ dài, đưa lại vào prompt | 4 | Log JSONL trường `observation`, `observation_truncated` |
| R4 | Tự sửa lỗi (self-debug) | Prompt sửa lỗi ngắn: code lỗi + traceback rút gọn + gợi ý theo loại lỗi; tối đa 3 retry | 4 | Log `retry_index`; chỉ số self-debug success ở GĐ5 |
| R5 | Sinh báo cáo có biểu đồ | `report.py`: template báo cáo cố định, HTML + Plotly tương tác | 7 | File HTML mở được offline; test kiểm tra đủ mục trong template |
| R6 | Dataset bất kỳ: csv, xlsx, tsv, json, parquet | `data.py::load_data()` duy nhất, có chọn sheet xlsx, tự dò `;` và thập phân `,` | 3 | `tests/test_data.py` nạp đủ 5 định dạng + các file bẫy |
| R7 | Qwen2.5-Coder 14B qua Ollama | `llm.py` gọi `/api/chat` (HTTP), model ghi trong `config.py` | 3 | `kaggle/health_check.py` in tên model + thời gian phản hồi |
| R8 | ReAct | Vòng lặp tự viết (~150–200 dòng), không AgentExecutor | 4 | `agent.py` đếm dòng; log có chuỗi Thought → Code → Observation |
| R9 | LangChain Python REPL tool | Hàm tự viết bọc `@tool` → `sandbox.run()`; không dùng tool dựng sẵn | 3 | `tests/test_tool.py` trên `langchain-core` đã pin: `.invoke()` chạy trong container, in `pip show langchain-core` vào log test |
| R10 | Sandbox an toàn | Docker: `--network none`, `--memory`, `--cpus`, `--pids-limit`, data mount READ-ONLY, output riêng, user non-root, timeout mỗi bước, dọn container | 3 | 4 test cách ly bắt buộc (mục 5) |

### 2.2 Chỉ tiêu

| # | Chỉ tiêu | Định nghĩa đo (chốt từ bây giờ để không đổi luật về sau) | GĐ | Minh chứng |
|---|---|---|---|---|
| K1 | Task completion ≥ 80% | Số task có checker trả PASS / tổng số task trong `tasks.json` (chỉ số chính thức, tính cả đáp án lấy từ dự phòng `FINAL:`). Báo cáo kèm: theo từng nhóm 1–6, và **K1-strict** (chỉ tính đáp án từ biến `answer`) | 5 | `eval/results/<run_id>/summary.json` |
| K2 | Self-debug success ≥ 70% | **Sự cố** = một lần chạy code bị exception hoặc timeout. Sự cố **được sửa** nếu trong ≤3 lần thử kế tiếp có một lần chạy không lỗi. Tỉ lệ = sự cố được sửa / tổng sự cố. Báo cáo thêm bản chặt: sự cố được sửa VÀ task cuối cùng PASS | 5 | `summary.json` có cả tử số, mẫu số |
| K3 | So sánh ChatGPT Code Interpreter | 12 task cố định (2 task/nhóm, đánh dấu `chatgpt_subset: true`), chạy tay trên ChatGPT Plus, ghi vào `eval/chatgpt_results.xlsx`, chấm bằng **cùng checkers**; nhóm 5 chấm tay theo cùng checklist cho cả hai bên (mục 11) | 7 | Bảng so sánh trong README, ghi ngày chạy + tên model ChatGPT hiển thị |

Lưu ý trung thực về K2: nếu tổng số sự cố quá ít (ví dụ < 20), tỉ lệ sẽ dao động lớn. Khi đó báo cáo kèm khoảng tin cậy (Wilson) và nói rõ cỡ mẫu, không làm tròn thành "đạt".

### 2.3 Đánh giá

| # | Tiêu chí | Cách đo | GĐ | Minh chứng |
|---|---|---|---|---|
| E1 | Task completion rate | K1 | 5 | `score.py` |
| E2 | Số lần retry | Trung bình + phân phối số lần chạy lỗi mỗi task; số bước ReAct mỗi task | 5 | `score.py` |
| E3 | Code quality (lint score) | `pylint` score (0–10) trên code đã chạy thành công của mỗi task, file rc cố định trong repo; kèm số lỗi `ruff` nhóm F (pyflakes) | 5 | `score.py`, file `eval/pylintrc` |
| E4 | Report comprehensiveness (LLM judge) | Judge `qwen2.5:14b-instruct` (khác model agent) chấm theo rubric 1–5 cho 6 tiêu chí (khớp 6 nhóm chức năng), + checklist tự động (đủ mục template, có số liệu, có biểu đồ, có câu cảnh báo tương quan ≠ nhân quả). Đối chiếu: một tập nhỏ báo cáo được chấm tay bằng ChatGPT với cùng rubric (mục 12). Tên judge ghi vào kết quả | 7 | `eval/judge_results.json`, `eval/judge_calibration.xlsx` |

### 2.4 Demo và cơ chế

| # | Yêu cầu | Hạng mục | GĐ | Minh chứng |
|---|---|---|---|---|
| D1 | Web UI Streamlit: upload CSV/XLSX → tự phân tích → báo cáo HTML Plotly | `app.py` (dùng chung `load_data()`, chọn sheet) | 7 | Ảnh chụp/video demo; nút tải HTML |
| D2 | Self-correction tối đa 3 retry | `config.MAX_RETRIES = 3` | 4 | Test với LLM giả (mock) trả code lỗi 4 lần → agent dừng đúng sau 3 retry |
| D3 | Structured final report template | `report.py` template cố định (mục 7) | 7 | Test kiểm tra đủ mục |
| D4 | Reflexion nhẹ, bật/tắt bằng config | `config.REFLEXION = True/False`; bài học lưu trong phạm vi 1 task | 4, 6 | Ablation GĐ6 |
| D5 | Log JSONL đầy đủ | Mỗi bước: thought, code, output, lỗi, retry, thời gian, sandbox, hash config | 4 | `logs/<run_id>/<task_id>.jsonl` |
| D6 | Một file config | `config.py` (đọc thêm biến môi trường để đổi URL tunnel không cần sửa code) | 3 | Đổi `SANDBOX_MODE` là chuyển chế độ A/B |

### 2.5 Bộ task (6 nhóm, mở rộng bằng thêm dòng vào `tasks.json`)

| Nhóm | Nội dung | Số task dự kiến | Loại checker chính |
|---|---|---|---|
| G1 Đọc hiểu dữ liệu | schema, nhận diện cột, kiểu, missing, sheet xlsx | 6 | so khớp tập/dict, số nguyên |
| G2 Thống kê | trung bình, tỉ lệ, groupby, so sánh nhóm | 6 | số có dung sai, dict |
| G3 Quan hệ | tương quan, phân phối, ngoại lệ, không nhầm nhân quả | 6 | số có dung sai + kiểm tra câu kết luận |
| G4 Chất lượng dữ liệu | missing, ngày sai định dạng, giá trị không hợp lệ | 6 | số nguyên chính xác, tập chỉ số |
| G5 Trực quan hóa | đúng biến, đúng nhóm, đúng trục thời gian | 6 | đọc JSON Plotly đã lưu |
| G6 Tổng hợp | kết hợp các nhóm trên | 6 | checker tổng hợp |
| **Tổng** | | **36** (tối thiểu 30) | |

---

## 3. Hợp đồng đáp án (để chấm tự động, không phụ thuộc văn phong)

Đây là quyết định thiết kế quan trọng nhất của eval, chốt từ GĐ0:

1. Mỗi task có trường `answer_format`, được ghép vào câu hỏi gửi agent, ví dụ: *"Set variable `answer` to a float rounded to 2 decimals."*
2. Agent gán biến `answer` trong namespace của sandbox. Khi agent trả `FINAL:`, harness đọc `answer` từ sandbox (serialize JSON). Câu kết luận sau `FINAL:` được lưu để dùng cho báo cáo và cho checker dạng câu chữ.
3. Task trực quan hóa: sandbox cung cấp sẵn hàm `save_fig(fig, name)` ghi Plotly figure ra `output/<name>.json` (+ `.html`). Checker đọc JSON để kiểm tra loại biểu đồ, cột trục x/y, nhóm màu, trục thời gian có sắp xếp tăng dần.
4. Task "không nhầm tương quan với nhân quả": checker kiểm tra giá trị số trong `answer` và câu FINAL không chứa khẳng định nhân quả (mẫu regex phủ định/khẳng định, có test cả câu đúng lẫn câu sai).

Lý do: chấm bằng regex trên văn bản tự do là nguồn lỗi chồng lỗi phổ biến nhất; đọc biến `answer` là tất định.

### 3.1 Dự phòng khi biến `answer` không được gán

Khi agent trả `FINAL:` mà `answer` chưa được gán (hoặc không serialize được JSON), harness lấy đáp án từ dòng `FINAL:` theo thứ tự cố định:

1. Nếu text sau `FINAL:` có `ANSWER_JSON: <json>` → parse JSON đó (cùng hàm `parse_answer_text()` dùng cho ChatGPT, mục 11).
2. Nếu `answer_type` là số và text chỉ chứa **đúng một** con số → lấy con số đó.
3. Còn lại → không lấy được đáp án → task FAIL.

Không đoán thêm (không lấy "số cuối cùng", không parse list/dict từ văn xuôi) để tránh chấm sai.

### 3.2 Phân loại lỗi (ghi vào log và summary, mỗi lỗi một mã)

| Mã | Nghĩa | Tính vào K2 (self-debug)? | Làm task FAIL? |
|---|---|---|---|
| `exec_error` | code chạy bị exception | có | không (còn retry) |
| `timeout` | code quá thời gian, bị kill | có | không (còn retry) |
| `retry_exhausted` | lỗi lần thứ 4 liên tiếp | — | có |
| `format_error` | model không trả khối python hoặc `FINAL:` | không | chỉ khi lặp quá giới hạn |
| `answer_missing` | `answer` không được gán, **dùng dự phòng FINAL** | không | không (nếu dự phòng parse được và đúng) |
| `answer_unparseable` | dự phòng cũng không lấy được | không | có |
| `max_steps` | hết số bước ReAct | không | có |
| `llm_error` | HTTP/tunnel lỗi sau khi đã retry hết | không | task đánh dấu `infra_error`, **không tính vào mẫu số K1**, chạy lại bằng resume |

`summary.json` báo số task theo từng mã. `answer_missing` được báo riêng để thấy agent có tuân thủ hợp đồng hay không.

---

## 4. Dataset

### 4.1 Tổng hợp (sinh bởi `make_datasets.py`, seed cố định, ground truth tính từ quá trình sinh)

| File | Định dạng | Nội dung | Bẫy cài sẵn |
|---|---|---|---|
| `sales.csv` | csv | đơn hàng: ngày, vùng, sản phẩm, số lượng, đơn giá, doanh thu | dòng trùng, missing, outlier doanh thu |
| `hr.xlsx` | xlsx nhiều sheet | `employees`, `departments`, `notes` | dữ liệu chính không ở sheet đầu; cần join 2 sheet |
| `sensors.tsv` | tsv | chuỗi thời gian cảm biến theo giờ | khoảng trống thời gian, giá trị đóng băng |
| `customers.json` | json (list records) | khách hàng | null, kiểu lẫn (số dạng chuỗi) |
| `eu_sales.csv` | csv `;`, thập phân `,` | doanh số châu Âu | đọc mặc định sẽ ra 1 cột / số thành chuỗi |
| `clinic_messy.xlsx` | xlsx | bệnh nhân | ngày nhiều định dạng, tuổi âm/999, "unknown" trong cột số |
| `transactions.parquet` | parquet | ~50k giao dịch, kiểu category | biến gây nhiễu tạo tương quan giả (confounder) |

`ground_truth.json`: đáp án chuẩn cho mọi task, kèm số lượng bẫy đã cài (ví dụ đúng 17 tuổi không hợp lệ) để checker G4 so chính xác.

### 4.2 Dataset thật (demo, tải một lần, lưu local vì sandbox không có mạng)

Titanic, California Housing, Penguins, Tips (nguồn: seaborn-data / scikit-learn). Chỉ dùng cho demo và vài task G6; ghi nguồn + giấy phép trong README.

---

## 5. Sandbox: 4 test cách ly bắt buộc (tiêu chí xong GĐ3)

| Test | Code chạy trong sandbox | Kỳ vọng |
|---|---|---|
| Không có mạng | `socket.create_connection(("1.1.1.1", 53), timeout=3)` | lỗi `OSError` |
| Data read-only | ghi file vào `/data` | lỗi `PermissionError` / `Read-only file system` |
| Timeout | `while True: pass` với timeout 5s | bị kill, trả lỗi timeout, sandbox khởi động lại được |
| Giữ trạng thái | bước 1 `x = 41`, bước 2 `print(x + 1)` | in `42` |

Bổ sung: user trong container không phải root (`os.getuid() != 0`); container bị xóa sau `close()` (kiểm bằng `docker ps -a`).

---

## 6. Hai chế độ chạy

| | (A) CHÍNH | (B) DỰ PHÒNG |
|---|---|---|
| Agent, UI, data, eval | máy local Windows | notebook Kaggle |
| Sandbox | DockerSandbox | SubprocessSandbox (timeout, không cách ly mạng) |
| LLM | Ollama trên Kaggle qua tunnel | Ollama `localhost:11434` trong Kaggle |
| Dùng cho | demo, eval chính thức | khi Docker/tunnel không dùng được |
| Đổi chế độ | `SANDBOX_MODE`, `OLLAMA_URL` trong config/biến môi trường | như trái |

---

## 7. Template báo cáo cuối (tiếng Việt)

1. Tổng quan dataset (nguồn, kích thước, sheet đã chọn)
2. Cấu trúc và kiểu dữ liệu
3. Chất lượng dữ liệu (missing, ngày sai, giá trị không hợp lệ, trùng lặp)
4. Thống kê mô tả và so sánh nhóm
5. Quan hệ giữa các biến (tương quan, phân phối, ngoại lệ) + câu cảnh báo "tương quan không phải nhân quả"
6. Biểu đồ (Plotly tương tác)
7. Kết luận và khuyến nghị (mỗi kết luận dẫn kèm con số)
8. Phụ lục: các bước agent đã chạy, số retry, sandbox, model, thời gian

---

## 8. Tham số LLM dự kiến (chốt ở GĐ3)

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| model | `qwen2.5-coder:14b` (Q4_K_M, ~9GB) | vừa 1 GPU T4 16GB của Kaggle |
| temperature | 0.1 | |
| seed | 42 | |
| num_ctx | 8192 | KV cache ~1.6GB, vẫn vừa T4 |
| num_predict | 1024 | chặn câu trả lời dài |
| keep_alive | `-1` hoặc `24h` | không unload model giữa các task |
| stream | true | tránh lỗi 524 do Cloudflare cắt request > 100s |
| cắt observation | ~2000 ký tự (đầu + cuối) | |
| ngân sách prompt | profile ≤ ~2500 token | |
| max bước ReAct / task | 10 | khác với retry |
| max retry liên tiếp | 3 | lần lỗi thứ 4 liên tiếp → task thất bại |
| judge model | `qwen2.5:14b-instruct` (Q4_K_M) | chạy ở **pha riêng** sau khi agent eval xong, để mỗi lúc chỉ 1 model 14B trên GPU |
| judge temperature / seed | 0 / 42 | |

---

## 9. Sổ rủi ro (rà trước khi viết code)

| Rủi ro | Biểu hiện | Xử lý sẵn |
|---|---|---|
| Encoding Windows | `UnicodeDecodeError` / chữ Việt lỗi | mọi `open()` dùng `encoding="utf-8"`; subprocess đặt `PYTHONIOENCODING=utf-8`, `PYTHONUTF8=1` |
| Đường dẫn Windows khi mount Docker | `invalid mode`, mount rỗng | dùng `--mount type=bind,src=<abs path>,dst=/data,readonly` với `pathlib` resolve; không dùng `-v C:\...:...` |
| Docker Desktop chưa chạy | `failed to connect to the docker API ... dockerDesktopLinuxEngine` (đã gặp trên máy này hôm nay) | `docker_sandbox.py` kiểm tra `docker info` trước, báo lỗi rõ ràng |
| Quyền ghi output khi user non-root | `PermissionError` trong `/output` | tạo thư mục output từ host trước khi mount; container chạy với uid cố định |
| Process treo / zombie | timeout không kill được | worker chạy code trong process con; timeout → kill process con, giữ container; nếu worker chết → khởi động lại và báo "state lost" cho agent |
| Ollama HTTP 500 / model chưa load | lỗi lần gọi đầu | health check `/api/tags` + gọi thử 1 token; retry backoff mũ (1, 2, 4, 8s) |
| Tunnel đứt / URL đổi | `ConnectionError`, 502/530 | URL đọc từ biến môi trường; runner lưu từng task và resume |
| Cloudflare quick tunnel | 403 do Host header; 524 sau 100s | `cloudflared tunnel --url http://localhost:11434 --http-host-header localhost:11434`; dùng stream |
| Tunnel công khai | người lạ gọi được Ollama | URL ngẫu nhiên, tắt notebook khi xong; ghi chú trong README |
| Kaggle | phiên tối đa 12h, quota GPU ~30h/tuần, cần bật Internet (xác minh SĐT) | resume; chạy eval theo lô |
| Tràn context | model quên hướng dẫn, lặp lại | cắt observation, giới hạn profile, giữ lịch sử ngắn (chỉ N bước gần nhất + tóm tắt) |
| Model trả sai định dạng | không có khối python, có 2 khối | parser lấy khối đầu tiên; không có gì → 1 lần nhắc định dạng (không tính là retry lỗi code) |
| Subprocess mode không an toàn | code ghi/xóa file host | chỉ dùng trong Kaggle; ghi rõ trong kết quả |
| 2 model 14B cùng lúc trên GPU | OOM / model bị unload liên tục | agent eval và judge chạy 2 pha tách biệt; `health_check.py` nhận tên model làm tham số |
| Bảng tay lưu bằng CSV trong Excel locale Việt Nam | Excel lưu dấu `;`, thập phân `,`, mã hóa ANSI → hỏng JSON trong ô | dùng file **.xlsx** cho mọi bảng nhập tay; đọc bằng `openpyxl` |
| API `@tool` đổi giữa các phiên bản langchain-core | `.run`/`.invoke` khác hành vi | pin phiên bản, test trên đúng bản pin, agent chỉ dùng `.invoke()` |
| Docker Desktop khởi động chậm | lỗi kết nối `dockerDesktopLinuxEngine` dù đã bật (gặp lúc chốt GĐ0) | GĐ3 có lệnh kiểm tra `docker info` trước mọi test; đợi đến khi engine báo `Running` |

---

## 10. Tiêu chí "xong" của GĐ0

- [x] Mọi yêu cầu của đề có ít nhất 1 dòng trong bảng mục 2, kèm GĐ và minh chứng đo được.
- [x] Hai thay thế (E2B → Docker, LLM local → Kaggle + tunnel) được ghi rõ.
- [x] Định nghĩa K1, K1-strict, K2, E3, E4 được chốt (không đổi sau khi đã có kết quả).
- [x] Hợp đồng đáp án (biến `answer`, `save_fig`, dự phòng `FINAL:`, phân loại lỗi) được chốt.
- [x] 3 câu hỏi làm rõ đã được trả lời (mục 11, 12, thư mục `C:\projects\eda_agent`).
- [ ] Bạn gõ "xong".

---

## 11. Giao thức so sánh với ChatGPT Code Interpreter

### 11.1 Chọn 12 task (chốt trước khi chạy agent, tránh chọn task "đẹp")

- Ở GĐ2, mỗi nhóm G1–G6 chọn đúng 2 task, gắn `"chatgpt_subset": true` trong `tasks.json`.
- Quy tắc chọn: trong mỗi nhóm, 1 task dùng dataset có bẫy định dạng (`eu_sales.csv`, `clinic_messy.xlsx`, `hr.xlsx`...) và 1 task dùng dataset sạch hơn. Danh sách được chốt cùng GĐ2, không đổi sau khi đã thấy kết quả agent.

### 11.2 Cách chạy (giống nhau cho cả 12 task)

1. Mỗi task mở **một cuộc trò chuyện mới**; tắt Memory và Custom Instructions (ghi vào bảng nếu không tắt được).
2. Upload đúng file dataset của task (đúng file agent dùng).
3. Dán câu hỏi = `question` + câu định dạng cố định (11.3). Không gợi ý thêm, không hỏi lại. Chỉ 1 lượt.
4. Copy nguyên dòng `ANSWER_JSON: ...` vào bảng; chụp màn hình toàn bộ câu trả lời, lưu `eval/chatgpt_screens/<task_id>.png`.
5. Nhóm 5 (biểu đồ): chấm tay theo checklist 11.5; tải ảnh biểu đồ nếu có.

### 11.3 Câu cố định về định dạng đáp án (dán nguyên văn sau câu hỏi)

```text
At the end of your reply, output exactly one line that starts with "ANSWER_JSON:" followed by a single valid JSON value in this format: <ANSWER_FORMAT>. Use a dot as the decimal separator. Do not write anything after that line.
```

`<ANSWER_FORMAT>` được thay bằng trường `answer_format` của task (ví dụ `a number rounded to 2 decimals`). Câu này được sinh tự động từ `tasks.json` vào cột `prompt_to_paste` của bảng, để không gõ tay sai.

Với nhóm 5, thêm câu: `Use Plotly if possible and show the chart.` (Agent cũng được yêu cầu dùng Plotly, nên điều kiện như nhau.)

Agent và ChatGPT được chấm trên **cùng một giá trị JSON** bằng **cùng checker**; khác nhau duy nhất là kênh lấy đáp án (biến `answer` vs dòng `ANSWER_JSON:`), cả hai đi qua `parse_answer_text()`.

### 11.4 Mẫu bảng ghi kết quả: `eval/chatgpt_results.xlsx`

File được sinh tự động ở GĐ2 (sheet `results` điền sẵn 5 cột đầu). Người chạy chỉ điền các cột còn lại.

| Cột | Ai điền | Ví dụ | Ghi chú |
|---|---|---|---|
| `task_id` | tự sinh | `G2_03` | |
| `group` | tự sinh | `G2` | |
| `dataset_file` | tự sinh | `eu_sales.csv` | file cần upload |
| `sheet` | tự sinh | `employees` | trống nếu không phải xlsx |
| `prompt_to_paste` | tự sinh | câu hỏi + câu 11.3 | copy nguyên ô này |
| `run_date` | tay | `2026-12-05` | YYYY-MM-DD |
| `chatgpt_model` | tay | tên model hiển thị trên giao diện | |
| `new_chat_memory_off` | tay | `Y` / `N` | |
| `answer_line` | tay | `ANSWER_JSON: 1234.56` | dán nguyên dòng; trống nếu ChatGPT không xuất |
| `n_visible_errors` | tay | `1` | số lần code ChatGPT báo lỗi rồi tự chạy lại (đối chiếu với retry của agent) |
| `c1`…`c6` | tay (chỉ G5) | `1` / `0` | checklist 11.5 |
| `screenshot` | tay | `G2_03.png` | |
| `notes` | tay | | bất thường (ChatGPT hỏi lại, từ chối, hết lượt...) |
| `checker_result` | `score.py` | `PASS` / `FAIL` / `NO_ANSWER` | không điền tay |

Sheet thứ hai `checklist_agent` có cùng cột `c1`…`c6` cho 12 task của agent (chỉ G5 cần điền), để chấm tay agent bằng cùng checklist.

### 11.5 Checklist biểu đồ nhóm 5 (áp dụng giống nhau cho agent và ChatGPT)

| Mã | Tiêu chí | Bắt buộc để PASS |
|---|---|---|
| c1 | Đúng loại biểu đồ cho câu hỏi (line cho xu hướng thời gian, bar cho so sánh nhóm, histogram/box cho phân phối, scatter cho quan hệ) | có |
| c2 | Đúng biến trục x | có |
| c3 | Đúng biến/đại lượng trục y (đúng phép tổng hợp: sum/mean/count như câu hỏi) | có |
| c4 | Đúng nhóm (màu/facet theo đúng cột yêu cầu; không thiếu/thừa nhóm) | có (nếu câu hỏi có nhóm) |
| c5 | Trục thời gian đúng: kiểu ngày, sắp xếp tăng dần, đúng độ phân giải (ngày/tháng) | có (nếu có trục thời gian) |
| c6 | Có tiêu đề và nhãn trục đọc được | không (chỉ ghi nhận) |

PASS nhóm 5 = mọi tiêu chí bắt buộc áp dụng được đều = 1. Với agent, điểm chính thức K1 vẫn dùng checker tự động trên JSON Plotly; checklist tay dùng cho bảng so sánh với ChatGPT, và ghi thêm **tỉ lệ đồng thuận** giữa checker tự động và checklist tay trên 2 task G5 của agent (để kiểm chứng checker).

---

## 12. LLM judge và chấm tay đối chiếu

- Judge chính: `qwen2.5:14b-instruct` qua Ollama trên Kaggle (khác model agent `qwen2.5-coder:14b`, giảm thiên lệch tự chấm). Cấu hình ở `config.JUDGE_MODEL`, ghi vào mọi kết quả judge.
- Rubric cố định (prompt lưu ở `eval/judge_prompt.txt`, không đổi sau khi đã chấm): 6 tiêu chí, mỗi tiêu chí 1–5 — (1) mô tả dữ liệu, (2) thống kê, (3) quan hệ + cảnh báo nhân quả, (4) chất lượng dữ liệu, (5) biểu đồ, (6) kết luận có số liệu dẫn chứng. Judge trả JSON `{"c1":..,"c6":..,"comment":".."}`; parse lỗi → retry 1 lần → ghi `judge_error`.
- Chấm tay đối chiếu: chọn ngẫu nhiên (seed cố định) **8 báo cáo** của agent, dán **cùng prompt rubric** + báo cáo vào ChatGPT (cuộc trò chuyện mới mỗi báo cáo, xóa thông tin hệ thống nào tạo báo cáo). Ghi vào `eval/judge_calibration.xlsx` (cột: `report_id`, `judge`, `c1`…`c6`, `total`, `run_date`, `chatgpt_model`).
- Báo cáo: tương quan Spearman và độ lệch trung bình tuyệt đối giữa Qwen-judge và ChatGPT-judge. Nếu lệch lớn, nói rõ trong README; không đổi rubric để "khớp" sau khi đã thấy điểm.
- Lưu ý thiên lệch: nếu sau này dùng ChatGPT chấm cả báo cáo do chính ChatGPT viết, phải ghi rõ đó là tự chấm.
