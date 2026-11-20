"""Generate eval/chatgpt_results.xlsx: the manual results sheet for ChatGPT (spec 11.4).

Usage (from the project root):
    .venv\\Scripts\\python.exe eval\\make_chatgpt_template.py           # create
    .venv\\Scripts\\python.exe eval\\make_chatgpt_template.py --force   # overwrite even if filled

The script refuses to overwrite a workbook that already contains manual
entries, so results typed in by hand cannot be lost by re-running it.
xlsx (not csv) on purpose: Excel with a Vietnamese locale saves CSV with ';'
and ',' decimals in the ANSI codepage, which corrupts JSON pasted in cells.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

EVAL_DIR = Path(__file__).resolve().parent
TASKS_FILE = EVAL_DIR / "tasks.json"
OUT_FILE = EVAL_DIR / "chatgpt_results.xlsx"

FORMAT_SENTENCE = ('At the end of your reply, output exactly one line that starts with '
                   '"ANSWER_JSON:" followed by a single valid JSON value in this format: {fmt}. '
                   'Use a dot as the decimal separator. Do not write anything after that line.')
PLOT_SENTENCE = "Use Plotly if possible and show the chart."
CAUSAL_SENTENCE = "Before the ANSWER_JSON line, state your conclusion in one or two sentences."

AUTO_COLS = ["task_id", "group", "dataset_file", "sheet", "prompt_to_paste"]
MANUAL_COLS = ["run_date", "chatgpt_model", "new_chat_memory_off", "answer_line", "final_text",
               "n_visible_errors", "c1", "c2", "c3", "c4", "c5", "c6", "screenshot", "notes"]
SCORE_COLS = ["checker_result"]
RESULT_COLS = AUTO_COLS + MANUAL_COLS + SCORE_COLS
CHECKLIST_COLS = ["task_id", "group", "c1", "c2", "c3", "c4", "c5", "c6", "notes"]

CHECKLIST = [
    ("c1", "Correct chart type (line=time trend, bar=group comparison, histogram/box=distribution, "
           "scatter=relationship)", "required"),
    ("c2", "Correct x-axis variable", "required"),
    ("c3", "Correct y variable and aggregation (sum/mean/count as asked)", "required"),
    ("c4", "Correct grouping (colour/facet by the requested column, no missing/extra groups)",
     "required if the question has groups"),
    ("c5", "Correct time axis: date type, ascending order, right resolution (day/month)",
     "required if there is a time axis"),
    ("c6", "Readable title and axis labels", "recorded only"),
]

INSTRUCTIONS = [
    "ChatGPT Code Interpreter comparison - protocol (spec section 11)",
    "1. One NEW chat per task. Turn off Memory and Custom Instructions (column new_chat_memory_off).",
    "2. Upload exactly the file in column dataset_file (from the datasets/ folder).",
    "3. Paste the cell prompt_to_paste unchanged. One turn only: no hints, no follow-up.",
    "4. Copy the whole ANSWER_JSON: line into answer_line (leave empty if ChatGPT did not print it).",
    "5. G3 causal task: copy the one or two conclusion sentences into final_text.",
    "6. G5 plot tasks: no ANSWER_JSON; fill c1..c6 with 1 / 0 / NA using the checklist below.",
    "7. n_visible_errors = number of code cells ChatGPT showed as failing before it continued.",
    "8. Save a full screenshot as eval/chatgpt_screens/<task_id>.png and write the name in screenshot.",
    "9. Do NOT fill checker_result: score.py writes it.",
    "",
    "Plot checklist (same for agent and ChatGPT). PASS = every applicable required item is 1.",
]


def load_tasks(path: Path = TASKS_FILE) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["tasks"]


def render_chatgpt_prompt(task: dict) -> str:
    parts = [f"Use the uploaded file {Path(task['file']).name}.", task["question"]]
    if task["checker"] == "plot":
        parts.append(PLOT_SENTENCE)
    else:
        if task["checker"] == "causal":
            parts.append(CAUSAL_SENTENCE)
        parts.append(FORMAT_SENTENCE.format(fmt=task["answer_format"]))
    return "\n\n".join(parts)


def has_manual_data(path: Path) -> bool:
    if not path.exists():
        return False
    wb = load_workbook(path, read_only=True)
    if "results" not in wb.sheetnames:
        return False
    ws = wb["results"]
    header = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    idx = [i for i, h in enumerate(header) if h in MANUAL_COLS]
    for row in ws.iter_rows(min_row=2, values_only=True):
        if any(row[i] not in (None, "") for i in idx if i < len(row)):
            return True
    return False


def _style_header(ws, ncols: int) -> None:
    fill = PatternFill("solid", fgColor="DDDDDD")
    for c in range(1, ncols + 1):
        cell = ws.cell(row=1, column=c)
        cell.font = Font(bold=True)
        cell.fill = fill
    ws.freeze_panes = "B2"


def build(out: Path = OUT_FILE, tasks_file: Path = TASKS_FILE, force: bool = False) -> Path:
    if has_manual_data(out) and not force:
        raise FileExistsError(f"{out} already contains manual entries; use --force to overwrite")
    subset = [t for t in load_tasks(tasks_file) if t.get("chatgpt_subset")]
    wb = Workbook()

    ws = wb.active
    ws.title = "results"
    ws.append(RESULT_COLS)
    for t in subset:
        ws.append([t["id"], t["group"], t["file"], t.get("sheet") or "", render_chatgpt_prompt(t)]
                  + [None] * (len(MANUAL_COLS) + len(SCORE_COLS)))
    _style_header(ws, len(RESULT_COLS))
    widths = {"prompt_to_paste": 70, "answer_line": 40, "final_text": 40, "notes": 30,
              "dataset_file": 28, "chatgpt_model": 18}
    for i, name in enumerate(RESULT_COLS, start=1):
        letter = ws.cell(row=1, column=i).column_letter
        ws.column_dimensions[letter].width = widths.get(name, 12)
        if name in ("prompt_to_paste", "final_text", "answer_line"):
            for r in range(2, len(subset) + 2):
                ws.cell(row=r, column=i).alignment = Alignment(wrap_text=True, vertical="top")
    last = len(subset) + 1
    yn = DataValidation(type="list", formula1='"Y,N"', allow_blank=True)
    flag = DataValidation(type="list", formula1='"1,0,NA"', allow_blank=True)
    ws.add_data_validation(yn)
    ws.add_data_validation(flag)
    col = {name: ws.cell(row=1, column=i).column_letter for i, name in enumerate(RESULT_COLS, start=1)}
    yn.add(f"{col['new_chat_memory_off']}2:{col['new_chat_memory_off']}{last}")
    for c in ("c1", "c2", "c3", "c4", "c5", "c6"):
        flag.add(f"{col[c]}2:{col[c]}{last}")

    ck = wb.create_sheet("checklist_agent")
    ck.append(CHECKLIST_COLS)
    for t in subset:
        ck.append([t["id"], t["group"]] + [None] * (len(CHECKLIST_COLS) - 2))
    _style_header(ck, len(CHECKLIST_COLS))
    flag2 = DataValidation(type="list", formula1='"1,0,NA"', allow_blank=True)
    ck.add_data_validation(flag2)
    flag2.add(f"C2:H{len(subset) + 1}")

    ins = wb.create_sheet("instructions")
    for line in INSTRUCTIONS:
        ins.append([line])
    ins.append(["code", "criterion", "rule"])
    for row in CHECKLIST:
        ins.append(list(row))
    ins.column_dimensions["A"].width = 110
    ins.column_dimensions["B"].width = 90
    ins.column_dimensions["C"].width = 32

    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Create the ChatGPT manual results workbook.")
    ap.add_argument("--out", default=str(OUT_FILE))
    ap.add_argument("--force", action="store_true", help="overwrite even if it has manual entries")
    args = ap.parse_args(argv)
    try:
        path = build(Path(args.out), force=args.force)
    except FileExistsError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 1
    n = sum(1 for t in load_tasks() if t.get("chatgpt_subset"))
    print(f"wrote {path} ({n} tasks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
