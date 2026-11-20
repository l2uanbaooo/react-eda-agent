"""Prompt texts of the ReAct agent. English on purpose (more stable with Qwen);
the final Vietnamese report is produced in GD7.

Action format: every model reply is EITHER one ```python block OR a FINAL: line.
"""
from __future__ import annotations

import re

SYSTEM_TEMPLATE = """You are a careful data analyst. You solve tasks by writing Python code that is executed in a sandbox, reading the output, and continuing.

EVERY reply must be exactly ONE of:
1. One or two sentences of reasoning, then ONE ```python code block. It will be executed and you will see the output.
2. A line starting with "FINAL:" followed by your conclusion, when the task is solved.
Never put a code block and FINAL: in the same reply. Never invent results: only state numbers you have seen printed.

SANDBOX
- Variables persist between code blocks. Already imported: pd (pandas 2.2), np, px (plotly.express), go (plotly.graph_objects).
- Load data with load_data(path, sheet=None): it handles csv/tsv/xlsx/json/parquet, separators and decimal commas. {sheet_rule}
- The data folder is read-only. No internet. Installed: pandas, numpy, scipy, scikit-learn, matplotlib, seaborn, plotly.
- Use print() to see values; keep output short (.head(), .shape, round()).
- Charts: build them with plotly and call save_fig(fig, "<name>"). Do not call fig.show().

ANSWER CONTRACT
- Before FINAL:, assign the result to the variable `answer` in a code block and print it, in exactly the requested format, using plain Python types (int, float, str, bool, list, dict).
- Then reply FINAL: with a short conclusion that quotes the numbers you computed.

ANALYSIS RULES
- Inspect the data first when needed (columns, dtypes, missing values, sample rows).
- Follow the task's definitions exactly (cleaning steps, filters, rounding, units).
- Correlation is not causation: never claim that one variable causes another from a correlation; name possible confounders."""

SHEET_RULES = {
    "first": "For Excel files, sheet=None returns the FIRST sheet; pass sheet=\"<name>\" for another one.",
    "largest": "For Excel files, sheet=None returns the largest sheet; pass sheet=\"<name>\" for another one.",
}

FORMAT_REMINDER = ("Your reply had no complete ```python code block and no line starting with FINAL:. "
                   "Reply with exactly ONE ```python code block, or with a line starting with FINAL:.")

# (pattern on the error line, hint). First match wins; checked in order.
ERROR_HINTS: list[tuple[str, str]] = [
    (r"^TimeoutError", "The code took too long (limit {timeout:g} s). Avoid Python loops over rows; "
                       "use vectorised pandas operations and work on aggregated data."),
    (r"^SandboxCrash|^MemoryError", "The process ran out of memory or crashed and all variables were lost. "
                                    "Reload the data and avoid building very large objects."),
    (r"^KeyError", "A column or key does not exist. Print df.columns.tolist() and use the exact name "
                   "(case, spaces, underscores)."),
    (r"^NameError", "A variable is not defined: a typo, or it was created in a step that failed or before "
                    "a sandbox restart. Re-create it in this code block."),
    (r"^(FileNotFoundError|DataLoadError)", "Use exactly the DATA FILE path given in the task with load_data()."),
    (r"^(ModuleNotFoundError|ImportError)", "Only pandas, numpy, scipy, scikit-learn, matplotlib, seaborn and "
                                            "plotly are installed and there is no internet."),
    (r"^(SyntaxError|IndentationError)", "Fix the syntax. Send one complete, correctly indented code block."),
    (r"^ParserError", "Wrong separator or decimal mark: read the file with load_data(path)."),
    (r"(could not convert|Unable to parse|invalid literal)", "A column contains text that is not a number or "
                                                             "date. Convert with pd.to_numeric(..., errors='coerce') "
                                                             "or pd.to_datetime(..., errors='coerce') and handle the NaN."),
    (r"^ValueError", "Check shapes, dtypes and argument values; print the intermediate objects before using them."),
    (r"^TypeError", "Check the dtypes (df.dtypes): object columns may mix numbers and text."),
    (r"^AttributeError", "Check the object type and the pandas 2.2 API (e.g. DataFrame.append was removed: "
                         "use pd.concat)."),
    (r"(Read-only file system|Permission denied)", "The data folder is read-only. Write files only with save_fig()."),
    (r"^ZeroDivisionError", "Guard against empty groups or zero denominators."),
]
DEFAULT_HINT = "Read the traceback, fix the root cause and keep the parts that already worked."


def system_prompt(sheet_policy: str) -> str:
    return SYSTEM_TEMPLATE.format(sheet_rule=SHEET_RULES[sheet_policy])


def task_prompt(question: str, data_path: str, answer_format: str, extra: str = "",
                profile_text: str | None = None) -> str:
    parts = [f"TASK: {question}",
             f"DATA FILE: {data_path}",
             f"REQUIRED ANSWER FORMAT: set the variable `answer` to {answer_format}."]
    if extra:
        parts.append(extra)
    if profile_text:
        parts.append("DATA PROFILE (computed automatically before you started; may be truncated):\n"
                     + profile_text)
    return "\n".join(parts)


def answer_status(found: bool, value) -> str:
    if not found:
        return "`answer` is not set yet."
    shown = repr(value)
    return f"`answer` is currently set to: {shown if len(shown) <= 300 else shown[:300] + ' ...'}"


_NEXT = ("Next: if the task is solved and `answer` holds the result in the required format, reply with a "
         "line starting with FINAL:. Otherwise send ONE ```python code block.")


def observation_prompt(observation: str, status: str) -> str:
    return f"OBSERVATION:\n{observation}\n\n{status}\n{_NEXT}"


def repeat_prompt(status: str) -> str:
    return ("You sent exactly the same code as in your previous step. It was NOT run again; its output is "
            f"shown above.\n{status}\n{_NEXT} Do not send the same code again.")


def error_hint(error: str | None, timeout_s: float) -> str:
    err = error or ""
    for pattern, hint in ERROR_HINTS:
        if re.search(pattern, err):
            return hint.format(timeout=timeout_s)
    return DEFAULT_HINT


def lesson(step: int, error: str | None, hint: str) -> str:
    err = (error or "unknown error").splitlines()[0][:160]
    return f"step {step}: {err} -> {hint}"


def fix_prompt(code: str, error: str | None, traceback_tail: str, attempt: int, max_retries: int,
               hint: str, lessons: list[str], state_lost: bool) -> str:
    code_shown = code if len(code) <= 1500 else code[:1500] + "\n# ... (truncated)"
    parts = [f"The code failed (fix attempt {attempt} of {max_retries}).",
             f"ERROR: {error}"]
    if traceback_tail:
        parts.append(f"TRACEBACK (last lines):\n{traceback_tail}")
    parts.append(f"FAILED CODE:\n```python\n{code_shown}\n```")
    if state_lost:
        parts.append("The sandbox was restarted: ALL variables are gone; reload the data first.")
    parts.append(f"HINT: {hint}")
    if lessons:
        parts.append("LESSONS FROM EARLIER ERRORS IN THIS TASK (do not repeat them):\n"
                     + "\n".join(f"- {x}" for x in lessons))
    parts.append("Reply with ONE corrected ```python code block. Fix the cause; do not repeat the same mistake.")
    return "\n".join(parts)


def history_summary(lines: list[str]) -> str:
    return ("EARLIER STEPS (summarised to save space; variables created by successful steps still exist):\n"
            + "\n".join(lines))
