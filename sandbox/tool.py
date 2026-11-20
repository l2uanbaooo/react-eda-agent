"""LangChain `python_repl` tool, hand-wrapped with @tool around OUR sandbox.

No ready-made LangChain REPL tool is used: langchain_experimental's
PythonREPLTool executes code on the host. This function sends the code to the
Docker (or subprocess) sandbox and returns the observation text.
The ReAct loop (agent.py, GD4) calls it as python_repl.invoke({"code": code});
LangChain does not drive the agent.
"""
from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from .base import ExecResult, Sandbox, format_observation


def make_python_repl_tool(sandbox: Sandbox, max_chars: int = 2000,
                          on_result=None) -> BaseTool:
    """Build the tool bound to one sandbox session.

    on_result(ExecResult) is called after every run so the caller can log the
    structured result (error type, timeout, state_lost) next to the text.
    """

    @tool("python_repl")
    def python_repl(code: str) -> str:
        """Run Python code in a persistent, isolated sandbox session and return its output.

        Variables persist between calls. Pre-loaded: pd, np, px, go, load_data(path, sheet=None),
        DATA_DIR (read-only data), OUTPUT_DIR, save_fig(fig, name). Assign the final result
        to the variable `answer`. Use print() to see values.
        """
        res: ExecResult = sandbox.run(code)
        if on_result is not None:
            on_result(res)
        return format_observation(res, max_chars)

    return python_repl
