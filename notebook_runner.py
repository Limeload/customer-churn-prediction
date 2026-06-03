"""Isolated Jupyter notebook execution with structured output collection."""
import re
import tempfile

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[mK]')


def _clean(text: str) -> str:
    return _ANSI_RE.sub('', str(text))


def _collect_outputs(nb) -> list[dict]:
    outputs = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            ot = out.get("output_type", "")
            if ot == "stream":
                txt = _clean(out.get("text", ""))
                if txt.strip():
                    outputs.append({"kind": "stream", "name": out.get("name", "stdout"), "text": txt})
            elif ot in ("display_data", "execute_result"):
                d = out.get("data", {})
                if "image/png" in d:
                    outputs.append({"kind": "image", "b64": d["image/png"]})
                elif "text/html" in d:
                    outputs.append({"kind": "html", "html": d["text/html"]})
                elif "text/plain" in d:
                    txt = _clean(d["text/plain"])
                    if txt.strip():
                        outputs.append({"kind": "text", "text": txt})
            elif ot == "error":
                tb = _clean("\n".join(out.get("traceback",
                    [f"{out.get('ename', 'Error')}: {out.get('evalue', '')}"])))
                outputs.append({"kind": "error", "text": tb})
    return outputs


def run_notebook(content: bytes, timeout: int = 600) -> dict:
    """Execute a Jupyter notebook from raw bytes in an isolated temp directory.

    Returns {"success": bool, "error": str | None, "outputs": list[dict]}.
    """
    nb = nbformat.reads(content.decode("utf-8"), as_version=4)
    ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
    success, exec_error = True, None

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            ep.preprocess(nb, {"metadata": {"path": tmpdir}})
        except Exception as exc:
            success = False
            exec_error = _clean(str(exc))

    return {"success": success, "error": exec_error, "outputs": _collect_outputs(nb)}
