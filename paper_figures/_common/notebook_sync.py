"""Keep each figure's script (make_figure.py) and notebook (<figure>.ipynb) in sync.

Both use the same cells. In the script, a line "# %%" starts a code cell (text
after it becomes the cell's first comment line) and "# %% [markdown]" starts a
markdown cell whose lines are written as "# text".

    python notebook_sync.py to-notebook ../fig_clean_h0/make_figure.py
    python notebook_sync.py to-script   ../fig_clean_h0/fig_clean_h0.ipynb
    python notebook_sync.py run         ../fig_clean_h0/fig_clean_h0.ipynb

"to-notebook" writes <folder>/<folder>.ipynb next to the script, "to-script"
writes make_figure.py next to the notebook, and "run" executes the notebook in
its own folder (outputs go to output/ exactly as with the script).
"""

import argparse
import re
from pathlib import Path

import nbformat

MARKER = re.compile(r"^# %%(?P<md> \[markdown\])?(?P<title>.*)$")
KERNEL = {"display_name": "Python 3 (tn_tda)", "language": "python", "name": "python3"}
EDIT_NOTE = (
    "**How to edit this figure.** Run all cells. The figure is saved to `output/` "
    "(PNG and PDF) and shown below the last plotting cell. Style (fonts, sizes, line "
    "widths) comes only from `../_common/paper.mplstyle`, and colors per system size "
    "from `../_common/paper_style.py`, so edit those files to change every figure at once. "
    "Data are read from `data_raw/` and checked against their SHA-256 manifest, never modified. "
    "After editing, run `python ../_common/notebook_sync.py to-script <this notebook>` to "
    "update `make_figure.py`, and re-sync the manuscript with "
    "`python paper_manuscript/tools/sync_figures.py` from the project root."
)


def script_to_cells(text):
    cells, kind, lines = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(lines).strip("\n")
        if kind == "markdown":
            body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(body))
        elif body.strip():
            cells.append(nbformat.v4.new_code_cell(body))

    for line in text.splitlines():
        m = MARKER.match(line)
        if m:
            flush()
            kind = "markdown" if m.group("md") else "code"
            title = m.group("title").strip()
            lines = [f"# {title}"] if (kind == "code" and title) else []
        else:
            if kind is None:
                kind = "code"
            lines.append(line)
    flush()
    return cells


def to_notebook(script):
    script = Path(script).resolve()
    folder = script.parent
    cells = script_to_cells(script.read_text(encoding="utf-8"))
    nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": KERNEL})
    insert_at = 1 if cells and cells[0].cell_type == "markdown" else 0
    nb.cells.insert(insert_at, nbformat.v4.new_markdown_cell(EDIT_NOTE))
    target = folder / f"{folder.name}.ipynb"
    nbformat.write(nb, target)
    print(f"wrote {target}")


def to_script(notebook):
    notebook = Path(notebook).resolve()
    nb = nbformat.read(notebook, as_version=4)
    parts = []
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            if cell.source.startswith("**How to edit this figure.**"):
                continue
            parts.append("# %% [markdown]\n" + "\n".join("# " + l if l else "#" for l in cell.source.splitlines()))
        elif cell.cell_type == "code":
            src = cell.source.splitlines()
            if src and src[0].startswith("# ") and not src[0].startswith("# %%"):
                parts.append(f"# %% {src[0][2:]}\n" + "\n".join(src[1:]))
            else:
                parts.append("# %%\n" + cell.source)
    target = notebook.parent / "make_figure.py"
    target.write_text("\n\n".join(p.rstrip("\n") for p in parts) + "\n", encoding="utf-8")
    print(f"wrote {target}")


def run(notebook):
    from nbclient import NotebookClient
    notebook = Path(notebook).resolve()
    nb = nbformat.read(notebook, as_version=4)
    NotebookClient(nb, timeout=1800, kernel_name="python3",
                   resources={"metadata": {"path": str(notebook.parent)}}).execute()
    nbformat.write(nb, notebook)
    print(f"executed {notebook}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("to-notebook", "to-script", "run"))
    parser.add_argument("path")
    args = parser.parse_args()
    {"to-notebook": to_notebook, "to-script": to_script, "run": run}[args.command](args.path)


if __name__ == "__main__":
    main()
