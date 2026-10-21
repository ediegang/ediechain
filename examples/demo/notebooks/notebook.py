"""Run the experiments under `configs/` from a notebook.

    uv run marimo edit examples/demo/notebooks/notebook.py

Nothing here defines a task or a chain. The notebook picks an experiment, picks a store, and
looks at what came back — which is the whole point of keeping experiments in `configs/`.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", layout_file="layouts/notebook.slides.json")


@app.cell
def _():
    import json
    import marimo as mo
    import polars as pl
    from ediechain import Runner, render

    from demo.config import PIPELINE_DIR

    return PIPELINE_DIR, Runner, json, mo, pl, render


@app.cell
def _(mo):
    mo.md("""
    # Experiments

    Every config under `configs/` exposes one `experiment`. None of them binds a runner, so
    the store is chosen here.
    """)
    return


@app.cell
def _():
    from configs.experiment_v0 import experiment as v0
    from configs.nested.experiment_v1 import experiment as v1

    return (v0,)


@app.cell
def _(PIPELINE_DIR, Runner, v0):
    pipeline = v0(seed=0).on(Runner(root=PIPELINE_DIR))
    # pipeline = v1(seed=0).on(Runner(root=PIPELINE_DIR))
    return (pipeline,)


@app.cell
def _(pipeline, render):
    print(render(pipeline))
    return


@app.cell
def _(pipeline):
    pipeline["features d0043e47"].arguments["data"]
    return


@app.cell
def _(pipeline):
    pipeline.store.artifact_path(pipeline["model"])
    return


@app.cell
def _(pipeline):
    print(pipeline)
    return


@app.cell(hide_code=True)
def _(pipeline, render):
    print(render(pipeline))
    return


@app.cell
def _(pipeline, render):
    forced = pipeline.force("raw_data")
    print(render(forced))
    return


@app.cell
def _(pipeline):
    pipeline.value("accuracy 5751bb34")
    return


@app.cell
def _(mo, pipeline, render):
    mo.mermaid(f"{render(pipeline, view="mermaid")}")
    return


@app.cell
def _(mo, pipeline, pl):
    outputs = pipeline.outputs
    named = outputs._asdict() if hasattr(outputs, "_asdict") else {"result": outputs}
    results = pl.DataFrame(
        {
            "terminal": list(named),
            "key": [node.key[:8] for node in named.values()],
            "value": [pipeline.value(node) for node in named.values()],
        }
    )
    mo.ui.table(results, selection=None)
    return (results,)


@app.cell
def _(mo):
    mo.md("""
    ## What landed in the store

    One directory per node, named by its key, holding the artifact and its `run_info.json`.
    Change the seed above and only the branches that depend on it are recomputed.
    """)
    return


@app.cell
def _(PIPELINE_DIR, mo, pl, results):
    stored = pl.DataFrame(
        {
            "task": [node.parent.name for node in sorted(PIPELINE_DIR.glob("*/*")) if node.is_dir()],
            "key": [node.name[:8] for node in sorted(PIPELINE_DIR.glob("*/*")) if node.is_dir()],
            "files": [
                ", ".join(sorted(child.name for child in node.iterdir()))
                for node in sorted(PIPELINE_DIR.glob("*/*"))
                if node.is_dir()
            ],
        }
    )
    mo.vstack([mo.md(f"`{PIPELINE_DIR}` holds {stored.height} nodes for {results.height} terminals."), stored])
    return


@app.cell
def _(json, mo, pipeline):
    node = pipeline.terminals[0]
    info = pipeline.runner.store.load_sidecar(node)
    mo.md(f"**Sidecar of `{node.task.name}`**\n\n```json\n{json.dumps(info, indent=4)}\n```")
    return


if __name__ == "__main__":
    app.run()
