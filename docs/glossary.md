# Glossary

Every term used in `src/` is defined here. A term that is not here must be agreed before it is
written down.

## The graph

**task** — a Python function decorated with `@task`. It declares its inputs as parameters and its
output as a return annotation. It is the unit that gets computed and stored.

**node** — one task plus one complete set of arguments. Built by calling the task; it describes a
computation without running it. Two nodes with equal arguments are the same node.

**key** — the hash that identifies a node. Computed from the task name, its version and its
arguments, with any argument that is itself a node replaced by that node's key. Recursive, so a
change anywhere upstream reaches every key downstream.

**input** — a node passed as an argument to another node. `node.inputs` is the flat, deduplicated
list of them.

**terminal node** — a node nobody else consumes. What you hand to `chain()`.

**chain** — `Chain`, and the `@chain` decorator that builds one. The decorated function composes
tasks and returns nodes; the chain walks back over their inputs and holds every node needed, in
dependency order, deduplicated by key. Building it runs nothing.

**outputs** — exactly what the composing function returned: one node, or a `NamedTuple` of them,
so each terminal keeps its name.

**version** — a number on a task, bumped by hand when its body changes meaning. It is part of the
key, so bumping it invalidates that task and everything downstream.

**experiment** — a Python module that composes tasks into terminal nodes. It is the definition of
a run, and it is type checked like any other code.

## Per-parameter marks

**transparent** — a parameter that stays out of the key while it holds its default. Adding one to
an existing task does not invalidate existing data; passing a non-default value does. Marked on a
task with `@task(transparent=...)`, on a class with `by_init_params(transparent=...)`.

**ignore** — a parameter that never reaches the key at all. For things that cannot change the
result: a logger, a client, a verbosity flag. Marked on a task with `@task(ignore=...)`, on a
class with `by_init_params(exclude=...)`.

## Keys

**encoder** — `encode()`. Turns any value into a string that identifies it. Two values that mean
different things never produce the same string, and a type it does not know raises.

**tag** — the name a type carries inside an encoding. A class name by default, never an import
path, so moving a file does not orphan the data. `register(cls, name=...)` overrides it.

**converter** — a function registered for a type, turning a value of that type into something the
encoder already knows. Registered with `register`.

## Values at two moments

**In[T]** — the annotation for a task parameter. At the call site it is a `Node[T]`, because the
artifact does not exist yet. Inside the body it is a `T`, because the executor substituted it.

**resolve** — narrows an `In[T]` back to `T` for the body. At run time it returns the value and
raises if it is still a node, which means the executor did not substitute.

## Storage

**artifact** — the file a task's result is stored in, at `{root}/{task}/{key}/data.{ext}`.

**node directory** — `{root}/{task}/{key}`, holding the artifact and its sidecar. It is the unit
the store commits: it is renamed into place in one step, so it is never seen half written, and a
second writer renaming onto it is refused.

**sidecar** — `run_info.json` beside the artifact. What ran, when, by whom, from which inputs. It
is provenance, never key material: nothing reads it to decide whether to recompute.

**store** — where artifacts live, and the object that puts them there and reads them back. Not a
cache: it is the only copy, and some artifacts cannot be recomputed.

**handler** — `TypeHandler`. Knows how to write one type of value to a path and read it back, and
nothing else. It is handed a staging path and never the artifact's real one, so the store owns the
commit and no handler can skip it.

**typed handler** — `TypedHandler`. A handler for a format that cannot describe itself, so it is
also handed the task's return annotation. It opts in with `typed = True`. Parquet carries its
column types and pickle carries the class; JSON has four scalars and string-only keys, so only
JSON needs this.

**commit** — the one step that makes a node visible: the staging directory is renamed onto the
node directory. Atomic, so a reader sees the node whole or not at all, and a second writer is
refused rather than merged.

**cache hit** — the executor finding an artifact already in the store and loading it instead of
computing. The store is not a cache; a hit against it is still a cache hit.

> taskchain uses **cache** for something narrower — `@cached` memoizes a method's return value
> (`cache.py`, `docs/utils.md`). Its artifact persistence is `data.py` and `base_dir`. Say
> **memoize** for the taskchain meaning.

## Running

**runner** — `Runner`. Holds a store and asks a node for its value. It *pulls*: a node already in
the store is loaded and nothing above it is touched.

**force** — `chain.force(target)`. Marks a node to be recomputed although its artifact exists. The
key does not change, only the content, so forcing always cascades to every node below. A runner
records what it has already redone, so a forced chain does the work once per runner, not once per
request.

**sweep** — `Store.clean()`. Removes staging directories a killed process left behind. It is not a
repair: a half-committed node cannot exist.

## Rendering

**render** — `render(chain)`. Draws the chain as text. It asks the store what already exists, so
it answers "what will run" before anything runs. It computes nothing.

**view** — which shape `render` draws. `tree` is the default, one branch per terminal; `flat` is
one line per node in plan order; `mermaid` is a graph a notebook can display.

**mark** — the symbol in front of a node in a rendering. It is the node's state now, read from the
store, not a record of what happened. A chain with no runner has no store to ask, so it is drawn
without marks.

| mark | name      | meaning                                                              |
|------|-----------|----------------------------------------------------------------------|
| `✓`  | stored    | the artifact is in the store; the runner loads it                    |
| `·`  | pending   | nothing in the store; the runner computes it                         |
| `○`  | memory    | `store=IN_MEMORY`; never written, so rebuilt in every process        |
| `!`  | forced    | marked by `chain.force()`; recomputed although the artifact is there |
| `↑`  | reference | drawn in full above; one shared node, not a second copy              |

A forced node is drawn `!` whether or not its artifact exists, because the mark says what the
runner will do, and the cascade below it is marked the same way.

**short key** — the first 8 characters of a node's key, which is what a rendering prints. It names
a node for a human. `chain["✓ features d0043e47"]` takes a whole pasted line: a token no node
answers is dropped, and the rest narrow together.

## Distribution

**rank** — one process of a job that runs the same chain as the others.

**coordination** — `Coordination`. Decides, per rank, whether this rank writes (`should_write`),
where the ranks meet (`barrier`), and whether they all saw the same thing (`agree`). Only the
write is restricted, never the body.

**collective** — `@task(collective=True)`. One computation spread across every rank, so every rank
must enter the body. The default is the opposite: only the writing rank runs it and the others
wait and load, which is why eight ranks make one API call rather than eight.
