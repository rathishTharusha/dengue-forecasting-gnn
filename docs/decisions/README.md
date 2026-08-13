# Architecture Decision Records

Short, numbered notes recording *why* a non-obvious modelling or engineering choice was made.
A reviewer asking "why residual-over-persistence?" or "why not `torch-geometric-temporal`?"
should find the answer here, not in someone's memory.

Write one when a decision:

- shapes results the report depends on,
- rejected a plausible alternative, or
- will look wrong to someone who wasn't in the room.

Copy [`TEMPLATE.md`](TEMPLATE.md), number it sequentially, and link it from the PR.
ADRs are immutable once merged — if a decision is reversed, write a new ADR that supersedes
the old one and add a `Superseded by 000N` line to it.

## Index

| # | Decision | Status |
|---|---|---|
| [0001](0001-baseline-training-refinements.md) | Residual-over-persistence, log1p target, and rolling-origin CV | Accepted |
| [0002](0002-plain-pytorch-geometric.md) | Plain PyTorch Geometric over `torch-geometric-temporal` | Accepted |
