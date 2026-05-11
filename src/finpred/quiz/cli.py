"""`finpred-quiz` — pose the N-way pick-the-real quiz to a human and/or the trained model.

STUB (Phase A.1 for the CLI skeleton; wired to a real checkpoint in Phase B). Behaviour:
  - load config (+ optional --checkpoint),
  - build one episode: pick a real window from the held-out era (default ticker NVDA), generate
    N-1 fakes from G (length-matched, start-price-aligned),
  - render all N to `quiz.out_dir` with identical axes/date formatting, shuffled, labelled 1..N,
  - prompt the human for a guess; if a checkpoint is given, also report the model's pick and
    each candidate's discriminator score; reveal the answer.

Uses typer; `app` is the entry point referenced in pyproject.toml.
"""

from __future__ import annotations

import typer

app = typer.Typer(add_completion=False, help="N-way pick-the-real stock-chart quiz.")


@app.command()
def main(
    config: str = typer.Option(..., "--config"),
    checkpoint: str | None = typer.Option(None, "--checkpoint"),
    ticker: str = typer.Option("NVDA", "--ticker"),
) -> None:
    """Run one N-way quiz episode. STUB."""
    raise NotImplementedError("quiz.cli.main — implement in Phase A.1 / Phase B")


if __name__ == "__main__":
    app()
