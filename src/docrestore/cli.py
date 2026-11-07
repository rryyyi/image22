from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from .config import load_config
from .io import imread, write_json
from .pipeline import DocumentPipeline
from .quality.assess import assess_quality, derive_shadow_mask, reference_metrics

app = typer.Typer(help="Restore photographed or scanned paper documents locally.")
console = Console()


@app.command()
def run(input: Path = typer.Argument(..., exists=True, dir_okay=False),
        output: Path = typer.Option(..., "--output", "-o"),
        config: Path | None = typer.Option(None, exists=True, dir_okay=False),
        intermediates: bool = typer.Option(False, "--intermediates")) -> None:
    pipeline = DocumentPipeline(load_config(config))
    result = pipeline.process_path(input, output, intermediates)
    console.print(f"Processed {len(result.pages)} page(s) in {result.elapsed_seconds:.2f}s -> {output}")
    for page in result.pages:
        console.print(f"page {page.index+1}: {page.before.readability_score:.1f} -> {page.after.readability_score:.1f}; warnings={len(page.warnings)}")


@app.command()
def assess(input: Path = typer.Argument(..., exists=True, dir_okay=False),
           target: Path | None = typer.Option(None, exists=True, dir_okay=False),
           output: Path | None = typer.Option(None, "--output", "-o")) -> None:
    image = imread(input)
    report = {"quality": assess_quality(image).to_dict()}
    if target is not None:
        clean = imread(target)
        shadow = derive_shadow_mask(image, clean)
        report["reference"] = reference_metrics(image, clean, {"full": __import__('numpy').ones(shadow.shape, bool),
                                                                  "shadow": shadow, "non_shadow": ~shadow})
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if output: write_json(output, report)
    console.print(text)


if __name__ == "__main__":
    app()
