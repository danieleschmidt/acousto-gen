"""Command-line interface for Acousto-Gen."""

from typing import Optional

import typer

app = typer.Typer(
    name="acousto-gen",
    help="Generative acoustic holography toolkit"
)


@app.command()
def version() -> None:
    """Show version information."""
    from acousto_gen import __version__
    typer.echo(f"Acousto-Gen version {__version__}")


@app.command()
def simulate(
    frequency: float = typer.Option(40000, "--frequency", "-f", help="Frequency in Hz"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path")
) -> None:
    """Run acoustic field simulation."""
    typer.echo(f"Running simulation at {frequency} Hz")
    if output:
        typer.echo(f"Output will be saved to: {output}")


def main() -> None:
    """Entry point for CLI."""
    app()