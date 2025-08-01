"""Tests for CLI functionality."""

from typer.testing import CliRunner
from acousto_gen.cli import app

runner = CliRunner()


def test_version_command():
    """Test version command output."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "Acousto-Gen version" in result.stdout


def test_simulate_command_basic():
    """Test basic simulate command."""
    result = runner.invoke(app, ["simulate"])
    assert result.exit_code == 0
    assert "Running simulation at 40000" in result.stdout


def test_simulate_command_with_frequency():
    """Test simulate command with custom frequency."""
    result = runner.invoke(app, ["simulate", "--frequency", "50000"])
    assert result.exit_code == 0
    assert "Running simulation at 50000" in result.stdout


def test_simulate_command_with_output():
    """Test simulate command with output file."""
    result = runner.invoke(app, ["simulate", "--output", "test.h5"])
    assert result.exit_code == 0
    assert "Output will be saved to: test.h5" in result.stdout