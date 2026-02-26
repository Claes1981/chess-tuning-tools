#!/usr/bin/env python

"""Tests for the chess-tuning-tools CLI module."""

from click.testing import CliRunner

from tune import cli


def test_cli_without_subcommand_returns_error():
    """CLI should return error code 2 when invoked without subcommand."""
    runner = CliRunner()
    result = runner.invoke(cli.cli)
    assert result.exit_code == 2


def test_cli_help_displayed_correctly():
    """CLI --help flag should display help text correctly."""
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["--help"])
    assert result.exit_code == 0
    assert "--help  Show this message and exit." in result.output
