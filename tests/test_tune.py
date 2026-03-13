#!/usr/bin/env python

"""Tests for the chess-tuning-tools CLI module."""

from click.testing import CliRunner

from tune import cli


class TestCli:
    """Tests for the CLI interface."""

    def test_without_subcommand_returns_error(self):
        """CLI should return error code 2 when invoked without subcommand."""
        runner = CliRunner()
        result = runner.invoke(cli.cli)
        assert result.exit_code == 2

    def test_help_displayed_correctly(self):
        """CLI --help flag should display help text correctly."""
        runner = CliRunner()
        result = runner.invoke(cli.cli, ["--help"])
        assert result.exit_code == 0
        assert "--help  Show this message and exit." in result.output
