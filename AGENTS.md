# Repository Guidelines

This document provides comprehensive guidelines for AI coding assistants working with this chess-tuning-tools repository.

## 1. Project Overview

The **chess-tuning-tools** repository is designed for research and tuning of chess engines. It includes:
- A CLI tool for running tuning experiments
- Database workers for storing results
- Algorithmic helpers for analysis
- Comprehensive test suite using pytest

## 2. Development Environment Setup

### Prerequisites
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager (recommended)
- Git

### Installation
```bash
# Clone the repository
git clone https://github.com/anomalyco/chess-tuning-tools.git
cd chess-tuning-tools

# Install dependencies in development mode
uv sync --group dev --extra dist

# Alternative: Use pip with virtual environment
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,dist]"
```

The `--extra dist` flag pulls optional packages needed for distributed execution (joblib, pandas, postgresql, sqlalchemy).

## 3. Running Tests

### Test Commands
| Command | Purpose |
|---------|---------|
| `uv run --group dev pytest` | Run entire test suite |
| `uv run --group dev pytest tests/test_tune.py` | Run specific test module |
| `uv run --group dev pytest tests/test_tune.py::test_my_feature` | Run single test function |
| `uv run --group dev nox -s tests` | CI-aligned test run |
| `uv run --group dev pytest -q` | Quiet output |
| `uv run --group dev pytest -s` | Show print statements |

### Best Practices
- Keep tests deterministic; avoid external network calls
- Use fixtures for reusable setup
- Mark parameterized tests with `@pytest.mark.parametrize`
- Place test files in `tests/` mirroring the `tune/` structure

## 4. Linting, Formatting, and Type Checking

### Code Quality Tools
| Command | Purpose |
|---------|---------|
| `uv run --group dev nox -s pre-commit` | Full ruff format & check |
| `uv run --group dev ruff format` | Format code only |
| `uv run --group dev ruff check` | Check style/lint errors |
| `uv run --group dev mypy .` | Static type checking |

**Note:** All tools obey configuration in `pyproject.toml`:
- Max line length: 80 characters
- Ignored rules: E203, E501, F403, F405
- `mypy.ini` ignores imports for third-party libraries

### Makefile Shortcuts
```bash
make lint       # Runs ruff format & check
make test-all   # Runs full test suite
make coverage   # Generates coverage report
make clean      # Cleans build artifacts
```

## 5. Coding Style Guidelines

### Imports
- **Order**: Standard library → Third-party → Local
- **Grouping**: Separate each group by blank lines
- **Absolute imports**: Always use absolute imports within package
- **Avoid wildcards**: Never use `from x import *`

Example:
```python
import os
import sys
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

from tune.utils import logger
from tune.config import settings
```

### Naming Conventions
- **Functions/Variables**: snake_case (e.g., `calculate_score`, `data_frame`)
- **Classes/Data Structures**: PascalCase (e.g., `TunerConfig`, `ResultSet`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_ITERATIONS`, `DEFAULT_TIMEOUT`)
- **Private members**: Prefix with underscore (e.g., `_internal_helper()`)

### Type Hints
- Provide annotations for public functions and classes
- Use `typing` primitives: `List`, `Dict`, `Tuple`, `Optional`
- Use `Any` sparingly; prefer specific types
- For complex signatures, break across multiple lines

Good example:
```python
def process_results(
    data: List[Dict[str, Any]],
    config: Optional[TunerConfig] = None,
) -> Tuple[int, float]:
    ...
```

### Docstrings
- Use Google or reST style
- Include brief description, parameters, return values
- No empty docstrings

Example:
```python
def evaluate_position(board_state: BoardState) -> EvaluationScore:
    """
    Evaluates the chess position and returns a score.
    
    Args:
        board_state: Current state of the chessboard.
    
    Returns:
        Score representing position evaluation.
    """
    ...
```

### Error Handling
- Raise specific exceptions derived from `Exception`
- Avoid `sys.exit` in library code
- Use custom exception classes for domain-specific errors

Example:
```python
class TunerError(Exception):
    pass

if invalid_config:
    raise TunerError("Invalid configuration provided")
```

### Logging
- Use `logger = logging.getLogger(__name__)`
- Prefer structured logging via `extra` parameter
- Log levels: DEBUG → INFO → WARNING → ERROR → CRITICAL

Example:
```python
logger.info("Processing started", extra={"stage": "initialization"})
logger.error("Failed to load data", exc_info=True)
```

## 6. Git Workflow

### Commit Messages
- Imperative mood, short form
- Reference issues when applicable (e.g., "Fixes #123")
- Focus on "why" rather than "what"

Examples:
- "Add support for parallel tuning experiments"
- "Fix race condition in database worker"
- "Update documentation for CLI usage"

### Best Practices
- Run pre-commit hooks before pushing:
  ```bash
  uv run --group dev nox -s pre-commit
  ```
- Never commit `.venv` or large binary files
- Use feature branches with descriptive names
- Keep commits focused and atomic

## 7. Pull Request Policy

### Requirements
- Reference related issue in PR description
- Include CI status badges
- Provide summary of changes
- List all modified components

### Template
```markdown
# Summary

- Added new feature X
- Fixed bug Y
- Updated documentation Z

# Changes

- Modified `tune/core/tuner.py` to add parallel execution
- Added tests in `tests/test_tuner_parallel.py`
- Updated README with new examples

# Testing

All tests passing:
- Unit tests: ✓
- Integration tests: ✓
- Linting: ✓
- Type checking: ✓
```

## 8. Project Structure

```
tree
├── tune/                  # Main package
│   ├── cli/               # Command-line interface
│   ├── core/              # Core algorithms
│   ├── db/                # Database workers
│   ├── utils/             # Utilities
│   └── config.py          # Configuration
├── tests/                 # Test suite (mirrors tune/)
├── docs/                  # Documentation
├── examples/              # Usage examples
├── .github/workflows/     # CI configuration
├── Makefile               # Command shortcuts
├── pyproject.toml         # Build & dependencies
├── noxfile.py             # Reproducible environments
└── .pre-commit-config.yaml # Pre-commit hooks
```

## 9. Common Tasks

### Adding a New Feature
1. Create branch from main
2. Add implementation in appropriate module
3. Write comprehensive tests
4. Update documentation
5. Run full test suite
6. Submit pull request

### Fixing a Bug
1. Identify failing test or reproduce issue
2. Implement fix with minimal changes
3. Add regression test if needed
4. Verify all existing tests pass
5. Submit pull request

### Updating Dependencies
1. Modify `pyproject.toml`
2. Run `uv lock`
3. Test compatibility
4. Document breaking changes if any

## 10. Additional Resources

- [Python Style Guide](https://peps.python.org/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [mypy Documentation](https://mypy.readthedocs.io/)
