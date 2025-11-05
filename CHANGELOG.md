# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-11-05

### Added
- Modern Python project structure with `pyproject.toml`
- Type hints for key functions and modules
- Pre-commit hooks configuration (`.pre-commit-config.yaml`)
- Code formatting with Black and Ruff
- Development dependencies for linting and testing
- Comprehensive `.gitignore` file
- `CHANGELOG.md` for tracking changes
- `setup.py` for backward compatibility

### Changed
- **BREAKING**: Updated Python requirement from 3.6-3.7 to 3.8+
- **BREAKING**: Updated PyTorch requirement from 1.0-1.1 to 2.0+
- **BREAKING**: Updated NetworkX from 1.9 to 3.0+
- **BREAKING**: Replaced deprecated `nx.write_yaml()`/`nx.read_yaml()` with `nx.write_graphml()`/`nx.read_graphml()`
  - Graph files now use `.graphml` format instead of `.yaml`
  - Backward compatibility maintained - will try to load `.graphml` files when `.yaml` is specified
- Updated all dependencies to modern versions with proper version constraints
- Fixed hard-coded Windows paths (`C:/dataset/`) to be cross-platform compatible
  - Now uses `~/dataset/` with fallback to `./dataset/`
- Improved README with modern installation instructions
- Enhanced code quality with type hints in core modules

### Fixed
- Cross-platform compatibility issues with file paths
- Import organization and code formatting
- Deprecated NetworkX API usage

### Migration Guide

If you're upgrading from the old version:

1. **Python Version**: Ensure you're using Python 3.8 or higher
2. **Dependencies**: Reinstall dependencies with `pip install -r requirements.txt`
3. **Graph Files**: The first run will regenerate graph files in `.graphml` format. Old `.yaml` graph files can be deleted.
4. **Dataset Path**: Update any custom dataset paths if you were using the hardcoded `C:/dataset/`

For development:
```bash
pip install -e ".[dev]"
pre-commit install
```
