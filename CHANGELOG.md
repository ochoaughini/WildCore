# Changelog

All notable changes to this project will be documented in this file. This project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2025-06-24
### Added
- Initial public release of WildCore.
- Core package `wildcore` with simulation agent (`GutoVectorWildcard`) and self-regulated anomaly detector (`AutoRegulatedPromptDetector`).
- Comprehensive `README.md` with installation guide, project structure and usage example.
- Automated CI workflow (`.github/workflows/ci.yml`) covering multi-version testing and linting.
- Development tooling configuration via `pyproject.toml` (Black, Isort, Ruff, MyPy, Flake8).
- Documentation in `docs/` covering agent, detector, utilities and project index.
- Unit tests for detector in `tests/`.

### Changed
- Adopted modern `src/` layout for Python packaging.
- Added optional dependency group `full` for heavy ML packages (PyTorch, Transformers, etc.) to keep base install lightweight.

### Fixed
- Removed emojis and informal language for a professional presentation.

[0.1.0]: https://github.com/ochoaughini/WildCore/releases/tag/v0.1.0
