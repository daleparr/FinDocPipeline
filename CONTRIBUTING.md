# Contributing to Financial ETL Pipeline

Thank you for your interest in contributing to the Financial ETL Pipeline! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Basic understanding of ETL processes and NLP

### Development Setup
1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/financial-etl-pipeline.git
   cd financial-etl-pipeline
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements_frontend.txt
   pip install -r requirements_dev.txt  # Development dependencies
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## 🛠️ Development Workflow

### Branch Strategy
- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/feature-name`: Individual feature development
- `bugfix/bug-description`: Bug fixes
- `hotfix/critical-fix`: Critical production fixes

### Making Changes
1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   python -m pytest tests/
   python test_etl.py
   python test_parsers.py
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

5. **Push and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## 📝 Coding Standards

### Python Style Guide
- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://black.readthedocs.io/) for code formatting
- Maximum line length: 88 characters
- Use type hints where appropriate

### Code Formatting
```bash
# Format code
black .

# Check formatting
black --check .

# Sort imports
isort .
```

### Linting
```bash
# Run linter
flake8 .

# Type checking
mypy src/
```

### Documentation
- Use docstrings for all functions and classes
- Follow [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) docstrings
- Update README.md for significant changes
- Add inline comments for complex logic

### Example Function Documentation
```python
def process_document(file_path: Path, institution: str) -> List[Dict[str, Any]]:
    """Process a financial document and extract structured data.
    
    Args:
        file_path: Path to the document file
        institution: Name of the financial institution
        
    Returns:
        List of dictionaries containing extracted sentence data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the institution name is invalid
        
    Example:
        >>> records = process_document(Path("earnings.pdf"), "JPMorgan")
        >>> len(records)
        150
    """
```

## 🧪 Testing

### Test Structure
```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── fixtures/       # Test data
└── conftest.py     # Pytest configuration
```

### Writing Tests
- Use `pytest` for testing
- Aim for >80% code coverage
- Test both success and failure cases
- Use fixtures for test data

### Example Test
```python
import pytest
from src.etl.parsers.pdf_parser import PDFParser

class TestPDFParser:
    def test_parse_valid_pdf(self, sample_pdf_path):
        """Test parsing a valid PDF file."""
        parser = PDFParser("TestBank", "Q1_2025")
        result = parser.parse(sample_pdf_path)
        
        assert len(result) > 0
        assert all("text" in record for record in result)
        
    def test_parse_invalid_file(self):
        """Test parsing an invalid file raises appropriate error."""
        parser = PDFParser("TestBank", "Q1_2025")
        
        with pytest.raises(FileNotFoundError):
            parser.parse(Path("nonexistent.pdf"))
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_parsers.py

# Run tests matching pattern
pytest -k "test_pdf"
```

## 📊 Performance Guidelines

### ETL Performance
- Process documents in batches when possible
- Use generators for large datasets
- Implement progress tracking for long operations
- Cache expensive computations

### Memory Management
- Use pandas efficiently (avoid copying large DataFrames)
- Clean up temporary files
- Stream large files instead of loading entirely into memory

### Example Optimization
```python
# Good: Generator for large datasets
def process_sentences(text: str) -> Iterator[Dict[str, Any]]:
    for sentence in segment_text(text):
        yield process_sentence(sentence)

# Avoid: Loading everything into memory
def process_sentences_bad(text: str) -> List[Dict[str, Any]]:
    return [process_sentence(s) for s in segment_text(text)]
```

## 🐛 Bug Reports

### Before Submitting
1. Check existing issues
2. Test with the latest version
3. Provide minimal reproduction case

### Bug Report Template
```markdown
**Bug Description**
A clear description of the bug.

**To Reproduce**
Steps to reproduce the behavior:
1. Upload file '...'
2. Set institution to '...'
3. Click process
4. See error

**Expected Behavior**
What you expected to happen.

**Environment**
- OS: [e.g., Windows 10, macOS 12.0]
- Python version: [e.g., 3.9.7]
- Package version: [e.g., 1.2.3]

**Additional Context**
Any other context about the problem.
```

## 💡 Feature Requests

### Feature Request Template
```markdown
**Feature Description**
A clear description of the feature you'd like to see.

**Use Case**
Describe the problem this feature would solve.

**Proposed Solution**
How you envision this feature working.

**Alternatives Considered**
Other solutions you've considered.

**Additional Context**
Any other context or screenshots.
```

## 📋 Pull Request Process

### PR Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
- [ ] CI/CD pipeline passes

### PR Template
```markdown
**Description**
Brief description of changes.

**Type of Change**
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

**Testing**
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

**Checklist**
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
```

## 🏗️ Architecture Guidelines

### Module Structure
- Keep modules focused and cohesive
- Use dependency injection
- Implement proper error handling
- Follow SOLID principles

### Adding New Parsers
1. Inherit from `BaseParser`
2. Implement required methods
3. Add comprehensive tests
4. Update configuration
5. Document usage

### Example Parser Structure
```python
from src.etl.parsers.base_parser import BaseParser

class NewFormatParser(BaseParser):
    """Parser for new document format."""
    
    def __init__(self, bank_name: str, quarter: str):
        super().__init__(bank_name, quarter)
        
    def parse(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse the document and return structured data."""
        # Implementation here
        pass
        
    def _validate_file(self, file_path: Path) -> bool:
        """Validate file format."""
        # Implementation here
        pass
```

## 🤝 Community

### Communication
- GitHub Issues: Bug reports and feature requests
- GitHub Discussions: General questions and ideas
- Pull Requests: Code contributions

### Code of Conduct
- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow the project's code of conduct

## 📚 Resources

### Documentation
- [Project Wiki](https://github.com/yourusername/financial-etl-pipeline/wiki)
- [API Documentation](https://financial-etl-pipeline.readthedocs.io/)
- [Examples](https://github.com/yourusername/financial-etl-pipeline/tree/main/examples)

### Learning Resources
- [ETL Best Practices](https://example.com/etl-best-practices)
- [NLP with Python](https://example.com/nlp-python)
- [Financial Data Processing](https://example.com/financial-data)

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to the Financial ETL Pipeline! 🚀