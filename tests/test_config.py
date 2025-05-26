"""Tests for configuration loading and validation."""
import os
import pytest
import tempfile
import yaml
from pathlib import Path
from etl.config import ConfigManager, ETLConfig, ConfigError

# Sample valid configuration
SAMPLE_CONFIG = """
banks:
  - name: "Citigroup"
    quarters: ["Q1_2025", "Q4_2024"]
    data_sources:
      - type: "presentation"
        format: "pdf"
        file_pattern: "presentation.pdf"
      - type: "transcript"
        format: "pdf"
        file_pattern: "transcript.pdf"

processing:
  text_cleaning:
    remove_stopwords: true
    min_word_length: 3
    max_word_length: 50
  topic_modeling:
    num_topics: 10
    min_topic_size: 5
    n_gram_range: [1, 2]
    embedding_model: "all-MiniLM-L6-v2"
  sentiment_analysis:
    model_name: "ProsusAI/finbert"
    batch_size: 32
"""

@pytest.fixture
def temp_config():
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(SAMPLE_CONFIG)
    yield Path(f.name)
    # Cleanup
    if os.path.exists(f.name):
        os.unlink(f.name)

def test_config_loading(temp_config):
    """Test that configuration loads correctly from a YAML file."""
    # When
    config_manager = ConfigManager(config_path=temp_config)
    config = config_manager.get_config()
    
    # Then
    assert len(config.banks) == 1
    assert config.banks[0].name == "Citigroup"
    assert len(config.banks[0].data_sources) == 2
    assert config.banks[0].data_sources[0].type == "presentation"
    assert config.banks[0].data_sources[0].format == "pdf"

def test_missing_banks():
    """Test that configuration fails with missing banks section."""
    # Given
    config_yaml = """
    processing:
      text_cleaning:
        remove_stopwords: true
    """
    
    # When/Then
    with pytest.raises(ValueError):
        ETLConfig(**yaml.safe_load(config_yaml))

def test_environment_overrides(temp_config, monkeypatch):
    """Test that environment variables can override config values."""
    # Given
    monkeypatch.setenv('ETL_BANKS__0__NAME', 'ModifiedBank')
    monkeypatch.setenv('ETL_PROCESSING__TEXT_CLEANING__REMOVE_STOPWORDS', 'false')
    
    # When
    config_manager = ConfigManager(config_path=temp_config)
    config = config_manager.get_config()
    
    # Then
    assert config.banks[0].name == "ModifiedBank"
    assert config.processing.text_cleaning.remove_stopwords is False

def test_invalid_yaml(tmp_path):
    """Test handling of invalid YAML."""
    # Given
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("invalid: yaml: : :")
    
    # When/Then
    with pytest.raises(ConfigError):
        ConfigManager(config_path=bad_yaml).get_config()

def test_missing_config():
    """Test handling of missing config file."""
    # Given
    non_existent = Path("/non/existent/config.yaml")
    
    # When/Then
    with pytest.raises(FileNotFoundError):
        ConfigManager(config_path=non_existent).get_config()
