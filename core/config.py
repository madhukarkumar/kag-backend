import yaml
from typing import Dict, Any
from pathlib import Path
import re
import os

class ConfigLoader:
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """Load configuration from yaml file."""
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    def reload(self):
        """Reload configuration from file."""
        self._load_config()
    
    @property
    def knowledge_creation(self) -> Dict[str, Any]:
        """Get knowledge creation settings."""
        return self._config['knowledge_creation']
    
    @property
    def retrieval(self) -> Dict[str, Any]:
        """Get retrieval settings."""
        return self._config['retrieval']
    
    def get_chunking_rules(self) -> str:
        """Get formatted chunking rules for Gemini prompt."""
        rules = self.knowledge_creation['chunking']['semantic_rules']
        return "\n".join(f"{i+1}. {rule}" for i, rule in enumerate(rules))
    
    def get_extraction_prompt(self, text: str) -> str:
        """Get the extraction prompt with the text inserted."""
        template = self._config["knowledge_creation"]["entity_extraction"]["extraction_prompt_template"]
        
        # Simple text cleaning - just remove problematic characters
        cleaned_text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()
        
        # Format the template with text as a raw string to avoid formatting issues
        return template.format(text=cleaned_text)
    
    def get_response_prompt(self, query: str, context: str) -> str:
        """Get formatted response generation prompt."""
        template = self.retrieval['response_generation']['prompt_template']
        return template.format(query=query, context=context)

    def update_config(self, new_config: Dict[str, Any]) -> None:
        self._config.update(new_config)
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self._config, f)

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

# Global instance
config = ConfigLoader()
