


# Per-State LLM Configuration with GGUF models through LMStudio
# Each state has its own model and output format specification
class DynamicLLMConfigs(dict):
    """
    A dictionary that returns the config for 'infer_entity_relationships' for any key starting with 'infer_relationship_'.
    """
    def __getitem__(self, key):
        if key in super().keys():
            return super().__getitem__(key)
        if key.startswith('infer_relationship_'):
            return super().__getitem__('infer_entity_relationships')
        raise KeyError(key)

    def get(self, key, default=None):
        if key in super().keys():
            return super().get(key, default)
        if key.startswith('infer_relationship_'):
            return super().get('infer_entity_relationships', default)
        return default