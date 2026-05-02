__version__ = 'compat'


def load(filename):
    import json
    from pathlib import Path
    path = Path(filename)
    if path.suffix.lower() == '.json':
        return json.loads(path.read_text())
    if path.suffix.lower() in ('.yml', '.yaml'):
        try:
            import yaml
        except Exception as exc:
            raise ImportError('PyYAML is required to load YAML config files') from exc
        return yaml.safe_load(path.read_text())
    raise IOError('Only json/yaml config loading is implemented in the local mmcv compatibility shim')
