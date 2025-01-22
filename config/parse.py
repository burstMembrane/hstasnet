import yaml
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def parse_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

