import os

def load_dotenv():
    """Simple implementation of dotenv's load_dotenv function"""
    try:
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip().strip('"\'')
        return True
    except FileNotFoundError:
        return False