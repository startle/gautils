import os
import unittest
from pathlib import Path


def load_env_test():
    env_path = Path(__file__).resolve().parents[1] / '.env.test'
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


def require_env(*names):
    load_env_test()
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise unittest.SkipTest('missing integration test env: ' + ', '.join(missing))
    return [os.environ[name] for name in names]
