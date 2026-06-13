import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


class TestSetupMetadata(unittest.TestCase):
    def test_setup_includes_subpackages_and_runtime_dependencies(self):
        setup_path = Path(__file__).resolve().parents[1] / 'setup.py'

        with patch('setuptools.setup') as mock_setup:
            spec = importlib.util.spec_from_file_location('gautils_setup_for_test', setup_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

        kwargs = mock_setup.call_args.kwargs
        self.assertIn('gautils.feishu', kwargs['packages'])
        self.assertIn('gautils.feishu.core', kwargs['packages'])

        install_requires = kwargs['install_requires']
        for dep in ('numpy', 'requests', 'lark-oapi', 'SQLAlchemy'):
            self.assertTrue(
                any(item.startswith(dep) for item in install_requires),
                f'{dep} should be declared in install_requires'
            )


if __name__ == '__main__':
    unittest.main()
