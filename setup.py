from setuptools import find_packages, setup

setup(
  name = 'gautils',
  packages = find_packages(),
  version = '1.1.2',
  license='MIT',
  description = 'gau\'s utils',
  author = 'GaU',
  author_email = '690478206@qq.com',
  url = 'https://github.com/startle/gautils/',
  download_url = 'https://github.com/startle/gautils/archive/refs/heads/main.zip',
  keywords = ['gau', 'utils'],
  install_requires=[
    'numpy>=2.2.6',
    'pandas>=2.2.3',
    'mysql-connector==2.2.9',
    'PyYAML>=6.0.2',
    'requests>=2.31.0',
    'lark-oapi>=1.4.24',
    'SQLAlchemy>=2.0.43',
  ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3.6',
  ],
)
