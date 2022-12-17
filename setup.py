from distutils.core import setup

setup(
  name = 'gautils',
  packages = ['gautils'],
  version = '0.2',
  license='MIT',
  description = 'gau\'s utils',
  author = 'GaU',
  author_email = '690478206@qq.com',
  url = 'https://github.com/startle/gautils/',
  download_url = 'https://github.com/startle/gautils/archive/refs/heads/main.zip',
  keywords = ['gau', 'utils'],
  install_requires=[
    'pandas>=1.3.2',
    'mysql-connector==2.2.9',
    'PyYAML>=5.4.1',
  ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3.6',
  ],
)