from setuptools import setup, find_packages

setup(
    name='freq_tools',
    version='0.1',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'matplotlib', 'allantools']
)
