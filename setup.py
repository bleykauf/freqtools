from setuptools import setup, find_packages

import re
VERSIONFILE = "freq_tools/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))


setup(
    name='freq_tools',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'matplotlib', 'allantools'],
        version=verstr
)
