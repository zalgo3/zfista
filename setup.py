import sys
from pathlib import Path

from setuptools import find_packages, setup

from zfista import __version__ as version

LICENSE = "MIT"

try:
    with open("README.md") as f:
        readme = f.read()
except IOError:
    readme = ""

if not (3, 7) <= sys.version_info[:2]:
    raise Exception(
        "zfista requires Python 3.7 or later. \n Now running on {0}".format(sys.version)
    )

with Path("requirements.txt").open() as f:
    INSTALL_REQUIRES = [line.strip() for line in f.readlines() if line]

setup(
    name="zfista",
    author="Hiroki Tanabe",
    author_email="tanabe.hiroki.45n@kyoto-u.jp",
    url="https://github.com/zalgo3/fista",
    description="A globally convergent fast iterative shrinkage-thresholding algorithm with a new momentum factor for single and multi-objective convex optimization",
    long_description=readme,
    version=version,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    test_suite="tests",
    license=LICENSE,
)
