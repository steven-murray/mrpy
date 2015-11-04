from setuptools import setup, find_packages

import os
import sys

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    os.system("python setup.py bdist_egg upload")
    sys.exit()

setup(
    name="mrpy",
    version=0.1,
    packages=['mrpy'],
    install_requires=["numpy>=1.6.2",
                      "scipy>=0.12.0",
                      "mpmath"],
    #scripts=[],
    author="Steven Murray",
    author_email="steven.murray@curtin.edu.au",
    description="An efficient alternative halo mass function distribution",
    long_description=read('README.rst'),
    license="MIT",
    keywords="halo mass function bayesian",
    url="https://github.com/steven-murray/mrpy",
    # could also include long_description, download_url, classifiers, etc.
)
