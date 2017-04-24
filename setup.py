from setuptools import setup

import re

def version_number():
    with open("sslf/__init__.py") as f:
        contents = f.read()
        return re.search(r"__version__ = \"(\S+)\"", contents).group(1)

setup(name="sslf",
      version=version_number(),
      packages=["sslf"],
      install_requires=["numpy", "scipy"],
      license="MIT",
      long_description=open("README.md").read(),
)
