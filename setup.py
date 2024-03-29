from setuptools import setup, find_packages
from ml import __version__
import os
import io


def read(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    with io.open(filepath, mode="r", encoding="utf-8") as f:
        return f.read()


setup(
    name="syllogio-ml",
    version=__version__,
    url="https://github.com/syllogio/syllogio-ml",
    license="MIT",
    author="Peter Sieg <chasingmaxwell@gmail.com>, Patrick Coffey <patrickcoffey48@gmail.com>",
    author_email="chasingmaxwell@gmail.com",
    description="Do artificially intelligent things with propositions in an argument.",
    packages=find_packages(exclude=["tests"]),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=read("requirements.txt").splitlines(),
    zip_safe=False,
)
