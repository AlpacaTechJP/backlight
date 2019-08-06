import setuptools
import sys
import os
from glob import glob

# hack to extract metadata directly from the python package
sys.path.append("src")  # noqa
from backlight import __author__, __version__, __license__


def read(fname):
    with open(fname, "r", encoding="utf-8") as fh:
        long_description = fh.read()
        return long_description


class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches our version"""

    description = "verify that the git tag matches our version"

    def run(self):
        tag = os.getenv("CIRCLE_TAG")

        if tag != VERSION:
            info = "Git tag: {0} does not match the version of this app: {1}".format(
                tag, VERSION
            )
            sys.exit(info)


setuptools.setup(
    name="backlight",
    version=__version__,
    description="Model evaluation framework for AlpacaForecast",
    author=__author__,
    author_email="info@alpaca.ai",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    license=__license__,
    url="https://github.com/AlpacaDB/backlight.git",
    keywords="",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    py_modules=[
        os.path.splitext(os.path.basename(path))[0] for path in glob("src/*.py")
    ],
    install_requires=[
        "pandas==0.21.0",
        "numpy>=1.15.0",
        "matplotlib>=2.2.2",
        "boto3>=1.9.36",
    ],
    tests_require=[
        "pytest-cov>=2.5.1",
        "pytest-mock>=1.7.1",
        "pytest-flake8>=1.0.0",
        "pytest-sugar>=0.9.1",
        "pytest>=3.5.0",
        "autopep8>=1.2.3",
        "flake8>=3.5.0",
    ],
    cmdclass={"verify": VerifyVersionCommand},
)
