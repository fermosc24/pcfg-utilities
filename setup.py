from setuptools import setup, find_packages

setup(
    name="pcfg_utilities",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "nltk",
        "numpy",
        # entropy_estimators must be installed manually as per README
    ],
    description="Utilities for working with PCFGs in Python",
    author="Fermin Moscoso del Prado Martin",
    license="MIT",
    python_requires=">=3.7",
)

