from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    install_requires=requirements,
    author="Adrien Corenflos, Zheng Zhao",
    version="1.0.0"
)
