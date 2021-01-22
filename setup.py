from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    install_requires=requirements
)
