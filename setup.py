from setuptools import setup
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
setup(
    name='COTM',
    version='0.01',
    # packages=['COTM'],
    url='https://github.com/alyrazik/CompareTopicModels',
    license='',
    author='Aly Abdelrazek',
    author_email='alyrazik@gmail.com',
    description='Compare Topic Models',
    long_description=long_description,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ],
    package_dir={"": "COTM"},
    packages=setuptools.find_packages(where="COTM"),
    python_requires=">=3.6",
)