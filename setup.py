import os
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
# def read(fname):
#     return open(os.path.join(os.path.dirname(__file__), fname)).read()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = "cdcwonderpy",
    version = "0.0.1",
    author = "Theodore L Caputi",
    author_email = "tcaputi@gmail.com",
    description = "A Python package to automatically fill out the CDC WONDER MCD form",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tlcaputi/cdcwonderpy",
    license = "No License",
    keywords = "",
    packages=find_packages()
)
