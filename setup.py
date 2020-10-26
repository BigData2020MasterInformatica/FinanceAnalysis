import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "finanSP",
    version = "0.0.1",
    author = "Big Data subject team",
    author_email = "ajnebro@uma.es",
    description = ("Project to analyze finance data with Apache Spark"),
    license = "MIT",
    keywords = "Spark Finance Streaming Clustering Classification",
    packages=['finansp'],
    long_description=read('README.rst'),
    classifiers=[
        "Development Status :: 1 - Planning",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
    ],
)
