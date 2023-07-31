# -*- coding: utf-8 -*-

# header

"""Setup and Install Script."""

from setuptools import find_packages, setup


with open("README.md") as f:
    long_description = f.read()


setup(
    name="simulator-aspire-poc",
    version="0.0.0a1",
    description="Laboratory Automation Simulator for Scheduling and Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/coleygroup/simulator-aspire",
    classifiers=[
        "Environment :: Console",
        # "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Medical Science Apps",
        "Intended Audience :: Science/Research",
    ],
    packages=find_packages(),
    # package_data={
    #     "ord_schema.proto": [
    #         "dataset.proto",
    #         "reaction.proto",
    #         "test.proto",
    #     ],
    # },
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.23.0",
    ],
    extras_require={
        #     "docs": [
        #         "ipython>=7.18.1",
        #         "Pygments>=2.7.2",
        #         "sphinx>=3.3.1",
        #         "sphinx-rtd-theme>=0.5.0",
        #         "sphinx-tabs>=1.3.0",
        #     ],
        #     "examples": [
        #         "glob2>=0.7",
        #         "matplotlib>=3.3.4",
        #         "scikit-learn>=0.24.1",
        #         "tensorflow>=2.4.1",
        #         "tqdm>=4.61.2",
        #         "wget>=3.2",
        #     ],
        "tests": [
            # "black[jupyter]>=22.3.0",
            # "coverage>=6.5.0",
            # "pylint>=2.13.9",
            "pytest>=7.1.3",
            "pytest-cov>=4.0.0",
            # "pytype>=2022.5.19",
            # "testing-postgresql>=1.3.0",
            # "treon>=0.1.3",
        ],
    },
)
