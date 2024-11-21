#!/usr/bin/env python3

# Copyright (C) 2024 RidgeRun, LLC (http://www.ridgerun.com)
# All Rights Reserved.

# The contents of this software are proprietary and confidential to RidgeRun,
# LLC.  No part of this program may be photocopied, reproduced or translated
# into another programming language without prior written consent of
# RidgeRun, LLC.  The user is free to modify the source code after obtaining
# a software license from RidgeRun.  All source code changes must be provided
# back to RidgeRun without any encumbrance.

from setuptools import setup, find_packages
from os import path
import unittest

here = path.abspath(path.dirname(__file__))

# automatically use README.md as long_description
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='CLRerNet',
    version='0.1.0',
    description=("CLRerNet model for lane detection"),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ridgerun-ai/CLRerNet',
    author='RidgeRun',
    author_email='support@ridgerun.com',
    # Exclude the tests from the distribution package
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    # Included scripts should be in the bin folder
    scripts=['demo/video_demo.py'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Programming Language :: Python :: 3.10.14',
        "License :: Other/Proprietary License",
    ],
    python_requires='>=3.10.14',
    # Package dependencies
    install_requires=required,
)