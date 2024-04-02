#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(name='sccastle',
      version='1.0.2',
      packages=find_packages(),
      description='single-cell Chromatin Accessibility Sequencing data analysis via discreTe Latent Embedding',
      long_description='',

      author='Xuejian Cui',
      author_email='cuixj19@mails.tsinghua.edu.cn',
      url="https://github.com/cuixj19/CASTLE",
      scripts=['CASTLE.py'],
      python_requires='>3.6.0',
      license='MIT',

      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.7',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
     ],
     
    install_requires=[
        'numpy==1.23.1',
        'scanpy>=1.9.1',
        'episcanpy==0.3.2',
        'umap>=0.1.1',
        'louvain>=0.8.0',
        'torch>=1.9.1',
        'tqdm>=4.26.0',
    ]
     )
