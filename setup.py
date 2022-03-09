"""
Created on Tue Jul 31 12:29:16 2018

@author: jordy
"""

"""sim_annealing package setup script

   To create a source distribution of the sim_annealing
   package run

   python setup_simanneal.py sdist

   which will create an archive file in the 'dist' subdirectory.
   The archive file will be  called 'sim_annealing-1.0.zip' and will
   unpack into a directory 'sim_annealing-1.0'.

   An end-user wishing to install the sim_annealing package can simply
   unpack 'sim_annealing-1.0.zip' and from the 'sim_annealing-1.0' directory and
   run

   python setup.py install

   which will ultimately copy the sim_annealing package to the appropriate
   directory for 3rd party modules in their Python installation
   (somewhere like 'c:\python27\libs\site-packages').

   To create an executable installer use the bdist_wininst command

   python setup.py bdist_wininst

   which will create an executable installer, 'sim_annealing-1.0.win32.exe',
   in the current directory.

"""

from setuptools import setup, find_packages

# package naming
DISTNAME = "sim_annealing"

# descriptions
DESCRIPTION = "'sim_annealing' package version"
LONG_DESCRIPTION = "'sim_annealing' package and extensions\n"

# developer(s)
AUTHOR = "VAR Strategies"
EMAIL = "jeroen.kerkhof@varstrategies.com"

# versioning
MAJOR = 0
MINOR = 0
MICRO = 1
ISRELEASED = False
VERSION = "%d.%d.%d" % (MAJOR, MINOR, MICRO)
QUALIFIER = ""

FULLVERSION = VERSION
write_version = True

# if not ISRELEASED:
# FULLVERSION += '.dev'

CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Finance/Research",
    "Programming Language :: Python",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 2.6",
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: 3.2",
    "Programming Language :: Python :: 3.3",
    "Programming Language :: Python :: 3.4",
    "Programming Language :: Python :: 3.6",
    "Topic :: Scientific/Finance",
]

setup(
    name=DISTNAME,
    version=FULLVERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    maintainer=AUTHOR,
    maintainer_email=EMAIL,
    long_description=LONG_DESCRIPTION
    #    , setup_requires=['numpy','scipy', 'matplotlib', 'pandas', 'statsmodels']
    #    , install_requires=['numpy','scipy', 'matplotlib', 'pandas', 'statsmodels']
    #    , packages=['sim_annealing',
    #               'sim_annealing.support']
    ,
    packages=find_packages(),
)
