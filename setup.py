import setuptools
from distutils.core import setup
import platform

try:
    from Cython.Distutils import build_ext
except:
    from distutils.command import build_ext

import numpy as np


__version__ = '0.1.0'

mappy_name = 'mtlchmm'
maintainer = 'Jordan Graesser'
maintainer_email = ''
description = 'Multi-temporal land cover maps with a Hidden Markov Model'
git_url = 'https://github.com/jgrss/mtlchmm.git'

with open('README.md') as f:
    long_description = f.read()

with open('LICENSE.txt') as f:
    license_file = f.read()

with open('AUTHORS.txt') as f:
    author_file = f.read()

required_packages = ['numpy',
                     'rasterio',
                     'affine']


def get_packages():
    return setuptools.find_packages()


def get_package_data():
    return {'': ['*.md', '*.txt']}


def setup_package():

    if platform.system() != 'Windows':
        include_dirs = [np.get_include()]
    else:
        include_dirs = None

    metadata = dict(name=mappy_name,
                    maintainer=maintainer,
                    maintainer_email=maintainer_email,
                    description=description,
                    license=license_file,
                    version=__version__,
                    long_description=long_description,
                    author=author_file,
                    packages=get_packages(),
                    package_data=get_package_data(),
                    cmdclass=dict(build_ext=build_ext),
                    zip_safe=False,
                    download_url=git_url,
                    install_requires=required_packages,
                    include_dirs=include_dirs)

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
