.. _intro-install:

==================
Installation guide
==================


Supported Python versions
=========================

Lamassu has been tested with Python 3.10. It may work with older versions of Python but it is not guaranteed.


Installing Lamassu
==================

If you are already familiar with installation of Python packages, we can install Lamassu and its dependencies from
`PyPI <https://pypi.org/project/lamassu/>`_ with::

    pip3 install lamassu

We strongly recommend that you install Lamassu in :ref:`a dedicated virtualenv <intro-using-virtualenv>`, to avoid
conflicting with your system packages.

If you're using `Anaconda <https://docs.anaconda.com/anaconda/>`_ or
`Miniconda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_, please allow me to
apologize because I hate those two, so we won't install the package from there.


Installing from Source
======================

When we want to apply a bug fix quicly by installing Lamassu locally, we can use::

    git clone https://github.com/QubitPi/lamassu.git
    cd lamassu
    pip3 install -e .


.. _intro-using-virtualenv:

Using a virtual environment (recommended)
-----------------------------------------

We recommend installing lamassu a virtual environment on all platforms.

Python packages can be installed either globally (a.k.a system wide), or in user-space. We do not recommend installing
lamassu system wide. Instead, we recommend installing lamassu within a "virtual environment" (:mod:`venv`),
which keep you from conflicting with already-installed Python system packages.

See :ref:`tut-venv` on how to create your virtual environment.

Once you have created a virtual environment, we can install lamassu inside it with ``pip3``, just like any other
Python package.
