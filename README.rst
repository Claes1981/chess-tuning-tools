This is a fork of Chess Tuning Tools by Karlson "kiudee" Pfannschmidt, (https://github.com/kiudee/chess-tuning-tools), with some modifications. 

These modifications are (at the time of writing):

* It saves the intermediate result to disk after each round and allows resuming from the previous round within an iteration. (Useful if you run games with long time control.)
* It allows tuning against multiple engines. It selects each round as engine2 at random one of all engines after the first specified in the configuration json file.
* The use of an opening book is optional.
* It displays the scales (currently a bit messy) of the partial dependence contour plots, including the differences between the maximum and minimum values.
* It plots an active subspace of the model predicted mean, with help from the ATHENA package, https://mathlab.github.io/ATHENA/.  
* It has an option to choose aqcuisition function at random each iteration, and an option to allow the user to set the lcb acquisition function alpha parameter.
* It lets the user set normalize_y of the optimizer to true or false through an option.
* It restarts the engines between each game (Cutechess-cli option "-each restart=on"), making sure no information is passed from previous games in the engine.
* It has an option (disabled by default) to reset the optimizer object of Bayes-skopt, https://github.com/kiudee/bayes-skopt/blob/master/bask/optimizer.py, each iteration in an attempt to workaround https://github.com/kiudee/chess-tuning-tools/issues/118.
* It lets the user, through an experimental option, multiply the noise of the observations by a coefficent when sent to the optimizer. 
  It is intended as an experimental attempt to work around the https://github.com/kiudee/chess-tuning-tools/issues/118 bug.
* It includes the current settings in the log when starting or resuming, and it saves the games in separate PGN files for different tuning configuration files.
* It allows using the latest 0.24.* version of Scikit-learn.

---------------


.. image:: https://raw.githubusercontent.com/kiudee/chess-tuning-tools/master/docs/_static/logo.png

|

.. image:: https://img.shields.io/pypi/v/chess-tuning-tools.svg?style=flat-square
        :target: https://pypi.python.org/pypi/chess-tuning-tools

.. image:: https://img.shields.io/travis/com/kiudee/chess-tuning-tools?style=flat-square
        :target: https://travis-ci.com/github/kiudee/chess-tuning-tools

.. image:: https://readthedocs.org/projects/chess-tuning-tools/badge/?version=latest&style=flat-square
        :target: https://chess-tuning-tools.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status
        
.. image:: https://zenodo.org/badge/234719111.svg?style=flat-square
   :target: https://zenodo.org/badge/latestdoi/234719111


A collection of tools for local and distributed tuning of chess engines.


* Free software: Apache Software License 2.0
* Documentation: https://chess-tuning-tools.readthedocs.io.


Features
--------

* Optimization of chess engines using state-of-the-art `Bayesian optimization <https://github.com/kiudee/bayes-skopt>`_.
* Support for automatic visualization of the optimization landscape.
* Scoring matches using a Bayesian-pentanomial model for paired openings.

Quick Start
-----------

In order to be able to start the tuning, first create a python
environment (at least Python 3.7) and install chess-tuning-tools by typing::

   pip install chess-tuning-tools

Furthermore, you need to have `cutechess-cli <https://github.com/cutechess/cutechess>`_
in the path. The tuner will use it to run matches.

To execute the local tuner, simply run::

   tune local -c tuning_config.json

Take a look at the `usage instructions`_ and the `example configurations`_ to
learn how to set up the ``tuning_config.json`` file.


.. _example configurations: https://github.com/kiudee/chess-tuning-tools/tree/master/examples
.. _usage instructions: https://chess-tuning-tools.readthedocs.io/en/latest/usage.html
