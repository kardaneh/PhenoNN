Testing
=======

PhenoNN ships two kinds of tests:

- a **smoke test** that checks the whole pipeline end to end on a tiny
  dataset (a couple of minutes, CPU-only);
- a set of **unit tests** for the individual modules (seconds, no data
  required).

Both use the standard library's :mod:`unittest`. You do not need
``pytest`` to run them.

Installation check
------------------

Before anything else, run the installation smoke test. It verifies that
every dependency and every public module imports, that the model factory
builds the three architectures the cluster jobs use, and that the CLI
answers::

    python tests/test_phenonn_installation.py

The script prints a summary and exits non-zero if any group fails::

    ✅ dependencies
    ✅ phenonn modules
    ✅ public API
    ✅ model factory
    ✅ CLI
    ✅ All checks passed — PhenoNN is ready to use.

A log file ``test_phenonn_installation.log`` is written next to the
script with the same content (minus the terminal formatting).

Unit tests
----------

The unit tests live in ``tests/`` and are run with :mod:`unittest`::

    python -m unittest tests.test_evaluater
    python -m unittest tests.test_rnn
    python -m unittest tests.test_diagnostics

To run all of them in one go::

    python -m unittest discover -s tests

Or, if you prefer the verbose per-test output::

    python -m unittest discover -s tests -v

The tests use no GPU and no external data files. A run of the full suite
takes under a minute on a laptop.

Running the pipeline smoke test
-------------------------------

The installation smoke test checks that everything imports. To check
that the *training pipeline itself* still works after a code change, run
the tiny training script::

    bash scripts/smoke_train.sh

It loads a few sites, trains an LSTM for two epochs, and writes a
checkpoint under ``runs/smoke/smoke_lstm/``. If the checkpoint exists at
the end, the data loading, model construction, loss, optimizer and
checkpoint writer all work together.

Testing the CLI
---------------

The two sub-commands are also reachable through the ``phenonn`` console
script once the package is installed::

    phenonn --version
    phenonn --help
    phenonn train --help
    phenonn predict --help

Their behaviour is covered by the ``CLI`` section of
``tests/test_phenonn_installation.py``.

Writing new tests
-----------------

Follow the layout of the existing test modules:

1. Put the file in ``tests/`` and name it ``test_<module>.py``.
2. Start with the standard copyright header used across the repository.
3. Import from the public surface of ``phenonn``
   (``phenonn.data``, ``phenonn.models``, ``phenonn.utils``, …), not from
   the legacy code in ``legacy/``.
4. Write a class per module or per behaviour, deriving from
   :class:`unittest.TestCase`.
5. At the bottom of the file, provide a ``run_tests()`` function that
   builds a suite and returns ``result.wasSuccessful()``, plus the
   standard ``if __name__ == "__main__": sys.exit(0 if run_tests() else 1)``
   block. This lets the file run both under ``python -m unittest`` and as
   ``python tests/test_<module>.py``.

A minimal template::

    import sys
    import unittest

    import torch

    from phenonn.utils.evaluater import nan_safe_loss


    class TestNanSafeLoss(unittest.TestCase):

        def test_mse_zero_when_identical(self):
            x = torch.randn(2, 1, 36)
            loss = nan_safe_loss(x, x.clone(), loss_type="mse")
            self.assertAlmostEqual(loss.item(), 0.0, places=6)


    def run_tests():
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        suite.addTests(loader.loadTestsFromTestCase(TestNanSafeLoss))
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return result.wasSuccessful()


    if __name__ == "__main__":
        success = run_tests()
        sys.exit(0 if success else 1)

Continuous integration
----------------------

Every push runs the installation checks and each unit test module explicitly on
GitHub Actions. The installation checks are separate commands because discovery
does not execute the script-style check in ``tests/test_phenonn_installation.py``.

For a local full run, discovery still works:

.. code-block:: bash

    python -m unittest discover -s tests

The workflow file at ``.github/workflows/ci.yaml`` contains the authoritative
list of commands used by CI.
