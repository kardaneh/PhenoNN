Installation
============

Prerequisites
-------------

- Python 3.10 or higher.
- PyTorch 2.0 or higher.
- CUDA-capable GPU (optional, for faster training).
- Git (to clone the repository).

Installing the ``uv`` package manager
-------------------------------------

``uv`` is a fast Python package installer and resolver. Install it with:

.. code-block:: bash

    # Linux / macOS
    curl -LsSf https://astral.sh/uv/install.sh | sh

You should see output similar to:

.. code-block:: text

    downloading uv 0.11.2 x86_64-unknown-linux-gnu
    installing to /home/username/.local/bin
      uv
      uvx
    everything's installed!

Add ``uv`` to your ``PATH``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use ``uv`` from anywhere, add it to your shell configuration:

.. code-block:: bash

    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc

Verify:

.. code-block:: bash

    uv --version
    which uv

Expected output:

.. code-block:: text

    uv 0.11.2
    /home/username/.local/bin/uv

Installing PhenoNN
------------------

1. **Clone the repository**:

   .. code-block:: bash

       git clone https://github.com/kardaneh/PhenoNN.git
       cd PhenoNN

2. **Create a virtual environment**:

   .. code-block:: bash

       uv venv --python 3.10     # or 3.11, 3.12

   That creates a virtual environment in ``.venv``.

3. **Activate the virtual environment**:

   .. code-block:: bash

       source .venv/bin/activate

   The prompt now starts with ``(PhenoNN)`` or the name of the
   environment you activated.

4. **Install PhenoNN in editable mode**:

   Base install (lightweight import and CLI)::

       uv pip install -e .

   Full install (models, training, prediction and figures)::

       uv pip install -e ".[full]"

   Developer install (adds ruff and pre-commit)::

       uv pip install -e ".[full,dev]"

   Documentation install (adds Sphinx and its extensions)::

       uv pip install -e ".[full,docs]"

   All at once::

       uv pip install -e ".[full,dev,docs]"

   The ``-e`` flag installs the package in editable mode: changes to
   the Python source are picked up immediately, without a reinstall.

Verification
------------

After installation, check the environment:

.. code-block:: bash

    # Python version
    python --version

    # Installation smoke test — imports, public API, model factory, CLI
    python tests/test_phenonn_installation.py

    # CLI
    phenonn --version
    phenonn --help

    # PyTorch
    python -c "import torch; print(f'PyTorch {torch.__version__}')"

The smoke test prints one line per group and a final summary. On success
the last line is::

    ✅ All checks passed — PhenoNN is ready to use.

If a group fails, the line names the module or attribute that could not
be imported, which is enough to pinpoint the problem.

Dependencies
------------

The base install provides the lightweight import/CLI path. The ``full``
extra installs the model, training, prediction and figure stack:

- **PyTorch** ≥ 2.0 — deep learning framework
- **NumPy** ≥ 1.23
- **Pandas** ≥ 2.0
- **Xarray** ≥ 2023.1
- **netCDF4** ≥ 1.6
- **h5netcdf** ≥ 1.1
- **Matplotlib** ≥ 3.5
- **SciPy** ≥ 1.10
- **scikit-learn** ≥ 1.3
- **rich** ≥ 13.0 — console output
- **tqdm** ≥ 4.65
- **mpltex** ≥ 0.7 — Matplotlib styles used by the diagnostic plots

Optional extras
~~~~~~~~~~~~~~~

- **full** — the model, training, prediction and figure dependencies.
- **dev** — ``ruff`` and ``pre-commit``. Needed for the lint/format gate.
- **docs** — Sphinx, ``myst-parser``, ``sphinx-rtd-theme``,
  ``sphinx-copybutton``, ``sphinx-autodoc-typehints``. Needed to build
  the documentation locally.

Troubleshooting
---------------

**ModuleNotFoundError**

The virtual environment is not activated. Run::

    source .venv/bin/activate

**PyTorch installation issues**

If pip resolves a version that does not match your CUDA toolkit, install
PyTorch separately with the index URL that matches your system::

    uv pip uninstall torch
    uv pip install torch --index-url https://download.pytorch.org/whl/cpu

**Permission errors**

Do not use ``--user`` inside a virtual environment; that flag causes
more problems than it solves. Instead, recreate the environment from
scratch::

    rm -rf .venv
    uv venv --python 3.10
    source .venv/bin/activate
    uv pip install -e ".[full]"

**Python version issues**

PhenoNN requires Python 3.10 or higher. If your ``python --version``
returns something older, tell ``uv`` which interpreter to use::

    uv venv --python 3.10

For anything else, open an issue on the GitHub repository:
https://github.com/kardaneh/PhenoNN
