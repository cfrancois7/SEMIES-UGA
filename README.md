# SEMIES UGA

In this repo, you'll find the notebook to explain the methodology and the challenge.
The notebooks download and use the data coming from [Perspectiva-Solution](https://huggingface.co/perspectiva-solution).
The data are originally sourced from the [Grand Débat National 2019](www.data.gouv.fr/fr/datasets/donnees-ouvertes-du-grand-debat-national/).

## How to prepare the virtual environmet

1. Install `uv` : `pip install uv`.
   See the [UV documentation](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer).
2. Create one virtual environment: `uv venv --python 3.12 --seed`
   See the [virtual environment tutorial](https://docs.astral.sh/uv/pip/environments/).
3. Install the required package: `uv pip install -r requirements.txt`
4. Open the notebook with [VScode](https://code.visualstudio.com/) or [Jupyterlab](https://jupyter.org/) (installed through requirements)