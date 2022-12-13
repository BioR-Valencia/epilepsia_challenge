Epilepsia challenge
==============================

A short description of the project.

## How to start
To start, we need to do some steps:
- Create a virtual environment:
  - `python -m venv venv`
- Activate the virtual env (to do every time we enter to the terminal):
  - _Linux:_ `source venv/bin/activate`
  - _Windows:_ `venv\Scripts\activate.bat`
- Install pip packages:
  - `pip install -r requirements.txt`
- Install git precommit hooks:
  - `pre-commit install`

## How to commit
To clone the repository:
- `git clone https://github.com/BioR-Valencia/epilepsia_challenge.git`

To create a new branch:
- `git checkout -b <branch_name>`

To download the latest changes at the repo:
- `git pull`
- It's important to do this every time before start to code to avoid code incompatibility issues.

To make your changes available to everyone, you must commit your changes:
- `git add .`
- `git commit -m "A short description of your changes"`
- There is a pre-commit hook that review your changes before the commit. If it fails, you must repeat the two steps until all hooks are passed.
- It is recommended to commit every time you add/complete a new feature or function
- Finally, to upload to github: `git push`

## Useful things for jupyter notebooks
Executing the following lines in a jupyter cell, we can update our imported scripts and jupyter will reimport the script
automatically:
```
%load_ext autoreload
%autoreload 2
```

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
