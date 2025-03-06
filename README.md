# ChroMo PE
Chromatography model Parameter Estimator
This program serves as demonstration of the new approach of chromatography model parameters determination described in paper that is curretly being prepared

Diagram of the architecture: https://drive.google.com/file/d/12WWNgDYQipY8OY_eu51zKs-shQmxNdz2/view?usp=sharing

---


- [Chromo](#chromo)
  - [Description](#description)
  - [Requirements](#requirements)
    - [Used components](#used-components)
    - [Further requirements](#further-requirements)
  - [Installation](#installation)
  - [Usage](#usage)

## Description

This application can calculate parameters of chromatography model based on experimental data.

Equilibrium dispersion model with linear and Lanqmuir isotherm are implemented so far.

Application is in development.

Application can be downloaded on [GitHub repository](https://github.com/meloun67/ChroMo).

## Requirements

### Used components

Application is written in Python and requires Python interpreter to run, it was tested on Python 3.9

Required libraries are described in [requirements file](requirements.txt).

### Further requirements

To run application locally, MongoDB database running on local computer is required.

Application was tested on [MongoDB Community Server 6.0](https://www.mongodb.com/try/download/community-edition).

## Installation

1. Pull Git project from this repository.
2. Open command line and navigate to projects root folder.
3. Activate virtual environment\
  On Windows run:\
  `venv\Scripts\activate.bat`\
  On Linux run:\
  `source venv/Scripts/activate`
5. Install requirements `pip install -r requirements.txt`.
6. Run application `python main.py`.

To run calculations without web interface, remove Web_Server() from main function.
This will run calculation automatically with variables defined inside the main function, which you can change.

## Usage

Locally run application can be accessed with browser on address `localhost:6969`.

First you'll need to create an account and log in.

After logging in, you will be redirected to page where you upload the files with the experimental data. Example files can be found [here](data/Suc_Glu_GE).

Then you can choose between "Loss Function Analysis", "Manual Estimator" and "Optimization".

With "Loss Function Analysis" you can visualize loss function values in range of parameters.

With "Manual Estimator" you can get model results with manually chosen parameters.

With "Optimization" you can run parameter estimation.

Results of previously run estimations can be found in "Results" tab.

Number of differences and time for model calculation can be set in "Solver settings" tab.
