# Team 9 SCCM Datathon 2023 - COVID Trajectories Analysis

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Table of Contents

- [Team 9 SCCM Datathon 2023 - COVID Trajectories Analysis](#team-9-sccm-datathon-2023---covid-trajectories-analysis)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Dependencies](#dependencies)
  - [Usage](#usage)
  - [Citation](#citation)
  - [Acknowledgement](#acknowledgement)
  - [License](#license)

## Introduction

Here is our code to run experiments to analyse Social Determinants of Health associated with COVID-19 trajectory.

If you find this project interesting, we would appreciate your support by leaving a star ⭐ on this [GitHub repository](https://github.com/SCCMdatathon2023/team_09).

**Author:** Adrien Carrel, MSc, Meng; Tien "Amy" Bui; Yugang Jia; Lasse Hansen; Damien Archbold; Ivor S. Douglas, MD, FRCP (UK); Peter E. Morris, MD

## Installation

You can clone or fork this repository to your local machine.

```bash
git clone https://github.com/SCCMdatathon2023/team_09
```

## Dependencies

This project requires the following dependencies:

- fastdtw>=0.3.4
- hdbscan>=0.8.33
- kaleido>=0.1.0.post1
- matplotlib>=3.7.2
- nltk>=3.8.1
- numpy>=1.24.4
- pandas>=2.0.3
- plotly>=5.15.0
- scikit_learn>=1.3.0
- scipy>=1.11.1
- tableone>=0.8.0
- tqdm>=4.65.0
- ~aleido>=0.2.1

Please make sure you have the required dependencies installed before using the code in this project.

You can install all of them by running the command:

```bash
pip install -r requirements.txt
```

Some other packages may be required.

## Usage

To use our project or replicate the results, run our different python script or you can simply run the Python commands individually. For example, for the timeseries clustering project:

```python
config = {
    # your parameters here
}
project = Team9(**config)
project.run_clustering()
project.analyze_clusters()
```

## Citation

If you use piece of code from this project in your research or work, please consider citing it using the following BibTeX entry:

```bibtex
[To be completed, and complete the citation information in the CITATION.cff file provided in the repository.]
```

## Acknowledgement

SCCM Discovery Datathon 2023, the mentors and the organizers for helping us out during these two days.

## License

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/). Feel free to use and modify the code as per the terms of the license.
