# STORI_SPARE
Code for fitting protein-based univariate and bivariate Cox models on the STORI and SPARE datasets.

This approach is described in the following paper:
```
Pierre et al., External validation of serum biomarkers predicting short-term and mid/long-term relapse in Crohn’s disease patients stopping infliximab.
Under review, 2024
```

## Dependencies
The following dependencies are necessary to run the code: numpy, pandas, lifelines, and scikit-learn.

You can use `pip install -r requirements.txt` to install all the requirements.


## Data
The STORI and SPARE data will be available, should our paper be accepted. This data is needed to be able to run the analyses as shown below.


## Run the analysis

To start the analysis, run the following command:

```python cox.py --help```

This will display the parameters that you can use to configure the analysis.

To run the **univariate** analysis with the parameters used to obtain the results presented in the paper, run the following command (here for the short-term dataset, for example):

```python cox.py --type univariate --penalizer 0 --stratification_type before_6months```

To run the **bivariate** analysis with the parameters used to obtain the results presented in the paper, run the following command (here for the short-term dataset, for example):

```python cox.py --type bivariate --penalizer 0.01 --n_bootstraps 500 --stratification_type before_6months```

In each case, when the analysis is finished, a txt file will be generated containing the results of the analysis. This txt file can be opened with excel, for example.
