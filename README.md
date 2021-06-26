# House Prices - Advanced Regression Techniques

Repository for source code of kaggle competition: [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)


## Install
Package requirements are included in requirement.txt. This project uses **Python 3.8**. Run the following command in terminal to install the required packages:
`pip3 install -r requirements.txt`

## Project structure

- `data` contains original data
- `models` contains notebooks with models
- `src` contains helpers functions

## Run
Open `models/XGBoost.ipynb` and run all cells. Predictions will be in `data` folder

## Scores

| Model                      | CV             | RMSLE   |
| :------------------------- |:--------------:| :-------|
| DummyClassifier            | 0.19584+-0.017 | 0.44859 |
| LinearRegression           | 0.02436+-0.009 | 0.14748 |
| XGBoost                    | 0.01432+-0.002 | 0.12099 |

Results of XGBoost model entered in 8.6% of Top Leader Board on 27/06/2021
