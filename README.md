# Wine Quality Analysis
## Data Source
The data was sourced from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/wine+quality).

Special thanks to the following for the data:
P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

## About the Data
There are two data files physicochemical and a quality variables for red and white wine (one file each).

The two datasets contain information on the variants of the Portuguese "Vinho Verde" wine.

### Data Variables
Fixed Acidity	Tartaric, malic, citric, and succinic acids measure.

  - Volatile Acidity:	A lower value indicates a fruiter wine
  - Citric Acid:	Added to increase acidity 
  - Residual Sugar:	Natural grape sugars after fermentation
  - Chlorides:	Contributes to saltiness of wine
  - Free Sulfur Dioxide:	Preservative anti-bacterial
  - Total Sulfur Dioxide:	Total Sulfur Dioxide (TSO2) is the portion of SO2 that is free in the wine plus the portion that is bound to other chemicals in the wine such as aldehydes, pigments, or sugars
  - Density:	Density against water, 1 = Density of water.
  - pH:	Scale of acidity, affects the color, oxidation, and lifespan of the wine
  - Sulphates:	Protects against wine oxidation
  - Alcohol:	Alcohol by volume %
  - Quality:	Quality rank 1 through 10
  - Wine Type:	If the wine is red or white
  - ID:	Distinct id of observation

** There is no missing data so no imputation is needed.

## Important Notes
* There is a variable (chlorides) with a skew of 5.4, indicating a very highly skewed positive variable. Fixed acidity, volatile acidity, residual sugar, free sulfur dioxide, and, sulfates have a high positive skew, this indicates the value of the median and mode are less than that of the mean.
* There are 488 observations with a z-score greater than 3
* There are 49 observations with a z-score greater than 6
* There are 20 observations with a z-score greater than 9
* There are 4 observations with a z-score greater than 12
* There are 4 observations with a z-score greater than 15
* In the EDA step of the analysis, 9 variables were identified being great outliers
* High negative correlation between density and alcohol (-0.69), high positive correlation between free sulfur dioxide and total sulfur dioxide, and high positive correlation between residual sugar and density (0.55)

## Data Preparation Steps
  - Create a column in each csv file with either the words red or white, indicating if the wine is red or white
  - Join red and white wine data frames
  - Create ID column
  - LOG10 transformation for radar plot
  - Min Max Standardization for outliers
  - New data frame with trimmed of 9 outliers
  - Remove highly correlated variables.
  - New variable: 1 = if quality is greater than or equal to 7, 0 = quality is less than 7.
