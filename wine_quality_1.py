import pandas as pd
from pandasql import sqldf as psql
import numpy as np
import plotly.express as px
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import joblib
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from math import pi


# Import the two csv data files
df_white = pd.read_csv(r"path")
df_red = pd.read_csv(r"path")

##### Join DFs and ADD new COLUMN (wine_type)
df_red.insert(len(df_red.columns), "wine_type", "red", allow_duplicates=True)
df_white.insert(len(df_white.columns), "wine_type", "white", allow_duplicates=True)
frames = [df_white, df_red]

df_redwhite = pd.concat(frames)

# Summary Stats
df_redwhite.describe()
"MEDIAN:",df_redwhite_continuous.median()
"MODE:",df_redwhite_continuous.mode()
df_redwhite.isna().sum()

# Outliers
## z-score
#outlier z score
df_redwhite_zscore = pd.DataFrame()
df_redwhite_zscore['fixed acidity_z'] = stats.zscore(df_redwhite['fixed acidity'])
df_redwhite_zscore['volatile acidity_z'] = stats.zscore(df_redwhite['volatile acidity'])
df_redwhite_zscore['citric acid_z'] = stats.zscore(df_redwhite['citric acid'])
df_redwhite_zscore['residual sugar_z'] = stats.zscore(df_redwhite['residual sugar'])
df_redwhite_zscore['chlorides_z'] = stats.zscore(df_redwhite['chlorides'])
df_redwhite_zscore['free sulfur dioxide_z'] = stats.zscore(df_redwhite['free sulfur dioxide'])
df_redwhite_zscore['total sulfur dioxide_z'] = stats.zscore(df_redwhite['total sulfur dioxide'])
df_redwhite_zscore['density_z'] = stats.zscore(df_redwhite['density'])
df_redwhite_zscore['pH_z'] = stats.zscore(df_redwhite['pH'])
df_redwhite_zscore['sulphates_z'] = stats.zscore(df_redwhite['sulphates'])
df_redwhite_zscore['alcohol_z'] = stats.zscore(df_redwhite['alcohol'])

# 3 Z score
df_redwhite_zscore_outliers = df_redwhite_zscore[(df_redwhite_zscore['fixed acidity_z'] <= -3) | (df_redwhite_zscore['fixed acidity_z'] >= 3) |(df_redwhite_zscore['volatile acidity_z'] <= -3) | (df_redwhite_zscore['volatile acidity_z'] >= 3)| (df_redwhite_zscore['citric acid_z'] <= -3) | (df_redwhite_zscore['citric acid_z'] >= 3) | (df_redwhite_zscore['residual sugar_z'] <= -3) | (df_redwhite_zscore['residual sugar_z'] >= 3) | (df_redwhite_zscore['chlorides_z'] <= -3) | (df_redwhite_zscore['chlorides_z'] >= 3) | (df_redwhite_zscore['free sulfur dioxide_z'] <= -3) | (df_redwhite_zscore['free sulfur dioxide_z'] >= 3) | (df_redwhite_zscore['total sulfur dioxide_z'] <= -3) | (df_redwhite_zscore['total sulfur dioxide_z'] >= 3) | (df_redwhite_zscore['density_z'] <= -3) | (df_redwhite_zscore['density_z'] >= 3) | (df_redwhite_zscore['pH_z'] <= -3) | (df_redwhite_zscore['pH_z'] >= 3) | (df_redwhite_zscore['sulphates_z'] <= -3) | (df_redwhite_zscore['sulphates_z'] >= 3) | (df_redwhite_zscore['alcohol_z'] <= -3) | (df_redwhite_zscore['alcohol_z'] >= 3)]
df_redwhite_zscore_outliers.shape
# 488 Outliers with a z score (+/-) 3

# 6 Z score
df_redwhite_6zscore_outliers = df_redwhite_zscore[(df_redwhite_zscore['fixed acidity_z'] <= -6) | (df_redwhite_zscore['fixed acidity_z'] >= 6) |(df_redwhite_zscore['volatile acidity_z'] <= -6) | (df_redwhite_zscore['volatile acidity_z'] >= 6)| (df_redwhite_zscore['citric acid_z'] <= -6) | (df_redwhite_zscore['citric acid_z'] >= 6) | (df_redwhite_zscore['residual sugar_z'] <= -6) | (df_redwhite_zscore['residual sugar_z'] >= 6) | (df_redwhite_zscore['chlorides_z'] <= -6) | (df_redwhite_zscore['chlorides_z'] >= 6) | (df_redwhite_zscore['free sulfur dioxide_z'] <= -6) | (df_redwhite_zscore['free sulfur dioxide_z'] >= 6) | (df_redwhite_zscore['total sulfur dioxide_z'] <= -6) | (df_redwhite_zscore['total sulfur dioxide_z'] >= 6) | (df_redwhite_zscore['density_z'] <= -6) | (df_redwhite_zscore['density_z'] >= 6) | (df_redwhite_zscore['pH_z'] <= -6) | (df_redwhite_zscore['pH_z'] >= 6) | (df_redwhite_zscore['sulphates_z'] <= -6) | (df_redwhite_zscore['sulphates_z'] >= 6) | (df_redwhite_zscore['alcohol_z'] <= -6) | (df_redwhite_zscore['alcohol_z'] >= 6)]
df_redwhite_6zscore_outliers.shape
# 49 Outliers with a z score (+/-) 6

# 9 Z score
df_redwhite_9zscore_outliers = df_redwhite_zscore[(df_redwhite_zscore['fixed acidity_z'] <= -9) | (df_redwhite_zscore['fixed acidity_z'] >= 9) |(df_redwhite_zscore['volatile acidity_z'] <= -9) | (df_redwhite_zscore['volatile acidity_z'] >= 9)| (df_redwhite_zscore['citric acid_z'] <= -9) | (df_redwhite_zscore['citric acid_z'] >= 9) | (df_redwhite_zscore['residual sugar_z'] <= -9) | (df_redwhite_zscore['residual sugar_z'] >= 9) | (df_redwhite_zscore['chlorides_z'] <= -9) | (df_redwhite_zscore['chlorides_z'] >= 9) | (df_redwhite_zscore['free sulfur dioxide_z'] <= -9) | (df_redwhite_zscore['free sulfur dioxide_z'] >= 9) | (df_redwhite_zscore['total sulfur dioxide_z'] <= -9) | (df_redwhite_zscore['total sulfur dioxide_z'] >= 9) | (df_redwhite_zscore['density_z'] <= -9) | (df_redwhite_zscore['density_z'] >= 9) | (df_redwhite_zscore['pH_z'] <= -9) | (df_redwhite_zscore['pH_z'] >= 9) | (df_redwhite_zscore['sulphates_z'] <= -9) | (df_redwhite_zscore['sulphates_z'] >= 9) | (df_redwhite_zscore['alcohol_z'] <= -9) | (df_redwhite_zscore['alcohol_z'] >= 9)]
df_redwhite_9zscore_outliers.shape
# 20 Outliers with a z score (+/-) 9

# 12 Z score
df_redwhite_12zscore_outliers = df_redwhite_zscore[(df_redwhite_zscore['fixed acidity_z'] <= -12) | (df_redwhite_zscore['fixed acidity_z'] >= 12) |(df_redwhite_zscore['volatile acidity_z'] <= -12) | (df_redwhite_zscore['volatile acidity_z'] >= 12)| (df_redwhite_zscore['citric acid_z'] <= -12) | (df_redwhite_zscore['citric acid_z'] >= 12) | (df_redwhite_zscore['residual sugar_z'] <= -12) | (df_redwhite_zscore['residual sugar_z'] >= 12) | (df_redwhite_zscore['chlorides_z'] <= -12) | (df_redwhite_zscore['chlorides_z'] >= 12) | (df_redwhite_zscore['free sulfur dioxide_z'] <= -12) | (df_redwhite_zscore['free sulfur dioxide_z'] >= 12) | (df_redwhite_zscore['total sulfur dioxide_z'] <= -12) | (df_redwhite_zscore['total sulfur dioxide_z'] >= 12) | (df_redwhite_zscore['density_z'] <= -12) | (df_redwhite_zscore['density_z'] >= 12) | (df_redwhite_zscore['pH_z'] <= -12) | (df_redwhite_zscore['pH_z'] >= 12) | (df_redwhite_zscore['sulphates_z'] <= -12) | (df_redwhite_zscore['sulphates_z'] >= 12) | (df_redwhite_zscore['alcohol_z'] <= -12) | (df_redwhite_zscore['alcohol_z'] >= 12)]
df_redwhite_12zscore_outliers.shape
# 4 Outliers with a z score (+/-) 12

# 15 Z score
df_redwhite_15zscore_outliers = df_redwhite_zscore[(df_redwhite_zscore['fixed acidity_z'] <= -15) | (df_redwhite_zscore['fixed acidity_z'] >= 15) |(df_redwhite_zscore['volatile acidity_z'] <= -15) | (df_redwhite_zscore['volatile acidity_z'] >= 15)| (df_redwhite_zscore['citric acid_z'] <= -15) | (df_redwhite_zscore['citric acid_z'] >= 15) | (df_redwhite_zscore['residual sugar_z'] <= -15) | (df_redwhite_zscore['residual sugar_z'] >= 15) | (df_redwhite_zscore['chlorides_z'] <= -15) | (df_redwhite_zscore['chlorides_z'] >= 15) | (df_redwhite_zscore['free sulfur dioxide_z'] <= -15) | (df_redwhite_zscore['free sulfur dioxide_z'] >= 15) | (df_redwhite_zscore['total sulfur dioxide_z'] <= -15) | (df_redwhite_zscore['total sulfur dioxide_z'] >= 15) | (df_redwhite_zscore['density_z'] <= -15) | (df_redwhite_zscore['density_z'] >= 15) | (df_redwhite_zscore['pH_z'] <= -15) | (df_redwhite_zscore['pH_z'] >= 15) | (df_redwhite_zscore['sulphates_z'] <= -15) | (df_redwhite_zscore['sulphates_z'] >= 15) | (df_redwhite_zscore['alcohol_z'] <= -15) | (df_redwhite_zscore['alcohol_z'] >= 15)]
df_redwhite_15zscore_outliers.shape
# 2 Outliers with a z score (+/-) 15





# SPLIT CONTINIOUS VARIABLES
df_redwhite_continuous = df_redwhite[["fixed acidity", "volatile acidity", "citric acid",	"residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates" , "alcohol"]]

# LOG10 TRANSFORMATION
df_log10 = np.log10((df_redwhite_continuous))


# Correlation heatmap 
plt.figure(figsize=(16, 6))
mask = np.triu(np.ones_like(df_redwhite_continuous[["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates" , "alcohol"]].corr(), dtype=np.bool))
heatmap = sns.heatmap(df_redwhite_continuous[["fixed acidity", "volatile acidity", "citric acid",	"residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates" , "alcohol"]].corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=16)

# Individual correlation heatmaps
def heatmap_cor(variable_name):
    plt.figure(figsize=(8, 12))
    heatmap = sns.heatmap(df_redwhite_continuous.corr()[[variable_name]].sort_values(by=variable_name, ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
    heatmap.set_title(variable_name + " correlation", fontdict={'fontsize':18}, pad=16)
    
heatmap_cor('fixed acidity')
heatmap_cor('volatile acidity')
heatmap_cor('citric acid')
heatmap_cor('residual sugar')
heatmap_cor('chlorides')
heatmap_cor('free sulfur dioxide')
heatmap_cor('total sulfur dioxide')
heatmap_cor('density')
heatmap_cor('pH')
heatmap_cor('sulphates')
heatmap_cor('alcohol')


### Box Plots
def box_plot(variable):
    alcohol_box = px.box(df_redwhite,
                                x = "wine_type",
                                y = variable,
                                title= variable + " " + "Box Plot",
                                color="wine_type"
                            )

    alcohol_box.show()

box_plot("fixed acidity")
box_plot("volatile acidity")
box_plot("citric acid")
box_plot("residual sugar")
box_plot("chlorides")
box_plot("free sulfur dioxide")
box_plot("density")
box_plot("pH")
box_plot("sulphates")
box_plot("alcohol")
box_plot("quality")

# Histograms
def histo_plot(variable):
    histo = px.histogram(df_redwhite,
                                    x = variable,
                                    title = variable + " " + "Histogram",
                                    color ="wine_type",
                        )
    histo.show()

histo_plot("fixed acidity")
histo_plot("volatile acidity")
histo_plot("citric acid")
histo_plot("residual sugar")
histo_plot("chlorides")
histo_plot("free sulfur dioxide")
histo_plot("density")
histo_plot("pH")
histo_plot("sulphates")
histo_plot("alcohol")
histo_plot("quality")

# Scatterplot Matrix

scater_fig = px.scatter_matrix(df_redwhite,
                                    dimensions=["fixed acidity", "volatile acidity", "citric acid",	"residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates" , "alcohol", "quality"],
                                    color = "wine_type"
                            )


scater_fig.update_layout(
    title='Correlation Matrix',
    dragmode='select',
    width = 2000,
    height = 2000,
    hovermode='closest'
)

scater_fig.show()


# Radar Chart
df_log10 = df_redwhite[["fixed acidity", "volatile acidity", "citric acid",	"residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates" , "alcohol"]]

df_log10 = np.log10((df_log10))

df_log10["wine_type"] = df_redwhite['wine_type']
 
# ------- PART 1: Create background
 
# number of variable
categories=list(df_log10)[1:]
N = len(categories)
 
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
 
# Initialise the spider plot
ax = plt.subplot(111, polar=True)
 
# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
 
# Draw one axe per variable + add labels
plt.xticks(angles[:-1], categories)
 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([-2,0,3], ["-2","0","3"], color="grey", size=1)
plt.ylim(-2,3)
 

# ------- PART 2: Add plots
 
# Plot each individual = each line of the data
# I don't make a loop, because plotting more than 3 groups makes the chart unreadable
 
# Ind1
values=df_log10.loc[0].drop('wine_type').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="group A")
ax.fill(angles, values, 'b', alpha=0.1)
 
# Ind2
values=df_log10.loc[1].drop('wine_type').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="group B")
ax.fill(angles, values, 'r', alpha=0.1)
 
# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

# Show the graph
plt.show()
