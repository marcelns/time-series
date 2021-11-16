#!/usr/bin/env python
# coding: utf-8

# ## Intro to Time Series
# 
# Forecasting is the process of making predictions based on past and present data. Good forecasts capture the genuine patterns and relationships which exist in the historical data, but do not replicate past events that will not occur again.
# 
# There is a difference between a **random fluctuation** and a **genuine pattern** that should be modelled and extrapolated.
# 
# #### Time series graphics
# The first thing we shoud do in quantitative forecasting is understand the data. This can be done with exploration analysis.
# We should look for:
# 1. Patterns
# 2. Unusual observations
# 2. Changes over time
# 4. Relationships between variables.
# 
# #### Time Series Patterns
# 
# 1. Trend: Long term increase or decrease, does not have to be linear.
# 2. Seasonal: The series is affected by a seasonal pattern, like fridays on beer consumption.
# 3. Cyclical: Rises and falls on a that are not of a fixed frequency.
# 
# Patterns help us build good models.

# In[1]:


# libraries needes for case studies
import pandas as pd # Pandas is the python library for working with and visualizing time series
import numpy as np # Numpy is a library for matricial operations and high-level mathematical functions
import matplotlib.pyplot as plt # Matplot lib is the basic python graphic library
import seaborn as sns # Seaborn a graphic library focused on pandas dataframe and based on matplotlib
import plotly.express as px # Plotly is graphic library to make interact plots using java
import scipy.stats as stats # scipy statistical module, it has distributions and relevant stats functions
import statsmodels.api as sm # statistical models library, it has good models implementation, like logit and OLS
from statsmodels.tsa import deterministic as dt # module to create deterministic time trend/seasonal patterns. 
import pyreadr # for reading different data types, including r data

# Global parameters for plt graphics
plt.rcParams['font.size'] = 15
plt.rcParams['figure.figsize'] = [10,6]


# ### Time series regression models
# Time series can be thought as a list of numbers indexed by time.
# 
# The basic idea of regression is that we try to **explain** a time series $y$ assuming a linear relationship with another time series $x$.
# 
# For example we might try to forecast the GDP $y$ using employment $X_1$ and interest rate $X_2$

# In[4]:


# artificial example for linear regression
x = stats.uniform.rvs(0,100,size = 100) # x is sample from a uniform distribuiton
error = stats.norm.rvs(0,5, size = 100) # error has a normal distribution in our case
b0 = 10
b1 = 0.5
# generating predictable variable
y = b0 + b1*x + error

## Crating Ordinary Least Squares (OLS) model
X = sm.add_constant(x)
model = sm.OLS(y,X)
result_model = model.fit()
# predicting results
y_pred = result_model.predict(X)

# Transforming it all on da Data Frame
generated_data = pd.DataFrame(np.array([x,y,y_pred]).T, columns = ['x','y','y_pred'])

# Ploting the regression
sns.scatterplot(data = generated_data, x = 'x', y = 'y', label = 'Generated Data')
sns.lineplot(data = generated_data, x = 'x', y = 'y_pred',color = 'r', label = 'Regression Estimation')
plt.title('Simple regression with generated data')
plt.grid();

# Comparing the true values ans their estimations
print(f'True Intercept: {b0}\t True Linear Coef: {b1}')
b0_est, b1_est = result_model.params
print(f'Est. Intercept: {b0_est:.2f}\t Est. Linear Coef: {b1_est:.2f}')


# ### Example: US consumption expenditure
# 
# The plot below shows time series of quarterly percentage changes (growth rates) of real personal consumption expenditure, $y$, and real personal disposable income, $x$, for the US from 1970 Q1 to 2016 Q3.

# In[5]:


# Reading Data
rdata = pyreadr.read_r(r"data/uschange.rda")
uschange_data = rdata['uschange']

# generating time index and indexing data
quarters = pd.date_range('1970', periods=len(uschange_data), freq = 'Q')
uschange_data.index = quarters
print(uschange_data.head(6))
px.line(uschange_data,x = uschange_data.index, y =  ['Consumption'], 
        title = 'US real personal consumption expenditure and real personal disposable income changes')


# In[6]:


X = sm.add_constant(uschange_data.loc[:,['Income']].to_numpy())
model = sm.OLS(uschange_data['Consumption'].to_numpy(),X)
result_model = model.fit()
y_pred = result_model.predict(X)

sns.lineplot(data = uschange_data, x = 'Income', y = y_pred, color = 'r')
sns.scatterplot(data = uschange_data, x = 'Income', y = 'Consumption')
plt.title('Income vs Consumption');


# ### Least squares estimation
# 
# This is a usual method in statistics, we try to explain the target variable by linking it with another set of variables using linear parameters. The covariables, the variables $x_{i,t}$ that we use to adjust the model, can have a non-linear relation with the target variable $y_t$ but the parameters have to be linear. This property helps us in building and finding solution for the model, that can be done using linear algebra. The geneal model can be written as:
# 
# $$y_t = \beta_0 + \beta_1x_{1,t} + \dots + \beta_kx_{k,t} + \epsilon_t$$
# 
# Rearranging the expression, we can calculate the error in each prediction:
# 
# $$\epsilon_t = y_t - (\beta_0 + \beta_1x_{1,t} + \dots + \beta_kx_{k,t})$$
# 
# But we also can define the squared sum of errors from prediction with given a set of $\beta$ 's:
# 
# $$\Sigma_{i=0}^{t} \epsilon_t^2 = \Sigma_{i=0}^{t} (y_t - \beta_0 + \beta_1x_{1,t} + \dots + \beta_kx_{k,t})^2$$
# 
# A set of $\beta$ 's can be estimated by minimizing the square of estimation error. We will use the $\hat{\beta_0},\dots,\hat{\beta_k}$ notation for the estimated parameters.
# 
# ### Adjusting the model
# In the US consumption expenditure we have the following model:
# $$y_t = \beta_0 + \beta_1x_{1,t} + \beta_2x_{2,t} + \beta_3x_{3,t} + \beta_4x_{4,t} + \epsilon_t$$
# 
# Where:
# 1. $y_t$:     Comsumption, the percentage change in real personal consumption expenditure
# 2. $x_{1,t}$: Income, the percentage change in real personal disposable income  
# 3. $x_{2,t}$: Production, the percentage change in industrial production   
# 4. $x_{3,t}$: Savings, the percentage change in personal savings
# 5. $x_{4,t}$: Unemployment, the change in the unemployment rate

# In[7]:


# Adjusting the model
y = uschange_data.iloc[:,0].to_numpy()
X = uschange_data.loc[:,['Income', 'Production', 'Savings', 'Unemployment']].to_numpy()
X = sm.add_constant(X)

model = sm.OLS(endog = y, exog = X)
result_model = model.fit()
# print(result_model.summary())


# In[8]:


# Fit results plot
comparative_data = pd.DataFrame(np.array([y,y_pred]).T, columns = ['Consumption', 'Prediction'], index = quarters)

x_line = np.linspace(comparative_data['Consumption'].max(),comparative_data['Consumption'].min(), 10)
sns.scatterplot(data = comparative_data,x = 'Consumption',y = 'Prediction', label = 'Actual Fit')
plt.plot(x_line,x_line, color = 'red', label = 'Theoretically Perfect Fit')
plt.title('Fit result')
plt.legend();
px.line(comparative_data,x = uschange_data.index, y =  ['Consumption', 'Prediction'], 
        title = 'US real personal consumption expenditure prediction')


# In[9]:


# Model Interpretation
print(result_model.summary())

y_pred = result_model.predict(X)


# There is a lot of information on a regression summary. 
# 
# First things first, **does the model predict something**?
# 
# We can calculate the proportion of $y$ that is explained by the combination of variables and the proportion that is resildual.
# $$F = \frac{MS_{regression}}{MS_{residuals}}$$
# 
# MS stands for Mean Squares. The higher the F statistic more of the target variable is explained by the model. We estimate the probability of F being equals zero, the lower the prob more certainty we can have that our model can predict something.
# With Prob (F-statistic) virtually equals 0, our model can actually predict something.
# 
# Second, how **good is our model**?
# 
# We can see it by the proportion of the 
# $$R^2 = \frac{SS_{regression}}{SS_{Total}} = \frac{\Sigma(\hat{y_i} - \bar{y})^2}{\Sigma(y_i - \bar{y})^2}$$
# 
# SS stands for Sum of Squares. The proportion of squared deviation from the mean of our estimation and the real proportionThis values is always positive and between 0 and 1. I f we add new variables, the $R^2$ will always rise, but we lose precision on predition, so we need to 'punish' additions that don't aggregate to the model using the **adjusted** $R^2$. Our model generates the adjusted $R^2 = 0.75$, with is a high value, so we are quite good in predicting the Consumption.
# 
# But can we predict the model using less information? Using less variables?
# We acan analyse each coefficient sepparetely: In effect, each coef is a estimation, with each observation we can estimate differente coefs, theoretcally if we do a lot of estimations the estimated value will have some variability, calculated by the standart error of each coefficient. The proportion of the estimated value and its variability is the t statistic. If this proportion is equals to zero than the coef can actually be irrelevant, the probability of t being 0 is the P>|t|. Production and Unemployment can be irrelevant in our estimation.
# 
# Third, **how does my model make mistakes?**
# 
# What I mean with this question is that this is data science, nothing is perfect, we make models for educated guests and those guests have a probability of being wrong. The problem is: is my error acceptable, is the model making mistakes *systematically*.
# 
# Those errors I am referring to are called residuals in statistics. One measure of how well the model has fitted the data is the standard deviation of the residuals, which is often known as the “residual standard error.”
# $$\hat{\sigma}_e = \sqrt{\frac{1}{n - k - 1} \Sigma e^2_t}$$
# 
# Where $n$ is the numbers of observations and $k$ the number of predictors in the model.
# 
# The standard error is related to the size of the average error that the model produces. We can compare this error to the sample mean of y or with the standard deviation of y to gain some perspective on the accuracy of the model.
# 
# The residual analysis and the analysis on the model assumptions will be in the section below.

# ### Evaluating the model
# After selecting the regression variables and fitting a regression model, it is necessary to plot the residuals to check that the *assumptions* of the model have been satisfied. The assumptions of the model are very important. For a mathematical model to be correct we have to guarantee that the assumptions are correct. The assumptions are the bulding blocks for modelling. If we those blocks lack integridy qe can't guarantee that the building will stand up.
# 
# There are a series of plots that should be produced in order to check different aspects of the fitted model and the underlying assumptions.

# In[11]:


## analysing rediduals
resid = result_model.resid
stderror_resid = ((resid**2).sum() / (len(resid) - X.shape[1] - 1))**(1/2)

## distribution of residuals in time
plt.plot(quarters,resid, marker = 'o')
plt.title('Distribution of residuals (mistakes on prediction) on time')
plt.ylabel('Residuals')
plt.show()
print(f'One interesting property of the residuals is that they sum is equals 0')
print(f'Residuals Sum: {resid.sum():.8f}')

# ## distribution of residuals
plt.hist(resid);


# One important concept in analysing residuals is **Heteroscedasticity** that happens when the variability of the disturbance is different across the elements of a vactor. We measure this variability using the variance. In other words, when the **residual variance is not constant across the time** we may have heteroscedasticity. There is a number of techniques used for dealing with this particular problem, one of them is the transformation of the forecast variable. In this particular model, the residuals don't have different variance and are **Homoscedastic**.
# 
# The time plot shows some changing variation over time, but is otherwise relatively unremarkable. This heteroscedasticity will potentially make the prediction interval coverage inaccurate.
# 
# The histogram shows that the residuals seem to be slightly skewed, which may also affect the coverage probability of the prediction intervals.

# #### Autocorrelation
# 
# With time series data, it is highly likely that the value of a variable observed in the current time period will be similar to its value in the previous period, or even the period before that, and so on. Therefore when fitting a regression model to time series data, it is common to find autocorrelation in the residuals. 
# 
# So, if the true *consumption* value in $t-1$ influences the value in $t$ and our model don't account that, the error we made in a previous observation is corralated with a error in the next observation, so we are sistematically making prediction mistakes.
# 
# In this case, the estimated model violates the assumption of no autocorrelation in the errors, and our forecasts may be inefficient — there is some information left over which should be accounted for in the model in order to obtain better forecasts. The forecasts from a model with autocorrelated errors are still unbiased, and so are not “wrong,” but they will usually have larger prediction intervals than they need to because the standart errors are effected.
# 
# In autocorrelation case we can see its effect in a **Auto Correlation Function** plot, or **ACF plot**. 

# In[13]:


## ACF plot
sm.graphics.tsa.plot_acf(resid, lags=range(1,11));


# The ACF plot presents autocorrelation coefficients and the confidence intervals (blue area). If the autocorrelation coefficient is in the confidence interval, it is regarded as not statistically significant. In this case, the lag 7 correlation is statistically significant. It is to say that the value observed 7 quarters ago is related to the present value.
# 
# The null Hypothesis is that there is no autocorrelation for any lag tested
# 
# $$H_0 = \{\rho_i = 0 \text{ for all } i\}$$
# 

# In[14]:


lagrange_stat, ls_pval, fval, f_pval = sm.stats.diagnostic.acorr_breusch_godfrey(result_model, nlags = 8)

print(f'Test p-value: {ls_pval:.2f}')


# The autocorrelation plot shows a significant spike at lag 7, but it is not quite enough for the Breusch-Godfrey to be significant at the 5% level. In any case, the autocorrelation is not particularly large, and at lag 7 it is unlikely to have any noticeable impact on the forecasts or the prediction intervals.

# ### Residual Plot Against Predictors
# If we plot residuals and predictors we should expect the residuals to be reamdonly sccatered without showing any sistematic patterns. If the residuals show some pattern maybe the relationship of that predictor and the predict varible may be non-linear and the model will need to be modified.

# In[15]:


# organizing a dataframe
predict_data = uschange_data.iloc[:,1:]
predict_data['ConsumptionPred'] = result_model.predict(X)
predict_data['Residuals'] = result_model.resid
plt.figure(figsize = [15,9])
plt.subplot(2,2,1)
sns.scatterplot(data = predict_data, x = 'Income', y = 'Residuals')
plt.subplot(2,2,2)
sns.scatterplot(data = predict_data, x = 'Production', y = 'Residuals')
plt.ylabel(' ')
plt.subplot(2,2,3)
sns.scatterplot(data = predict_data, x = 'Savings', y = 'Residuals')
plt.subplot(2,2,4)
sns.scatterplot(data = predict_data, x = 'Unemployment', y = 'Residuals')
plt.ylabel(' ')
plt.show()
print('Scatterplots of residuals versus each predictor')


# ### Outliers and influential Observations
# 
# Observations that take extreme values compared to the majority of the data are called outliers, they assume a extremely high/low value in the $y$ direction. 
# 
# Observations that have a large influence on the estimated coefficients of a regression model are called influential observations. Usually, influential observations are also outliers that are extreme in the $x$ direction.
# 
# Some measures like the **mean** are sensible to outliers and this makes a lot of models sensible to outliers to. For example, the the simple regression model, the intercept is defined by $B_0 = \bar{Y} - \beta_1\bar{X}$, so the model fit can change significantly with outliers and influential observations.

# ## Some Useful Predictors
# #### Trend
# It is common for time series data to be trending. A linear trend can be modelled by simply using $x_{1,t} = t$  **time as a predictor**.
# 
# $$Y_t = \beta_0 + \beta_1t + \epsilon_t$$
# 
# #### Dummy
# We can use a **categorical variable taking only 2 values *yes* and *no***. Imagine that we are trying to predict *icecream* sales. We can create a dummy variable which takes value 1 corresponding to summer and spring and and 0 corresponding to fall and winter and this situation can still be handled within the framework of multiple regression models.
# 
# A dummy variable can also be **used to account for an outlier** in the data. Rather than omit the outlier, a dummy variable removes its effect. In this case, the dummy variable takes value 1 for that observation and 0 everywhere else. 
# 
# The general rule is to use one **fewer dummy variables than categories**. So for quarterly data, use three dummy variables; for monthly data, use 11 dummy variables; and for daily data, use six dummy variables, and so on. The interpretation of each of the coefficients associated with the dummy variables is that it is a **measure of the effect of that category relative to the omitted category**.

# ### Example: Australian quarterly beer production
# 
# Recall the Australian quarterly beer production data

# In[3]:


# Reading Data
beer_data = pd.read_csv(r'data/ausbeer.csv')

## generating time index and indexing data
quarters = pd.date_range('1992', periods=len(beer_data), freq = 'Q')
beer_data.index = quarters
print(beer_data.head(6))
px.line(beer_data,x = beer_data.index, y =  ['megaliters'], 
        title = 'Australian beer production')


# #### Adding a Trend and Dummies
# We want to forecast the value of future beer production. We can model this data using a regression model with a linear trend and quarterly dummy variables:
# 
# $$y_t = \beta_0 + \beta_1t + \beta_2D_{2,t} + \beta_3D_{3,t} + \beta_4D_{4,t} + \epsilon_t$$
# 
# where $D_{i,t} = 1$ if $t$  is in quarter $i$ and $0$ otherwise. The first quarter variable has been omitted, so the coefficients associated with the other quarters are measures of the difference between those quarters and the first quarter.

# In[8]:


## Creating a Trend time series
# create a Object that generate trends
trend_generator = dt.TimeTrend(constant = True, order = 1)
# create a trend in sample means 'in data range'
# if we want to forecast we can usod .out_of_sample method
trend = trend_generator.in_sample(beer_data.index)

# creat a object that generates dummies for seasons
season_generator = dt.Seasonality(period = 4,initial_period=1)
season = season_generator.in_sample(beer_data.index).iloc[:,1:]

trend_season = pd.concat([trend,season], axis = 1)
print(trend_season.head())

## fiting the model
model = sm.OLS(endog = beer_data, exog = trend_season)
model = model.fit()
print(model.summary())

## predicting
beer_data['prediction'] = model.predict(trend_season)

px.line(beer_data,x = beer_data.index, y =  ['megaliters','prediction'])


# In[37]:


# generating a quarters list for a hue
quarters_list = [1,2,3,4]*19
quarters_list = quarters_list[:-2]
beer_data['quarters'] = quarters_list

perfect_fit = np.arange(min(beer_data['megaliters']),max(beer_data['megaliters']))
sns.scatterplot(data = beer_data, x = 'megaliters', y = 'prediction', hue = 'quarters', 
                label = 'Actual Fit', palette = "deep")
plt.plot(perfect_fit,perfect_fit,color = 'red', label = 'Theoretically Perfect Fit')
plt.legend();


# There is an average downward trend of -0.34 megalitres per quarter. On average, the second quarter has production of 34.7 megalitres lower than the first quarter, the third quarter has production of 17.8 megalitres lower than the first quarter, and the fourth quarter has production of 72.8 megalitres higher than the first quarter.

# In[43]:


# Residual analysis
resid = model.resid
plt.scatter(beer_data.index,resid);
sm.graphics.tsa.plot_acf(resid, lags=range(1,11));
lagrange_stat, ls_pval, fval, f_pval = sm.stats.diagnostic.acorr_breusch_godfrey(model, nlags = 8)
print(f'Test p-value: {ls_pval:.2f}, Null hyphothesis (no autocorrelation) cannot be rejected')

