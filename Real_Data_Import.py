# You can use the FRED (Federal Reserve Economic Data) API to access a wide range of US economic indicators.
# To get started, you'll need to install the fredapi and pandas packages, and obtain an API key from FRED.
# pip install fredapi pandas
# https://fred.stlouisfed.org/docs/api/api_key.html to request an API key



####################################
######## Script to import data #####
####################################


import pandas as pd
from fredapi import Fred

# Replace with your own API key obtained from FRED
api_key = '1a3ba5a60feccf8ab6e2ab7dcb2671ec'
fred = Fred(api_key=api_key)

# List of economic economic indicators
economic_indicators = {
    'GDP': 'GDPC1',
    'CPI': 'CPALTT01USM657N',
    'Unemployment Rate': 'UNRATE',
    'Federal Funds Rate': 'FEDFUNDS',
    '10-Year Treasury Constant Maturity Rate': 'GS10',
    'Personal Consumption Expenditures': 'PCE',
    'Real Disposable Personal Income': 'DSPIC96',
    'Industrial Production Index': 'INDPRO',
    'Consumer Sentiment Index': 'UMCSENT',
    'Nonfarm Payrolls': 'PAYEMS',
    'Housing Starts': 'HOUST'
}

# Function to fetch economic indicator data
def get_economic_indicator_data(series_id, start_date=None, end_date=None):
    data = fred.get_series(series_id, start_date, end_date)
    return data

# Store indicator data in arrays
indicator_data_arrays = {}

for indicator, series_id in economic_indicators.items():
    indicator_data_arrays[indicator] = get_economic_indicator_data(series_id)

# Print the fetched data
for indicator, data in indicator_data_arrays.items():
    print(f'{indicator} Data:')
    print(data)
    print()

#####################################################
#### Plot of economic indicators using real data ####
#####################################################
import pandas as pd
import matplotlib.pyplot as plt
from fredapi import Fred
import numpy as np

# Replace with your own API key obtained from FRED
api_key = '1a3ba5a60feccf8ab6e2ab7dcb2671ec'
fred = Fred(api_key=api_key)

# List of economic economic indicators with units
economic_indicators = {
    'GDP': {'series_id': 'GDPC1', 'unit': 'Billions of Chained 2012 Dollars'},
    'CPI': {'series_id': 'CPALTT01USM657N', 'unit': 'Index'},
    'Unemployment Rate': {'series_id': 'UNRATE', 'unit': 'Percent'},
    'Federal Funds Rate': {'series_id': 'FEDFUNDS', 'unit': 'Percent'},
    '10-Year Treasury Constant Maturity Rate': {'series_id': 'GS10', 'unit': 'Percent'},
    'Personal Consumption Expenditures': {'series_id': 'PCE', 'unit': 'Billions of Dollars'},
    'Real Disposable Personal Income': {'series_id': 'DSPIC96', 'unit': 'Billions of Chained 2012 Dollars'},
    'Industrial Production Index': {'series_id': 'INDPRO', 'unit': 'Index'},
    'Consumer Sentiment Index': {'series_id': 'UMCSENT', 'unit': 'Index'},
    'Nonfarm Payrolls': {'series_id': 'PAYEMS', 'unit': 'Thousands of Persons'},
    'Housing Starts': {'series_id': 'HOUST', 'unit': 'Thousands of Units'}
}

# Function to fetch economic indicator data
def get_economic_indicator_data(series_id, start_date=None, end_date=None):
    data = fred.get_series(series_id, start_date, end_date)
    return data

# Store indicator data in arrays
indicator_data_arrays = {}

for indicator, series_info in economic_indicators.items():
    indicator_data_arrays[indicator] = get_economic_indicator_data(series_info['series_id'])

# Calculate the number of indicators
n_indicators = len(economic_indicators)

# Set up plot colors
colors = plt.cm.viridis(np.linspace(0, 1, n_indicators))

# Create a grid of plots for the economic indicators
n_cols = 3
n_rows = (n_indicators + n_cols - 1) // n_cols
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, n_rows * 5))

for i, (indicator, data) in enumerate(indicator_data_arrays.items()):
    row, col = divmod(i, n_cols)
    ax = axes[row, col]
    data.plot(ax=ax, color=colors[i])
    ax.set_title(indicator)
    ax.set_xlabel('Year')
    ax.set_ylabel(f"{indicator} ({economic_indicators[indicator]['unit']})")
    ax.tick_params(axis='both', labelsize=12)

# Remove empty subplots
if n_rows * n_cols > n_indicators:
    for i in range(n_indicators, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        fig.delaxes(axes[row, col])

# Adjust the layout and show the plot
plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.tight_layout()
plt.show()

#####################################################
######## Calculated versus real data for GDP ########
#####################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred

# Replace with your own API key obtained from FRED
api_key = '1a3ba5a60feccf8ab6e2ab7dcb2671ec'
fred = Fred(api_key=api_key)

# List of economic economic indicators
economic_indicators = {
    'Real GDP': 'GDPC1',
    'GDP Growth Rate': 'A191RL1Q225SBEA'
}

# Function to fetch economic indicator data
def get_economic_indicator_data(series_id, start_date=None, end_date=None):
    data = fred.get_series(series_id, start_date, end_date)
    return data

# Store indicator data in arrays
indicator_data_arrays = {}

for indicator, series_id in economic_indicators.items():
    indicator_data_arrays[indicator] = get_economic_indicator_data(series_id)

# Calculate GDP growth rate from Real GDP data
gdp = indicator_data_arrays['Real GDP']
gdp_growth_rate_calculated = gdp.pct_change() * 100

# Create the plot comparing the FRED GDP Growth Rate data and the calculated GDP growth rate
fig, ax = plt.subplots(figsize=(12, 6))

indicator_data_arrays['GDP Growth Rate'].plot(ax=ax, label='FRED Data')
gdp_growth_rate_calculated.plot(ax=ax, label='Calculated', linestyle='--')

ax.set_title('GDP Growth Rate')
ax.set_xlabel('Year')
ax.set_ylabel('GDP Growth Rate (%)')
ax.tick_params(axis='both', labelsize=12)
ax.legend()

plt.tight_layout()
plt.show()

##################################
##### NFP data from 3 sources ####
##################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred

# Replace with your own API key obtained from FRED
api_key = '1a3ba5a60feccf8ab6e2ab7dcb2671ec'
fred = Fred(api_key=api_key)

# List of nonfarm payrolls data sources
nonfarm_payrolls_sources = {
    'BLS (FRED)': 'PAYEMS',
    'ADP (FRED)': 'DSPIC96',
    'Household Survey (FRED)': 'CE16OV'
}

# Function to fetch nonfarm payrolls data
def get_nonfarm_payrolls_data(series_id, start_date=None, end_date=None):
    data = fred.get_series(series_id, start_date, end_date)
    return data

# Store nonfarm payrolls data in arrays
nonfarm_payrolls_data_arrays = {}

for source, series_id in nonfarm_payrolls_sources.items():
    nonfarm_payrolls_data_arrays[source] = get_nonfarm_payrolls_data(series_id)

# Create the plot comparing nonfarm payrolls data from different sources
fig, ax = plt.subplots(figsize=(12, 6))

for source, data in nonfarm_payrolls_data_arrays.items():
    data.plot(ax=ax, label=source)

ax.set_title('Nonfarm Payrolls Comparison')
ax.set_xlabel('Year')
ax.set_ylabel('Nonfarm Payrolls')
ax.tick_params(axis='both', labelsize=12)
ax.legend()

plt.tight_layout()
plt.show()

###########################################################################################################
######## Correlation between monthly changes in nonfarm parolls data and SandP 500's monthly returns ######
###########################################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred

# Replace with your own API key obtained from FRED
api_key = '1a3ba5a60feccf8ab6e2ab7dcb2671ec'
fred = Fred(api_key=api_key)

# List of nonfarm payrolls data sources
nonfarm_payrolls_sources = {
    'BLS (FRED)': 'PAYEMS',
    'ADP (FRED)': 'DSPIC96',
    'Household Survey (FRED)': 'CE16OV'
}

# Function to fetch nonfarm payrolls data
def get_data(series_id, start_date=None, end_date=None):
    data = fred.get_series(series_id, start_date, end_date)
    return data

# Store nonfarm payrolls data in arrays
nonfarm_payrolls_data_arrays = {}

for source, series_id in nonfarm_payrolls_sources.items():
    nonfarm_payrolls_data_arrays[source] = get_data(series_id)

# Calculate the monthly changes in nonfarm payrolls
monthly_changes = {}
for source, data in nonfarm_payrolls_data_arrays.items():
    monthly_changes[source] = data.diff()

# Get S&P 500 data and calculate the monthly returns
sp500_data = get_data('SP500')
sp500_monthly_returns = sp500_data.pct_change()

# Calculate correlations between nonfarm payrolls monthly changes and S&P 500 monthly returns
correlations = {}
for source, data in monthly_changes.items():
    correlations[source] = data.corr(sp500_monthly_returns)

# Print the correlations
print("Correlations between nonfarm payrolls monthly changes and S&P 500 monthly returns:")
for source, correlation in correlations.items():
    print(f"{source}: {correlation}")

# Visualize the nonfarm payrolls monthly changes and S&P 500 monthly returns
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))

for source, data in monthly_changes.items():
    data.plot(ax=axes[0], label=source)

axes[0].set_title('Monthly Changes in Nonfarm Payrolls')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Change in Nonfarm Payrolls')
axes[0].tick_params(axis='both', labelsize=12)
axes[0].legend()

sp500_monthly_returns.plot(ax=axes[1], label='S&P 500')
axes[1].set_title('S&P 500 Monthly Returns')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Monthly Return')
axes[1].tick_params(axis='both', labelsize=12)
axes[1].legend()

plt.tight_layout()
plt.show()


#####################################################
###### Rolling correlation and risk management ######
#####################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred

# Replace with your own API key obtained from FRED
api_key = '1a3ba5a60feccf8ab6e2ab7dcb2671ec'
fred = Fred(api_key=api_key)

# List of nonfarm payrolls data sources
nonfarm_payrolls_sources = {
    'BLS (FRED)': 'PAYEMS',
    'ADP (FRED)': 'DSPIC96',
    'Household Survey (FRED)': 'CE16OV'
}

# Function to fetch nonfarm payrolls data
def get_data(series_id, start_date=None, end_date=None):
    data = fred.get_series(series_id, start_date, end_date)
    return data

# Store nonfarm payrolls data in arrays
nonfarm_payrolls_data_arrays = {}

for source, series_id in nonfarm_payrolls_sources.items():
    nonfarm_payrolls_data_arrays[source] = get_data(series_id)

# Calculate the monthly changes in nonfarm payrolls
monthly_changes = {}
for source, data in nonfarm_payrolls_data_arrays.items():
    monthly_changes[source] = data.diff()

# Get S&P 500 data and calculate the monthly returns
sp500_data = get_data('SP500')
sp500_monthly_returns = sp500_data.pct_change()

# Combine all data into a single DataFrame
data_combined = pd.concat([sp500_monthly_returns] + list(monthly_changes.values()), axis=1)
data_combined.columns = ['S&P 500 Returns'] + list(monthly_changes.keys())

# Drop rows with missing values
data_combined.dropna(inplace=True)

# Calculate rolling correlations between nonfarm payrolls monthly changes and S&P 500 monthly returns
rolling_correlations = {}
window = 12
for source in monthly_changes.keys():
    rolling_correlations[source] = data_combined[source].rolling(window=window).corr(data_combined['S&P 500 Returns'])

# Visualize the rolling correlations
fig, ax = plt.subplots(figsize=(12, 6))

for source, data in rolling_correlations.items():
    data.plot(ax=ax, label=source)

ax.set_title(f'Rolling {window}-Month Correlations: Nonfarm Payrolls Monthly Changes and S&P 500 Monthly Returns')
ax.set_xlabel('Year')
ax.set_ylabel('Correlation')
ax.tick_params(axis='both', labelsize=12)
ax.legend()

plt.tight_layout()
plt.show()

###############################################################
##### Market movements using 5 economic indicators ############
###############################################################
###### Correesponds to point 1, see supporting document #######
###############################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred

# Replace with your own API key obtained from FRED
api_key = '1a3ba5a60feccf8ab6e2ab7dcb2671ec'
fred = Fred(api_key=api_key)

# Economic indicators
economic_indicators = {
    'Nonfarm Payrolls': 'PAYEMS',
    'Unemployment Rate': 'UNRATE',
    'Consumer Price Index': 'CPIAUCSL',
    'GDP': 'GDP',
    'Federal Funds Rate': 'FEDFUNDS'
}

# Function to fetch data
def get_data(series_id, start_date=None, end_date=None):
    data = fred.get_series(series_id, start_date, end_date)
    return data

# Store data in arrays
economic_data_arrays = {}

for indicator, series_id in economic_indicators.items():
    economic_data_arrays[indicator] = get_data(series_id)

# Calculate the monthly changes for each indicator
monthly_changes = {}
for indicator, data in economic_data_arrays.items():
    monthly_changes[indicator] = data.diff()

# Get S&P 500 data and calculate the monthly returns
sp500_data = get_data('SP500')
sp500_monthly_returns = sp500_data.pct_change()

# Combine all data into a single DataFrame
data_combined = pd.concat([sp500_monthly_returns] + list(monthly_changes.values()), axis=1)
data_combined.columns = ['S&P 500 Returns'] + list(monthly_changes.keys())

# Drop rows with missing values
data_combined.dropna(inplace=True)

# Calculate rolling correlations between the economic indicators' monthly changes and S&P 500 monthly returns
rolling_correlations = {}
window = 12
for indicator in monthly_changes.keys():
    rolling_correlations[indicator] = data_combined[indicator].rolling(window=window).corr(data_combined['S&P 500 Returns'])

# Visualize the rolling correlations
fig, ax = plt.subplots(figsize=(12, 6))

for indicator, data in rolling_correlations.items():
    data.plot(ax=ax, label=indicator)

ax.set_title(f'Rolling {window}-Month Correlations: Economic Indicators and S&P 500 Monthly Returns')
ax.set_xlabel('Year')
ax.set_ylabel('Correlation')
ax.tick_params(axis='both', labelsize=12)
ax.legend()

plt.tight_layout()
plt.show()

###############################################################
######## Market movements in tech sector using NASDAQ #########
###############################################################
######## Only data from yfinance is used ######################
######## to get yfinance, pip install yfinance and enter ######
###### Correesponds to point 2, see supporting document #######
###############################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Selected stocks
stock_symbols = {
    'Walmart': 'WMT',
    'Procter & Gamble': 'PG',
    'Intel': 'INTC',
    'Microsoft': 'MSFT',
    'Johnson & Johnson': 'JNJ'
}

# Function to fetch data
def get_stock_data(ticker, start_date=None, end_date=None):
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)['Adj Close']
    return data

# Store data in arrays
stock_data_arrays = {}

for name, symbol in stock_symbols.items():
    stock_data_arrays[name] = get_stock_data(symbol)

# Calculate the monthly changes for each stock
monthly_changes = {}
for name, data in stock_data_arrays.items():
    monthly_data = data.resample('M').last()
    monthly_changes[name] = monthly_data.pct_change()

# Get NASDAQ 100 data from Yahoo Finance and calculate the monthly returns
ticker = "^NDX"
start_date = "2000-01-01"
end_date = None

nasdaq100_data = yf.download(ticker, start=start_date, end=end_date, progress=False)['Adj Close']
nasdaq100_monthly_data = nasdaq100_data.resample('M').last()
nasdaq100_monthly_returns = nasdaq100_monthly_data.pct_change()

# Combine all data into a single DataFrame
data_combined = pd.concat([nasdaq100_monthly_returns] + list(monthly_changes.values()), axis=1)
data_combined.columns = ['NASDAQ 100 Returns'] + list(monthly_changes.keys())

# Drop rows with missing values
data_combined.dropna(inplace=True)

# Calculate rolling correlations between the stocks' monthly changes and NASDAQ 100 monthly returns
rolling_correlations = {}
window = 12
for stock in monthly_changes.keys():
    rolling_correlations[stock] = data_combined[stock].rolling(window=window).corr(data_combined['NASDAQ 100 Returns'])

# Visualize the rolling correlations
fig, ax = plt.subplots(figsize=(12, 6))

for stock, data in rolling_correlations.items():
    data.plot(ax=ax, label=stock)

ax.set_title(f'Rolling {window}-Month Correlations: Selected Stocks and NASDAQ 100 Monthly Returns')
ax.set_xlabel('Year')
ax.set_ylabel('Correlation')
ax.tick_params(axis='both', labelsize=12)
ax.legend()

plt.tight_layout()
plt.show()


###############

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred
import yfinance as yf

# Replace with your own API key obtained from FRED
api_key = '1a3ba5a60feccf8ab6e2ab7dcb2671ec'
fred = Fred(api_key=api_key)

# Get Consumer Price Index data
cpi_data = fred.get_series('CPIAUCSL')
cpi_monthly = cpi_data.resample('M').last()
cpi_monthly_changes = cpi_monthly.pct_change()

# Get NASDAQ 100 data from Yahoo Finance and calculate the monthly returns
ticker = "^NDX"
start_date = "2000-01-01"
end_date = None

nasdaq100_data = yf.download(ticker, start=start_date, end=end_date, progress=False)['Adj Close']
nasdaq100_monthly_data = nasdaq100_data.resample('M').last()
nasdaq100_monthly_returns = nasdaq100_monthly_data.pct_change()

# Combine both data into a single DataFrame
data_combined = pd.concat([nasdaq100_monthly_returns, cpi_monthly_changes], axis=1)
data_combined.columns = ['NASDAQ 100 Returns', 'Consumer Price Index']

# Convert the index to the same format for all data
data_combined.index = data_combined.index.to_period('M')

# Drop rows with missing values
data_combined.dropna(inplace=True)

# Calculate correlations between CPI monthly changes and NASDAQ 100 monthly returns for various lags
lags = list(range(0, 13))
lag_correlations = {}

for lag in lags:
    shifted_cpi = data_combined['Consumer Price Index'].shift(lag)
    lag_correlations[lag] = shifted_cpi.corr(data_combined['NASDAQ 100 Returns'])

# Convert lag_correlations to a DataFrame
lag_correlations_df = pd.DataFrame(lag_correlations, index=['Consumer Price Index'])

# Visualize the lag correlations
fig, ax = plt.subplots(figsize=(12, 6))

lag_correlations_df.T.plot(kind='bar', ax=ax)
ax.set_title('Lag Analysis: Correlations between Consumer Price Index and NASDAQ 100 Monthly Returns')
ax.set_xlabel('Lag (Months)')
ax.set_ylabel('Correlation')
ax.tick_params(axis='both', labelsize=12)
ax.legend(title='Economic Indicators')

plt.tight_layout()
plt.show()


#############################################################
########### Inclusion of copula ############################
#############################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from fredapi import Fred

# Replace with your own API key obtained from FRED
api_key = '1a3ba5a60feccf8ab6e2ab7dcb2671ec'
fred = Fred(api_key=api_key)

def get_data(series_id, start_date=None, end_date=None):
    data = fred.get_series(series_id, start_date, end_date)
    return data

# Fetch GDP and Unemployment Rate data
gdp_data = get_data('GDP')
unemployment_rate_data = get_data('UNRATE')

# Combine data and drop missing values
data_combined = pd.concat([gdp_data, unemployment_rate_data], axis=1)
data_combined.columns = ['GDP', 'Unemployment Rate']
data_combined.dropna(inplace=True)

# Rank data and normalize
data_ranked = data_combined.rank()
data_normalized = data_ranked / len(data_combined)

# Calculate copula parameters
def fit_clayton_copula(u, v, tau):
    alpha = 2 * tau / (1 - tau)
    return alpha

def clayton_copula(u, v, alpha):
    copula_density = (u ** (-alpha) + v ** (-alpha) - 1) ** (-alpha - 2) * u ** (-alpha - 1) * v ** (-alpha - 1)
    return copula_density

# Calculate Kendall's Tau
tau = data_normalized.corr(method='kendall')['GDP']['Unemployment Rate']

# Fit Clayton copula
alpha = fit_clayton_copula(data_normalized['GDP'], data_normalized['Unemployment Rate'], tau)

# Generate samples from Clayton copula
u_samples = np.random.uniform(0, 1, size=1000)
v_samples = np.random.uniform(0, 1, size=1000)
copula_density_samples = clayton_copula(u_samples, v_samples, alpha)

# Visualize data
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

sc = ax.scatter(u_samples, v_samples, c=copula_density_samples, cmap='viridis')
plt.colorbar(sc, label='Copula Density')
ax.set_xlabel('GDP')
ax.set_ylabel('Unemployment Rate')
ax.set_title('Clayton Copula')

plt.tight_layout()
plt.show()

################################################
####### Gaussian copulas #######################
################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred
from copulas.multivariate import GaussianMultivariate

# Replace with your own API key obtained from FRED
api_key = '1a3ba5a60feccf8ab6e2ab7dcb2671ec'
fred = Fred(api_key=api_key)

# Fetch data for GDP and Unemployment Rate
gdp_data = fred.get_series('GDP')
unemployment_data = fred.get_series('UNRATE')

# Combine data into a single DataFrame and drop missing values
data = pd.concat([gdp_data, unemployment_data], axis=1).dropna()
data.columns = ['GDP', 'Unemployment Rate']

# Fit a Gaussian copula
copula = GaussianMultivariate()
copula.fit(data)

# Generate synthetic data from the fitted copula
synthetic_data = copula.sample(len(data))

# Visualize the original data and synthetic data side by side
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

axes[0].scatter(data['GDP'], data['Unemployment Rate'])
axes[0].set_title('Original Data')
axes[0].set_xlabel('GDP')
axes[0].set_ylabel('Unemployment Rate')

axes[1].scatter(synthetic_data['GDP'], synthetic_data['Unemployment Rate'], color='r')
axes[1].set_title('Synthetic Data from Gaussian Copula')
axes[1].set_xlabel('GDP')
axes[1].set_ylabel('Unemployment Rate')

plt.tight_layout()
plt.show()


################################################################
################ 3D Guassian copula density ####################
################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fredapi import Fred
from scipy.stats import norm

api_key = '1a3ba5a60feccf8ab6e2ab7dcb2671ec'
fred = Fred(api_key=api_key)

def get_data(series_id, start_date=None, end_date=None):
    data = fred.get_series(series_id, start_date, end_date)
    return data

gdp_data = get_data('GDP', start_date='2000-01-01')
unemployment_data = get_data('UNRATE', start_date='2000-01-01')

gdp_monthly_data = gdp_data.resample('M').last()
unemployment_monthly_data = unemployment_data.resample('M').last()

gdp_monthly_returns = gdp_monthly_data.pct_change().dropna()
unemployment_monthly_returns = unemployment_monthly_data.pct_change().dropna()

data_combined = pd.concat([gdp_monthly_returns, unemployment_monthly_returns], axis=1)
data_combined.columns = ['GDP Returns', 'Unemployment Rate Returns']

# Drop rows with missing values
data_combined.dropna(inplace=True)

# Convert to ranks
rank_data = data_combined.rank() / len(data_combined)

# Convert ranks to Gaussian samples
gaussian_samples = norm.ppf(rank_data.values)

# Estimate the Gaussian copula's correlation parameter
rho = np.corrcoef(rank_data, rowvar=False)[0, 1]

mean = [0, 0]
cov = [[1, rho], [rho, 1]]

# Generate samples from the Gaussian copula
num_samples = 10000
gaussian_samples = np.random.multivariate_normal(mean, cov, num_samples)

# Convert samples to uniform marginals
uniform_samples = norm.cdf(gaussian_samples)

# Create a 2D histogram
hist, xedges, yedges = np.histogram2d(uniform_samples[:, 0], uniform_samples[:, 1], bins=50, density=True)

# Create a meshgrid for the 3D plot
xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0
dx = dy = 0.01 * np.ones_like(zpos)
dz = hist.ravel()

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True)
ax.set_xlabel("GDP Returns")
ax.set_ylabel("Unemployment Rate Returns")
ax.set_zlabel("Density")
ax.set_title("Gaussian Copula Density")
plt.show()
