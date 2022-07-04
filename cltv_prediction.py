################################################ ############
# CLTV Prediction with BG-NBD and Gamma-Gamma
################################################ ############

# 1. Data Preparation
# 2. Expected Number of Transaction with BG-NBD Model
# 3. Expected Average Profit with Gamma-Gamma Model
# 4. Calculation of CLTV with BG-NBD and Gamma-Gamma Model
# 5. Creating Segments by CLTV
# 6. Functionalization of work


################################################ ############
# 1. Data Preparation
################################################ ############

# An e-commerce company divides its customers into segments and according to these segments
# wants to set marketing strategies.

# Dataset Story

# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

# The data set named Online Retail II was created by a UK-based online store.
# Includes sales from 01/12/2009 to 09/12/2011.

# Variables

# InvoiceNo: Invoice number. The unique number of each transaction, that is, the invoice. Aborted operation if it starts with C.
# StockCode: Product code. Unique number for each product.
# Description: Product name
# Quantity: Number of products. It expresses how many of the products on the invoices have been sold.
# InvoiceDate: Invoice date and time.
# UnitPrice: Product price (in GBP)
# CustomerID: Unique customer number
# Country: Country name. Country where the customer lives.


##########################
# Required Library and Functions
##########################

pip install lifetimes
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler


# The task of outlier_threshoulds is to set a threshold value for the variable entered into it.
# Outliers are values that are quite outside of the general distribution of a variable.
# We want to suppress units that are outside the general behavior of the variable.
# Therefore, we first need to set a threshold value.

# Normally, when calculating quarter values, 25% and 75% th values are calculated
# but for this dataset, we have defined the quartile values as 0.01 and 0.99 to the function

def outlier_thresholds(dataframe,variable):
	quartile1 = dataframe[variable].quantile(0.01) # quartile values are calculated
	quartile3 = dataframe[variable].quantile(0.99) # quartile values are calculated
	interquantile_range = quartile3 - quartile1 # Calculate the quarter value difference
	up_limit = quartile3 + 1.5 * interquantile_range # 3rd quarter over one and a half IQR
	low_limit = quartile1 - 1.5 * interquantile_range # 1st quarter six and a half IQR
	return low_limit,up_limit


# replace with thresholds function is a function that can be used to suppress outliers on many issues.

def replace_with_thresholds(dataframe,variable):
	low_limit , up_limit = outlier_thresholds(dataframe,variable)
	dataframe.loc[(dataframe[variable] < low_limit),variable] = low_limit
	dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
	
	
# Since the models we will establish are statistical/probabilistic, the distributions of the variables we will use while establishing these models will directly affect the results.
# Therefore, after creating the variables we have, we need to touch the divergence values in this variable. We will set the threshold values.
# Then, we will replace the outliers we have determined with a certain threshold value with the suppression method, which can be evaluated within the scope of combating outliers.
# To do this, we need two functions. We will spend some time on these functions.


########################
# Reading Data
########################

df_ = pd.read_excel("/Users/nuri/PycharmProjects/pythonProject_n/dsmlbc_nuri/crm_analytics/datasets/online_retail_II.xlsx",sheet_name="Year 2010-2011")
df = df_.copy()
df.describe().T
df.head()
df.isnull().sum()
df.describe([0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99]).T


#########################
# Data Preprocessing
#########################


# We didn't focus on outliers before, but we need to focus here because there's a modeling process here.
# We can't deal with customers without an ID.

df.dropna(inplace=True)
df.describe().T

# C statements in invoice variable are returned, let's get rid of them too

df = df[~df["Invoice"].str.contains("C",na=False)]
df.describe().T
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]
df.describe().T


# I will print with that threshold value
replace_with_thresholds(df,"Quantity")
replace_with_thresholds(df,"Price")
df.describe().T

# There are drastic changes in the standard deviation and maximum values,
# this way we shaved the good side, it could even be more

df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011, 12, 11)

# We determined the day the analysis was made, in fact, we can't put history into this day again.


#########################
# Preparation of Lifetime Data Structure
#########################

# There is a special data format that BG NBD and gamma gamma models expect from us, we need to prepare this data format.
# We went through some pre-processing processes within the framework of its own dynamics, we completed a certain preparation.
# Now, over this data, it can respond to the functions of the lifetimes module, aggregated over users and individualized according to users.
# we need to convert it to a form. Let's remember some of our metrics.


# recency: The elapsed time since the last purchase. Weekly. (user specific)
# For Recency, the note expresses the difference between the last purchase and the first purchase by the customer, not according to the analysis dates.
# is dynamic in particular.
# T: The age of the customer. Weekly. (how long before the analysis date the first purchase was made)
# frequency: total number of repeat purchases (frequency>1)
# Repeat means that the customer has come and shopped for at least the second time and is now our customer.
# we will be performing the calculation.
# monetary: average earnings per purchase

# The frequency, monetary and recency values that appear here are different from the mathematical calculations I've seen before, this is an issue that needs attention.

cltv_df = df.groupby("Customer ID").agg({"InvoiceDate" : [lambda InvoiceDate : (InvoiceDate.max() - InvoiceDate.min()).days,
                                                          lambda InvoiceDate : (today_date - InvoiceDate.min()).days],
                                         "Invoice" : lambda Invoice : Invoice.nunique(),
                                         "TotalPrice" : lambda TotalPrice : TotalPrice.sum()})

# to improve readability
cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ["recency","T","frequency","monetary"]

# we mentioned above as weekly for recency, money and others,
# but we didn't do anything about them being weekly here, let's do it.

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"] 
cltv_df.describe().T
cltv_df = cltv_df[(cltv_df["frequency"] > 1)]

# Convert recency and customer age variables to weekly value
# BG-NBD model expects this from us.
cltv_df["recency"] = cltv_df["recency"] /7
cltv_df["T"] = cltv_df["T"] /7

############
# 2. Establishment of BG-NBD Model
############

bgf = BetaGeoFitter(penalizer_coef=0.001)
# When you give me the frequency, recency and customer age values of the fit method via the model object, it says I will have installed this model. Here gamma and beta distributions
# is used. While finding the parameters here, the maximum likelihood method will be used. And while I can find the parameters, I need an argument says BetaGeoFitter
# penalizer_coef=0.001 : It is the penalty coefficient that will be applied to the coefficients at the stage of finding the parameters of this model.
# What we need to know within the scope of our topic is that the BGNBD Model provides us with the maximum likelihood method to find the parameters of the beta and gamma distributions and to make an estimation.
# creates this model.

# how do we finalize the model?

bgf.fit(cltv_df["frequency"],
        cltv_df["recency"],
        cltv_df["T"])

# lifetimes.BetaGeoFitter: fitted with 2845 subjects, a: 0.12, alpha: 11.41, b: 2.49, r: 2.18>
# If we enter the crm business in a long-term way and we will continue to work on BGNBD and Gamma, how to ensure the optimum values of these parameters
# details should be entered. More serious focus is required.

##############
# Who are the 10 customers we expect to purchase the most in 1 week?
##############

# There is a function for the bgf model, we will get help from it

# to indicate that we will make a 1 week forecast since the first written in the function is 1 week.

bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency'],
                                                        cltv_df['T']).sort_values(ascending=False).head(10)
# This information is very valuable. Business decisions can only be taken from this information and related customers.

bgf.predict(1,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)

# predict can be used here instead of long func name yes but gamma is not used in gamma model let's see let's see

# Let's calculate the purchases we expect in a week for all customers and add this to the cltv_df dataset.

cltv_df["expected_purc_1_week"] = bgf.predict(1,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T'])

##############
# Who are the 10 customers we expect to purchase the most in 1 month?
##############

bgf.predict(4,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)

cltv_df["expected_purc_4_week"] = bgf.predict(4,
                                              cltv_df['frequency'],
                                              cltv_df['recency'],
                                              cltv_df['T'])

# Number of sales expected by the company in a month

bgf.predict(4,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sum()


# this is a very valuable output.
# When examined specifically for these customer segmentations with various breakdowns and channel breakdowns,
# it is an output that will support many business units.


##############
# What is the Expected Number of Sales of the Whole Company in 3 Months?
##############

cltv_df["expected_puch_1_month"] = bgf.predict(4,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T'])

cltv_df["expected_puch_3_month"] = bgf.predict(4*3,
                                                cltv_df['frequency'],
                                                cltv_df['recency'],
                                                cltv_df['T'])

# How can we evaluate the success of the predictions we have made?


##############
# Evaluation of Forecast Results
##############

plot_period_transactions(bgf)
plt.show()

######################
  # Establishing the Gamma-Gamma Model )
######################

ggf = GammaGammaFitter(penalizer_coef = 0.01)

ggf.fit(cltv_df["frequency"],cltv_df["monetary"])



ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                        cltv_df["monetary"]).head(10)
# if we want to see it in descending order

ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                        cltv_df["monetary"]).sort_values(ascending=False).head(10)

# Of all our customers, the expected profit has brought us the average expected profit.


cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                        cltv_df["monetary"])


cltv_df["expected_average_profit"].sort_values(ascending=False).head(10)

cltv_df.sort_values("expected_average_profit",ascending=False).head(10)

# There are very valuable information in the tables created with BG-NBD and Gamma Gamma models,
# for example, those who are under age but have high profit expectations
# so it has the ability to catch potential customers.
####
# Calculation of CLTV with BG-NBD and GG Model
####
# Finally we come to our main purpose part
# We have modeled the expected frequencies with BG-NBD,
# we have modeled the expected profitability with the gamma gamma model,
# so we will be able to combine these two, perform the multiplication in our basic formula and calculate the Customer Lifetime value values.

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency"],
                                   cltv_df["T"],
                                   cltv_df["monetary"],
                                   time=3,# for 3 months
                                   freq="W", # T's frequency information
                                   discount_rate=0.01)

# Customer Lifetime Value method says give me gamma gamma model and bgnbd model, however, frequency, recency, customer age and monetaryd
# show the values, give me a time period, the time argument is monthly, is the data you entered monthly or weekly, give the frequency information of it,
# Which I expect in terms of recency and customer age. In time, you may have an idea to make a discount on the products you sell, let me take that into consideration.
# After expressing this, in our study, the customer lifetime value is calculated for each customer.

cltv.head()

# This is the place where we wanted to come from the beginning of the study. But we can't read it now, we can't evaluate it with other variables.
# Let's get rid of these problems and bring all the data together and make our final evaluations.
# First of all, there is an index problem in this output.

cltv = cltv.reset_index()
cltv.head()

# There is the cltv_df object that we created before. All of our operations were gradually included in it.
# Let's combine this dataset with the dataset containing cltv expressions

cltv_final = cltv_df.merge(cltv,on="Customer ID",how="left")
cltv_final.sort_values(by="clv",ascending=False).head(10)

# we will try to understand the results here more closely by making some interpretations.
# The values ​​we will focus on now are clv,expected_average_profit, monetary,frequency,T,recency,others we have set to get information,
# If we want to make a healthier review, we can add the expected sales of 3 months here.

# let's add it too...
# we added

# Now we will try to analyze, by asking various questions, we will try to understand why this order was formed in this way.


# The most critical point of the BG-NBD model
# How do these customers, whose recency value is so high in itself, promise the greatest value?
# The most critical point of BG NBD
# Normally, we had a perception that it was good for us to have low recency from our previous habits, but the concept of buy till you die says your regular average
# The customer with a behavior with processing capacity, if there is no churn, if there is no dropout, the probability of purchase increases as the customer's recency value increases.
# A very critical and important information is that the customer, who has a certain frequency as a result of the evaluation of the customer within himself, and also the relevant customer
# If the recency value is high, the probability of purchasing is approaching.
# is at hand. The customer made a purchase and then some of them were churn. After every transaction, the sense of reception is satisfied, his needs are met, etc.
# After you make the purchases of the mustetris, it drops.
# The customer's need to buy again begins to emerge. For this reason, the buyer and customer age couples have very high values ​​or are very close to each other.
# And sometimes, considering the age and recency, the potentials left by the customers are so high that the lifetime value values ​​are high.
# But still, we should interpret here not only by looking at the two of them, age and recency values, but also by looking at their frequency and monetary values.
# Any other way will be wrong.

# in rankings based on some values only, some customers in a ranking would be on top, in another app approach, other customers would come out on top
# but here is a ranking by giving importance to each variable.


####################
# Creating the Customer Segment )
####################

# In this section we will segment customers
# As can be seen, the cltv value is a common value formed by the effects of some variables and their weights.
# Therefore, it contains values such as the potential value that the customer will leave and the number of possible sales that the customer will make.
# In fact, when we sort from largest to smallest and start dealing with these customers, we will start to deal with the customers who are most valuable to us.

# Now, let's simply divide these customers into groups and we can start dealing with these groups, such as segment a segment b.
# cltv value had many effects, so let's add a variable called segment directly into our cltv final variable

cltv_final["segment"] = pd.qcut(cltv_final["clv"],4,labels=["D","C","B","A"])

cltv_final.sort_values(by="clv",ascending=False).head(50)

cltv_final.groupby("segment").agg({"count","mean","sum"})



############
# 6. Functionalization of Work
############

def create_cltv_p(dataframe, month=3):
    #1. Data Preprocessing
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    today_date = dt.datetime(2011, 12, 11)

    cltv_df = dataframe.groupby('Customer ID').agg(
        {'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                         lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
         'Invoice': lambda Invoice: Invoice.nunique(),
         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

    cltv_df.columns = cltv_df.columns.droplevel(0)
    cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
    cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
    cltv_df["recency"] = cltv_df["recency"] / 7
    cltv_df["T"] = cltv_df["T"] / 7

    # 2. Establishment of BG-NBD Model
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T'])

    cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                                  cltv_df['frequency'],
                                                  cltv_df['recency'],
                                                  cltv_df['T'])

    cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                                   cltv_df['frequency'],
                                                   cltv_df['recency'],
                                                   cltv_df['T'])

    cltv_df["expected_purc_3_month"] = bgf.predict(12,
                                                   cltv_df['frequency'],
                                                   cltv_df['recency'],
                                                   cltv_df['T'])

    # 3. Establishing the GAMMA-GAMMA Model
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary'])
    cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                                 cltv_df['monetary'])

    # 4. Calculation of CLTV with BG-NBD and GG model.
    
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency'],
                                       cltv_df['T'],
                                       cltv_df['monetary'],
                                       time=month,  # 3 aylık
                                       freq="W",  # T'nin frekans bilgisi.
                                       discount_rate=0.01)

    cltv = cltv.reset_index()
    cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
    cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

    return cltv_final


df = df_.copy()

cltv_final2 = create_cltv_p(df)

cltv_final2.to_csv("cltv_prediction.csv")
