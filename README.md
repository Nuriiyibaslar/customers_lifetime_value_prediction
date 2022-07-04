CLTV Forecast with BG-NBD and Gamma-Gamma

Business Problem

The UK-based retail company wants to set a roadmap for its sales and marketing activities. In order for the company to make a medium-long-term plan, it is necessary to estimate the potential value that existing customers will provide to the company in the future.

Dataset Story

The dataset named Online Retail II includes the online sales transactions of a UK-based retail company between 01/12/2009 - 09/12/2011. The company's product catalog includes souvenirs and it is known that most of its customers are wholesalers.

About Variables:

InvoiceNo: Invoice Number (If this code starts with C, it means that the transaction has been cancelled)
StockCode : Product code (unique for each product)
Description : Product name
Quantity: Number of products (How many of the products on the invoices were sold)
InvoiceDate : Invoice date
UnitPrice : Invoice price ( Sterling )
CustomerID : Unique customer number
Country : Country name

There was a basic operation to be learned in the CLTV calculation.
Average earnings per purchase * number of purchases
Here, too, our transactions will continue on these two metrics.
Here we will just replace the above operation to fit the mathematical flow.

number of purchases * Average earnings per purchase
CLTV = (Customer Value / Churn Rate) * Profit Margin
Customer Value = Purchase Frequency * Average Order Value