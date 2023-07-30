## Customer Segmentation and Insurance Claim Prediction Using Data Mining Techniques
### A leading bank wants to develop a customer segmentation to give promotional offers based on credit card usage, and a health insurance firm wants to identify the reason for higher claim frequency

        🐍 Language: Python
	    📚 Libraries: NumPy, Pandas, Seaborn, Matplotlib.pyplot, scipy, sklearn
	    🗃️ File Type: Data.csv

## Market Segmentation Analysis - Clustering

### Problem Statement 1
A leading bank wants to develop a customer segmentation to give promotional offers to its customers. They collected a sample that summarizes the activities of users during the past few months. You are given the task to identify the segments based on credit card usage
### Data Dictionary
1. spending: Amount spent by the customer per month (in 1000s)
2. advance_payments: Amount paid by the customer in advance by cash (in 100s)
3. probability_of_full_payment: Probability of payment done in full by the customer to the
bank
4. current_balance: Balance amount left in the account to make purchases (in 1000s)
5. credit_limit: Limit of the amount in credit card (10000s)
6. min_payment_amt : minimum paid by the customer while making payments for purchases
made monthly (in 100s)
7. max_spent_in_single_shopping: Maximum amount spent in one purchase (in 1000s)
### Insights from EDA (Exploratory Data Analysis)
* Dataset contains 210 rows and 7 columns, all of which are of float64 data type.
* No missing, null, duplicated, or bad values present in the dataset.
* Column indexes: spending, advance_payments, probability_of_full_payment, current balance, credit_limit, min_payment_amt, max_spent_in_single_shopping.
* Average spending per month: $14.845k
* Average advance payment per month: $14.55k (ranging from $12.41k to $17.25k)
* Probability of full payment: average of 0.87
* Average current balance: $5.62k (ranging from $4.90k to $6.67k)
* Minimum payment for monthly purchases: ranging from $0.77 to $8.46, with an average of $3.70.
* Average credit limit: $3.26k (ranging from $2.56k to $8.46k)
* Average maximum spent in a single shopping: $5.40k (ranging from $5.22k to $6.55k)
* Considerable correlation between several variables.
* Minimal presence of outliers.
* Highly correlated variables: 'Spending' and 'Advance_payments'.
### Customer Segmentation Results
#### The customer segmentation table indicates three clusters with different performance levels:
* Cluster 1: High performing customers (Tier 1)
* Cluster 2: Low performing customers/new customers (Tier 3)
* Cluster 3: Medium performing customers (Tier 2)
#### Cluster 1:
* High performing customers spend an average amount of 18.37 (in 1000s) per month, which is higher than the other clusters.
* They make advance cash payments of 16.14 (in 100s), indicating a good payment behavior with a mean probability of full payment at 0.88.
* High performing customers have the highest mean current balance of 6.16 (in 1000s).
* Their average credit balance is 3.68 (in 10000s), and they spend a maximum amount of 6.01 (in 1000s) in one purchase.
#### Cluster 2:
* Low performing customers spend an average amount of 11.87 (in 1000s) per month, which is significantly lower than the high performing customers.
* They make advance cash payments of 13.26 (in 100s) and have a mean probability of full payment at 0.84.
* The mean current balance for low performing customers is 5.24 (in 1000s), and their average credit balance is 2.86 (in 10000s).
* They make a minimum amount of 4.94 (in 100s) for monthly purchases and spend a maximum of 5.12 (in 1000s) in one purchase.
#### Cluster 3:
* Medium performing customers spend an average amount of 14.20 (in 1000s) per month, which is between the high and low performing customers.
* They make advance cash payments of 14.23 (in 100s) and have a mean probability of full payment at 0.88.
* The mean current balance for medium performing customers is 5.48 (in 1000s), and their average credit balance is 3.23 (in 10000s).
* They make a minimum amount of 2.61 (in 100s) for monthly purchases and spend a maximum of 5.09 (in 1000s) in one purchase.
### Promotion Strategies for Different Clusters 🎯📈
#### Cluster 0 - Low spending
* Target this segment by providing zero-cost annual charges or benefits like free coupons, discounts, cashbacks, & low-interest credit cards.
* Implement early payment offers/rewards to boost their advance payment rate.
* Lower their minimum payment amount by increasing their credit limit, encouraging more spending.
* Use a virtual point system to reward customers with points for credit card usage, redeemable during future purchases.
#### Cluster 1 - High Performing Customers
* Offer discounts or rewards on their next significant transaction to further increase their spending.
* Partner with luxury brands to drive high-performing customers to spend more in one purchase.
* Provide luxurious benefit cards, secure benefit cards, rewards, and loyalty points for every spent.
#### Cluster 2 - Medium Performing Customers
* Increase credit limits and minimize interest rates for customers making decent purchases and paying bills on time.
* Upsell by offering lucrative discounts to encourage them to use premium accounts, leading to increased transactions.
* Partner with premium e-commerce sites to boost their spending rate.
* Implement a customer loyalty program and reward bulk purchases with maximum discounts.
