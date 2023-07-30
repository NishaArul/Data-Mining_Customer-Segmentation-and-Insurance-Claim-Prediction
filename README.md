## Customer Segmentation and Insurance Claim Prediction Using Data Mining Techniques
#### A leading bank wants to develop a customer segmentation to give promotional offers based on credit card usage, and a health insurance firm wants to identify the reason for higher claim frequency
### Brief Overview
* Collected, cleaned, and analyzed data from various sources, ensuring data integrity and quality for consulting projects
* Applied EDA analysis, feature engineering, and clustering techniques (K-Means & Agglomerative) on the Bank dataset
* Determined optimal clusters using the Elbow curve, Dendrogram, and Silhouette score, enabling segmented offers to customers
* Built classification models (CART, Random Forest, Artificial Neural Network) on insurance firm dataset; identified CART model as optimized with 97% accuracy and recommended strategies to minimize claim frequency

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
* No missing, null, duplicated, or bad values are present in the dataset.
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
### Customer Segmentation Summary 📊🔍
#### The customer segmentation analysis revealed three distinct clusters with varying performance levels [K-Means Clustering]
* Cluster 1: High performing customers (Tier 1)
* Cluster 2: Low performing customers/new customers (Tier 3)
* Cluster 3: Medium performing customers (Tier 2)
#### Cluster 1:
High performing customers spend an average amount of $18.37 per month, showing a strong performance compared to the other clusters. They make advance cash payments of $16.14, maintain the highest mean current balance of $6.16, and have an average credit balance of $3.68. Additionally, they spend a maximum amount of $6.01 in a single purchase.
#### Cluster 2:
Low performing customers spend an average amount of $11.87 per month, significantly lower than high performing customers. They make advance cash payments of $13.26, have a mean current balance of $5.24, and an average credit balance of $2.86. Their minimum monthly purchases are $4.94, and they spend a maximum of $5.12 in one purchase.
#### Cluster 3:
Medium performing customers spend an average amount of $14.20 per month, falling between high and low performing customers. They make advance cash payments of $14.23, maintain a mean current balance of $5.48, and an average credit balance of $3.23. Their minimum monthly purchases are $2.61, and they spend a maximum of $5.09 in one purchase.

### Promotion Strategies for Different Clusters 🎯📈
#### Cluster 0 - Low spending:
* Offer zero-cost annual charges and benefits like free coupons, discounts, cashbacks, and low-interest credit cards.
* Implement early payment offers/rewards to boost advance payment rates.
* Increase credit limits to lower the minimum payment amount and encourage more spending.
* Use a virtual point system to reward credit card usage, with points redeemable during future purchases.
#### Cluster 1 - High Performing Customers:
* Offer discounts or rewards on their next significant transaction to increase spending.
* Partner with luxury brands to drive high-performing customers to spend more in one purchase.
* Provide luxurious benefit cards, secure benefit cards, rewards, and loyalty points for every spent.
#### Cluster 2 - Medium Performing Customers:
* Increase credit limits and minimize interest rates for customers making decent purchases and paying bills on time.
* Upsell by offering lucrative discounts to encourage them to use premium accounts, leading to increased transactions.
* Partner with premium e-commerce sites to boost their spending rate.
* Implement a customer loyalty program and reward bulk purchases with maximum discounts.

These strategies aim to cater to each cluster's unique characteristics, optimizing their spending behavior and enhancing their overall customer experience. 🚀💼📈

## Insurance Analysis - CART-RF-ANN

### Problem Statement 2
An Insurance firm providing tour insurance faced higher claim frequency. The management decided to collect data from the past few years. I was assigned the task to make a model that predicted the claim status and provided recommendations to management. I used CART, RF & ANN and compared the models' performances in train and test sets.
### Data Dictionary
1. Target: Claim Status (Claimed)
2. Code of tour firm (Agency_Code)
3. Type of tour insurance firms (Type)
4. Distribution channel of tour insurance agencies (Channel)
5. Name of the tour insurance products (Product)
6. Duration of the tour (Duration in days)
7. Destination of the tour (Destination)
8. Amount worth of sales per customer in procuring tour insurance policies in rupees (in 100’s)
9. The commission received for tour insurance firm (Commission is in percentage of sales)
10.Age of insured (Age)
### Insights from EDA (Exploratory Data Analysis)
* The dataset contains 3000 rows and 10 columns.
* It consists of 2 integer data types, 2 float types, and 6 object types.
* There are no null values in the dataset.
* The dataset has 130 duplicates.
* Outliers are indicated by drastic changes in min and max values across 4 features.
* The feature 'Duration' has a bad value of -1 and needs to be treated.
* The maximum value of the Tour agencies commission charge exceeds 100% and needs to be treated.
* There are 4 types of agency codes: EPX, C2B, CWT, and JZT, with EPX having the highest frequency.
* Most customers prefer Travel agencies for tour insurance.
* 924 customers claimed their insurance, while 2076 customers did not.
* 2954 customers chose online tour insurance agencies, and only 46 customers chose offline agencies.
* The 'Customized Plan' is the most preferred insurance plan with a value count of 1136, while the 'Gold Plan' is the least preferred with a value count of 109.
* Asia is the most preferred tourist destination with a value count of 2465, while Europe is the least preferred with a value count of 215.
* Correlation analysis shows a strong positive relationship between 'Sales' and 'Commission' (0.76).
* Weak correlations are found between 'Duration' & 'Commission' (0.46) and 'Duration' & 'Sales' (0.55).

Overall, the dataset provides valuable insights into customer preferences, agency choices, and tourist destinations for tour insurance. Additionally, data treatment and correlation analysis were performed to enhance the dataset's quality and understand the relationships between variables. 📊💼🌏
### Data Preparation and Model Building
After conducting Exploratory Data Analysis (EDA), data pre-processing, and data preparation, our dataset is now prepared for supervised modeling algorithms such as Decision Tree, Random Forest, and ANN. It's important to note that we didn't perform outlier treatment. Although treating outliers can sometimes improve model performance, it may also lead to overfitting, and supervised algorithms are generally not sensitive to outliers.

* Feature Engineering: Used label encoding for categorical values to convert them into simple numerical values without losing information.
* Data Splitting: Divided the data into a 70:30 ratio for the train and test sets.
* Classification Models: Built classification models using CART, Random Forest, and Artificial Neural Network (ANN).
* Hyperparameter Tuning: Performed hyperparameter tuning to find the best estimator for each model.
* Variable Importance: Retrieved the variable importance for better understanding.Performance Evaluation: Checked the performance of predictions on both the train and test sets.
* Evaluation Metrics: Utilized accuracy, and confusion matrix, plotted ROC curves, and obtained ROC_AUC scores and classification reports for each model.
### Recommendations
* Prioritize records with higher sales values, as they are more likely to result in insurance claims.
* Invest in resources and promotional campaigns to improve sales through the JZI agency and consider alternative agency partnerships.
* Promote bookings from the Travel Agency to increase insurance claim opportunities.
* Develop cross-selling strategies for insurance based on claim data patterns for customers buying airline tickets.
* Focus on streamlining online experiences to increase conversions and sales value for online reservations.
* Restructure the Gold plan to make it more attractive to customers and review procedures for the Silver plan to reduce insurance claims.
* Implement marketing plans to attract more customers for American and Europe tours, possibly through discounts or special offers.
### Key Performance Indicators (KPIs) of Insurance Claims:
* Increase customer satisfaction to drive higher revenue.
* Combat fraudulent transactions by deploying measures to detect and prevent fraudulent claims.
* Optimize claims recovery methods to efficiently handle and process claims.
* Reduce claim handling costs through process improvements and automation.

Overall, by implementing these recommendations and focusing on key performance indicators, the insurance firm can reduce claim frequency, enhance customer experience, and improve profitability. 📈💼🕵️‍♂
