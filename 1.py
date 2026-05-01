import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

rng = np.random.default_rng(42)

customers = 1000
income = rng.normal(loc = 50000, scale = 15000, size = customers)
age = rng.normal(loc = 40, scale = 12, size = customers)
spending_score = rng.uniform(0, 100, size = customers)

z = (
    -0.00005 * income
    -0.03 * spending_score
    +0.04 * age
)

prob = 1/(1+np.exp(-z))
churn = (prob>0.5).astype(int)

df = pd.DataFrame({
    'Income': income,
    'Age': age,
    'Spending Score': spending_score,
    'Churn': churn
})

df.head()
df.info()

churn_rate = df['Churn'].mean()
print(f'Churn Rate: {churn_rate:.2%}')


corr = df.corr(numeric_only = True)
print(corr["Churn"].sort_values(ascending = False))

group_means = df.groupby("Churn").mean()
print(group_means)

plt.figure(figsize=(6, 4))   
sns.histplot(data=df, x='Income', hue='Churn', kde=True)
plt.title("Income Distribution by Churn")
plt.xlabel("Income")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(data = df, x = "Age", y = "Spending Score", hue = "Churn", alpha = 0.6)
plt.title("Age vs Spending Score")
plt.show()

plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot = True, cmap = "coolwarm", fmt = ".2f")
plt.title("Correlation Matrix")
plt.show()


# print(df.head())
