import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

data = pd.read_csv("bank-additional-full.csv",sep=";")
df = pd.DataFrame(data)

# heat map
sb.set(font_scale=1.0)
corr = df.corr()
sb.heatmap(corr, annot=True, fmt='.2f')
plt.show()

# histogram --> age
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.hist(df['age'], bins=7)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('#Person')
plt.show()

# bar chart -> job
var_job = df['job'].value_counts()
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_xlabel('Job')
ax1.set_ylabel('Counts')
ax1.set_title("Type of Job Distribution")
var_job.plot(kind='bar', color=['maroon', 'orange', 'rosybrown', 'red', 'purple', 'seagreen', 'indianred', 'steelblue', 'lightsalmon', 'chocolate', 'slategray', 'blue'])
plt.show()

# bar chart -> education
var_education = df['education'].value_counts()
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_xlabel('education')
ax1.set_ylabel('Counts')
ax1.set_title("Type of Education Distribution")
var_education.plot(kind='bar', color=['lightsalmon', 'slategray', 'orange', 'rosybrown', 'red', 'purple', 'seagreen', 'lightsteelblue'])
plt.show()

# pie chart --> y
var_y = df['y'].value_counts(normalize=True) * 100
plt.pie(var_y, labels=['No', 'Yes'], autopct='%1.1f%%', colors=['steelblue', 'indianred'])
plt.title('Distribution of y')
plt.show()

# pie chart --> default
var_default = df['default'].value_counts(normalize=True) * 100
plt.pie(var_default, labels=['No', 'Yes', 'Unknown'], autopct='%1.1f%%', colors=['limegreen', 'slategrey','darkgoldenrod'])
plt.title('Distribution of Default')
plt.show()

# pie chart --> housing
var_housing = df['housing'].value_counts(normalize=True) * 100
plt.pie(var_housing, labels=['No', 'Yes', 'Unknown'], autopct='%1.1f%%', colors=['salmon', 'teal','lightsteelblue'])
plt.title('Distribution of Housing')
plt.show()

# pie chart --> loan
var_loan = df['loan'].value_counts(normalize=True) * 100
plt.pie(var_loan, labels=['No', 'Yes', 'Unknown'], autopct='%1.1f%%', colors=['cornflowerblue', 'palevioletred', 'darkkhaki'])
plt.title('Distribution of Loan')
plt.show()

# box plot --> marital
sb.boxplot(x='marital', y ='age', data=df)
plt.title("Marital Distribution by Age")
plt.show()

# categorical plot
sb.catplot(x="contact", y="duration", data=df)
plt.title("Last Contact Duration by Communication Type")
plt.show()

sb.catplot(x="day_of_week", y="duration", data=df)
plt.title("Last Contact Day of the Week by Duration")
plt.show()

sb.catplot(x="y", y="age", hue='marital', data=df)
plt.title("Marital Distribution by Age and y")
plt.show()