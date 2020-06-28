import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("bank-additional-full.csv",sep=";")
le = LabelEncoder()

print("Job")
job_labels = le.fit_transform(df['job'])
job_mappings = {index: label for index, label in enumerate(le.classes_)}
print(job_mappings)

print("Marital")
marital_labels = le.fit_transform(df['marital'])
marital_mappings = {index: label2 for index, label2 in enumerate(le.classes_)}
print(marital_mappings)

print("Education")
education_labels = le.fit_transform(df['education'])
education_mappings = {index: label3 for index, label3 in enumerate(le.classes_)}
print(education_mappings)

print("Default")
default_labels = le.fit_transform(df['default'])
default_mappings = {index: label4 for index, label4 in enumerate(le.classes_)}
print(default_mappings)

print("Housing")
housing_labels = le.fit_transform(df['housing'])
housing_mappings = {index: label5 for index, label5 in enumerate(le.classes_)}
print(housing_mappings)

print("Loan")
loan_labels = le.fit_transform(df['loan'])
loan_mappings = {index: label6 for index, label6 in enumerate(le.classes_)}
print(loan_mappings)

print("Contact")
contact_labels = le.fit_transform(df['contact'])
contact_mappings = {index: label7 for index, label7 in enumerate(le.classes_)}
print(contact_mappings)

print("month")
month_labels = le.fit_transform(df['month'])
month_mappings = {index: label8 for index, label8 in enumerate(le.classes_)}
print(month_mappings)

print("day_of_week")
day_of_week_labels = le.fit_transform(df['day_of_week'])
day_of_week_mappings = {index: label9 for index, label9 in enumerate(le.classes_)}
print(day_of_week_mappings)

print("poutcome")
poutcome_labels = le.fit_transform(df['poutcome'])
poutcome_mappings = {index: label10 for index, label10 in enumerate(le.classes_)}
print(poutcome_mappings)

print("y")
y_labels = le.fit_transform(df['y'])
y_mappings = {index: label11 for index, label11 in enumerate(le.classes_)}
print(y_mappings)

df['job'] = job_labels
df['marital'] = marital_labels
df['education'] = education_labels
df['default'] = default_labels
df['housing'] = housing_labels
df['loan'] = loan_labels
df['contact'] = contact_labels
df['month'] = month_labels
df['day_of_week'] = day_of_week_labels
df['poutcome'] = poutcome_labels
df['y'] = y_labels

