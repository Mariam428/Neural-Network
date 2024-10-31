import pandas as pd
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('birds.csv')

# change categorical values to numerical
gender_mapping = {'male': 0, 'female': 1, 'NA': 2}
df['gender'] = df['gender'].map(gender_mapping)

category_mapping = {'A': 0, 'B': 1, 'C': 2}
df['bird category'] = df['bird category'].map(category_mapping)


# scaling to numerical data
scaler = StandardScaler()

df[['body_mass', 'beak_length', 'beak_depth', 'fin_length']] = scaler.fit_transform(
    df[['body_mass', 'beak_length', 'beak_depth', 'fin_length']]
)

#print(df)



