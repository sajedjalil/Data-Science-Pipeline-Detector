from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(["paris", "paris", "tokyo", "amsterdam"])
print(y)