import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([
    [80, 5],
    [85, 6],
    [78, 5],
    [60, 2],
    [55, 1]
])

y = np.array([1, 1, 1, 0, 0])

model = LogisticRegression()
model.fit(X, y)


new_data = np.array([[58, 2]])

prediction = model.predict(new_data)

if prediction == 0:
    print("Concept Drift Detected")
else:
    print("Learning Pattern Normal")
