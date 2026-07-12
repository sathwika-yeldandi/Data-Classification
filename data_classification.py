import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    iris = load_iris()

    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['species'] = iris.target

    print("Dataset preview:")
    print(data.head())
    print("\n")

    X = data.drop('species', axis=1)
    y = data['species']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Accuracy:", accuracy)
    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, predictions))

    sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=X.columns)
    predicted_class = model.predict(sample)
    print("\nPredicted class for sample:", iris.target_names[predicted_class[0]])


if __name__ == "__main__":
    main()
