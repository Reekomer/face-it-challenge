import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from collections import namedtuple
from sklearn import metrics


DATASET_PATH = "/Users/jeremydegail/Desktop/repo/case_study/datasets/toxicity.csv"
TOXIC_LABELS = ["Offensive", "Profanity",
                "Very offensive", "Extremely offensive", "Hate speech"]
UNUSED_COLUMNS = ["row"]
FEATURE_NAMES = ["flirtation", "identity_attack", "insult",
                 "severe_toxicity", "sexually_explicit", "threat"]

TrainingSets = namedtuple("TrainingSets", "X_train X_test y_train y_test")


def main(raw_data: pd.DataFrame) -> None:
    dataset = _clean_unused_elements(raw_data)
    dataset["target"] = dataset["label"].apply(_populate_target_feature)
    training_sets = _split_dataset_for_training(dataset)
    random_forest_classifier = _train_random_forest(training_sets)
    _evaluate_random_forest_classifier(random_forest_classifier, training_sets)
    _display_feature_importances(random_forest_classifier)


def _clean_unused_elements(raw_data: pd.DataFrame) -> pd.DataFrame:
    raw_data.drop(UNUSED_COLUMNS, inplace=True, axis=1)
    return raw_data[raw_data["label"] != "Unknown"]


def _populate_target_feature(label: str) -> int:
    if label in TOXIC_LABELS:
        return 1
    return 0


# try with K-Fold
def _split_dataset_for_training(dataset: pd.DataFrame) -> TrainingSets:
    X = dataset[FEATURE_NAMES]  # Features
    y = dataset['target']  # Labels
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3)  # 70% training and 30% test
    return TrainingSets(X_train, X_test, y_train, y_test)


def _train_random_forest(training_sets: TrainingSets) -> RandomForestClassifier:
    # Create a Gaussian Classifier
    random_forest_classifier = RandomForestClassifier(n_estimators=100)
    # Train the model using the training sets y_pred=clf.predict(X_test)
    random_forest_classifier.fit(training_sets.X_train, training_sets.y_train)
    return random_forest_classifier


def _evaluate_random_forest_classifier(random_forest_classifier: RandomForestClassifier, training_sets: TrainingSets) -> None:
    y_pred = random_forest_classifier.predict(training_sets.X_test)
    logger.info("Accuracy: {}", metrics.accuracy_score(
        training_sets.y_test, y_pred))


def _display_feature_importances(random_forest_classifier: RandomForestClassifier) -> None:
    feature_imp = pd.Series(
        random_forest_classifier.feature_importances_,
        index=FEATURE_NAMES
        ).sort_values(ascending=False)
    sns.barplot(x=feature_imp, y=feature_imp.index)
    # Add labels to your graph
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main(pd.read_csv(DATASET_PATH))


