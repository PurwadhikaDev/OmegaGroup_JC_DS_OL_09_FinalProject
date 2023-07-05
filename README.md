# Hotel Booking Demand Machine Learning Project
by Omega team: Abdurrahman Saleh Adijaya, Dzul Fadli Rahman, & Vonny Yudianto

## Problem Statement
Cancellations have negative impacts on hotels, such as reduced revenue and inefficiencies in resource allocation. To mitigate these effects, it is necessary to identify variables that strongly correlate with cancellations. By doing so, we can enable the hotel to predict cancellations and, thus, produce more accurate forecasts.

## Goals
Our project aims to:
1. Identify variables that have a strong correlation with cancellations: By exploring different variables and their relationship with cancellations, we aim to unearth key factors that significantly influence the likelihood of a booking cancellation.
2. Create machine learning models from the identified variables: With these key variables identified, we will develop machine learning models that can predict the cancellation potential of bookings.

## Metric Evaluation
Given the imbalanced nature of the dataset, we are using metrics that provide better insights into the performance of our model on both the majority and minority classes:
- Precision: This metric shows the proportion of correctly predicted positive observations out of the total predicted positives. A model that has no false positives has a precision of 1.0.
- Recall (Sensitivity): Recall indicates the proportion of actual positives correctly identified. A model that produces no false negatives has a recall of 1.0.
- F1 Score: The F1 score is the harmonic mean of precision and recall, taking both false positives and false negatives into account. It's particularly useful when class imbalance is present in the data.
- Area Under the Receiver Operating Characteristic Curve (AUC-ROC): The ROC curve is a performance measurement for classification problems at various threshold settings, and AUC represents the likelihood of the model distinguishing observations from two classes.
- Accuracy: This metric essentially measures what percent of the total predictions our model got right.

Although we will evaluate our model on all these metrics, our primary focus will be on the Recall score. The recall score is critical in our scenario because a high recall rate will reduce the possibility of falsely predicting cancellations as not cancelled. This way, we can enable the hotel to take preventive actions to avoid potential cancellation of bookings, thus maximizing revenue and improving resource allocation.

## Prerequisites
Before running the scripts in this project, please ensure you have the following Python libraries installed in your environment. These libraries power various data processing, analysis, visualization, and machine learning tasks in our project:

- Warnings: For suppressing warnings.
- Pandas: For data manipulation and analysis.
- NumPy: For numerical operations.
- datetime: For working with dates and times.
- pickle: For serializing and de-serializing Python object structures.
- Matplotlib and Seaborn: For data visualization.
- missingno: For visualizing missing values in the dataset.
- IPython: For interactive computing and displaying.
- SciPy: For scientific computing and technical computing.
- sklearn: For machine learning and data processing tasks.
- category_encoders: For encoding categorical variables.
- RandomizedSearchCV, StratifiedKFold, train_test_split, cross_val_score, cross_validate: For model selection and evaluation.
- LogisticRegression, DecisionTreeClassifier, KNeighborsClassifier, RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, XGBClassifier, LGBMClassifier: For building various machine learning models.
- uniform, randint: For generating random numbers.
- imblearn: For handling imbalanced datasets.
Please make sure all the above libraries are installed and properly functioning in your Python environment.
