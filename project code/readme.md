# Upload the CSV file to Colab from local machine
from google.colab import files

uploaded = files.upload()
# Data manipulation
import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# For better table display
from IPython.display import display

# Set Seaborn style
sns.set(style="whitegrid")

# Make sure plots display inline
%matplotlib inline
# Read the uploaded dataset
df = pd.read_excel("HealthCareData.xlsx")

# Display first 5 rows
display(df.head())

# Print info about the dataset
print("\nDataset Info:\n")
df.info()

# Print shape
print("\nDataset Shape:", df.shape)
# Check the shape of the dataset
print("Dataset Shape:", df.shape)

# Check data types and non-null counts
print("\nDataset Info:\n")
df.info()

# Check for missing values
print("\nMissing Values per Column:\n")
print(df.isnull().sum())
# Check unique values in 'Gender' column
print("\nUnique values in 'Gender':", df['Gender'].unique())

# Manual encoding: Male → 1, Female → 0
df['Gender'] = [1 if gender == 'Male' else 0 for gender in df['Gender']]
# List numerical columns (excluding the target if needed)
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot((len(numerical_cols)+2)//3, 3, i+1)
    sns.boxplot(x=df[col], color='lightblue')
    plt.title(col)
    plt.tight_layout()
plt.show()
# Function to cap outliers using IQR method
def cap_outliers(col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower_bound, lower_bound,
                       np.where(df[col] > upper_bound, upper_bound, df[col]))

# Apply to all numerical columns (excluding 'Dataset')
for col in numerical_cols:
    cap_outliers(col)
from sklearn.model_selection import train_test_split

# Split into features and label
X = df.drop('Predicted Value(Out Come-Patient suffering from liver  cirrosis or not)', axis=1)
y = df['Predicted Value(Out Come-Patient suffering from liver  cirrosis or not)']

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
from sklearn.preprocessing import Normalizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Identify categorical columns
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns


# Convert columns with mixed types to string type
for col in categorical_cols:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)


# Create a column transformer to apply one-hot encoding to categorical columns and impute numerical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', SimpleImputer(strategy='mean'), numerical_cols) # Impute numerical columns
    ],
    remainder='passthrough' # This might pass through any columns not specified above, which we don't want.
)

# Create a pipeline that first preprocesses the data and then applies normalization
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('normalizer', Normalizer(norm='l1'))])


# Apply the pipeline to X_train and X_test
X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)

# Confirm shape
print("Normalized X_train shape:", X_train.shape)
print("Normalized X_test shape:", X_test.shape)
# Descriptive statistics for numerical columns
print("📊 Descriptive Statistics (Numerical Columns):\n")
display(df.describe())
# Descriptive statistics for categorical columns
print("🧾 Descriptive Statistics (Categorical Columns):\n")
display(df.describe(include='object'))
# For example, get value counts for Gender column
print("🔢 Value counts for 'Gender':\n")
print(df['Gender'].value_counts())
import seaborn as sns
import matplotlib.pyplot as plt

# Select numeric columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Plot histograms
plt.figure(figsize=(15, 12))
for i, col in enumerate(numerical_cols):
    plt.subplot((len(numerical_cols) + 2)//3, 3, i+1)
    sns.histplot(data=df, x=col, kde=True, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.tight_layout()
plt.show()
# Select categorical columns
categorical_cols = df.select_dtypes(include='object').columns.tolist()

# Plot countplots
plt.figure(figsize=(10, 5))
for i, col in enumerate(categorical_cols):
    plt.subplot(1, len(categorical_cols), i+1)
    sns.countplot(data=df, x=col, palette='Set2')
    plt.title(f'Countplot of {col}')
    plt.tight_layout()
plt.show()
# Boxplots for all numeric features
plt.figure(figsize=(15, 12))
for i, col in enumerate(numerical_cols):
    plt.subplot((len(numerical_cols) + 2)//3, 3, i+1)
    sns.boxplot(x=df[col], color='lightcoral')
    plt.title(f'Boxplot of {col}')
    plt.tight_layout()
plt.show()
# Compute correlation matrix
correlation_matrix = df.corr(numeric_only=True)

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()
# Models
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Evaluation
from sklearn.metrics import accuracy_score

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
def models_eval_mm(X_train, y_train, X_test, y_test):
    models = {
        "Logistic Regression": LogisticRegression(),
        "Logistic Regression CV": LogisticRegressionCV(cv=5),
        "XGBoost Classifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "Ridge Classifier": RidgeClassifier(),
        "KNN Classifier": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}

    print("Training Models...\n")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        acc = accuracy_score(y_test, y_pred)

        results[name] = {
            "Train Score": round(train_score, 4),
            "Test Score": round(test_score, 4),
            "Accuracy": round(acc, 4),
            "Model": model
        }

    return results
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


# 1. Encode all categorical columns
df_encoded = df.copy()  # make a copy to preserve original
label_encoder = LabelEncoder()
for col in df_encoded.select_dtypes(include='object').columns:
  # Convert to string type before encoding
  df_encoded[col] = df_encoded[col].astype(str)
  df_encoded[col] = label_encoder.fit_transform(df_encoded[col])

# 2. Split features and label
# Ensure the target column is treated as object/string type before splitting
df_encoded['Predicted Value(Out Come-Patient suffering from liver  cirrosis or not)'] = df_encoded['Predicted Value(Out Come-Patient suffering from liver  cirrosis or not)'].astype(str)

X = df_encoded.drop('Predicted Value(Out Come-Patient suffering from liver  cirrosis or not)', axis=1)
y = df_encoded['Predicted Value(Out Come-Patient suffering from liver  cirrosis or not)']

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 4. Preprocess the data using the pipeline defined in the previous cell
# Identify categorical columns
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

# Create a column transformer to apply one-hot encoding to categorical columns and impute numerical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', SimpleImputer(strategy='mean'), numerical_cols) # Impute numerical columns
    ],
    remainder='passthrough' # This might pass through any columns not specified above, which we don't want.
)

# Create a pipeline that first preprocesses the data and then applies normalization
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('normalizer', Normalizer(norm='l1'))])

# Apply the pipeline to X_train and X_test
X_train_processed = pipeline.fit_transform(X_train)
X_test_processed = pipeline.transform(X_test)

# Encode the target variable and convert to integer type
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train.astype(str)).astype(int)
y_test_encoded = le.transform(y_test.astype(str)).astype(int)


# 5. Define your model training function if not already defined
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


def models_eval_mm(X_train, y_train, X_test, y_test):
    models = {
        "Logistic Regression": LogisticRegression(),
        "Logistic Regression CV": LogisticRegressionCV(cv=5),
        "XGBoost Classifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "Ridge Classifier": RidgeClassifier(),
        "KNN Classifier": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}

    print("Training Models...\n")
    for name, model in models.items():
      model.fit(X_train, y_train)
      y_pred = model.predict(X_test)
      train_score = model.score(X_train, y_train)
      test_score = model.score(X_test, y_test)
      acc = accuracy_score(y_test, y_pred)

      results[name] = {
            "Train Score": round(train_score, 4),
            "Test Score": round(test_score, 4),
            "Accuracy": round(acc, 4),
            "Model": model
        }

    return results

# 6. Train models and evaluate using the processed data and encoded target
model_results = models_eval_mm(X_train_processed, y_train_encoded, X_test_processed, y_test_encoded)

# 7. Display results
print("\n📊 Model Evaluation Results:\n")
best_model_name = ""
best_accuracy = 0

for name, result in model_results.items():
    print(f"🔍 {name}")
    print(f"    ✅ Train Score: {result['Train Score']}")
    print(f"    ✅ Test Score : {result['Test Score']}")
    print(f"    ✅ Accuracy    : {result['Accuracy']}")
    print("-" * 45)

    if result["Accuracy"] > best_accuracy:
        best_accuracy = result["Accuracy"]
        best_model_name = name
        best_model = result["Model"]

print(f"\n🏆 Best Performing Model: {best_model_name} with Accuracy = {best_accuracy}")
# Display Results
print("\n📊 Model Evaluation Results:\n")
best_model_name = ""
best_accuracy = 0

for name, result in model_results.items():
    print(f"{name}")
    print(f"    Train Score: {result['Train Score']}")
    print(f"    Test Score : {result['Test Score']}")
    print(f"    Accuracy    : {result['Accuracy']}")
    print("-" * 45)

    if result["Accuracy"] > best_accuracy:
        best_accuracy = result["Accuracy"]
        best_model_name = name
        best_model = result["Model"]

print(f"\nBest Performing Model: {best_model_name} with Accuracy = {best_accuracy}")

