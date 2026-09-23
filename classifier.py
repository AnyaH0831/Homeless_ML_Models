import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report

# 1. Load data
df = pd.read_csv('data/merged_common_columns.csv')

numerical_cols = ['age']
categorical_cols = [
    'gender', 'has_dependents', 
    'outdoor_sleeping', 'chronic_homeless', 'youth', 'no_income', 'income_type'
    
]
#  'mental_health', 'substance_use','indigenous_flag',
X = df[numerical_cols + categorical_cols]
y = df['data_source'].astype(str)

# 2. Convert categories to integer tracking (HistGradientBoosting prefers Ordinal over One-Hot)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols), # No scaling or imputation required!
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Create the unified pipeline using the Tree-based classifier
import sklearn
from sklearn.pipeline import Pipeline

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(random_state=42))
])

# 4. Train and test 
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
