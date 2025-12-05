import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import warnings
import pickle

# ✅ use imblearn's Pipeline + SMOTE
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')



# Load your dataset
df = pd.read_csv("Dengue_clinical_dataset.csv")

# Drop non-model features
df_model = df.drop(columns=['Id','Location'])


# Label Encoding
le_gender = LabelEncoder()
le_outcome = LabelEncoder()
df_model['Gender'] = le_gender.fit_transform(df_model['Gender'].str.lower())
df_model['Outcome'] = le_outcome.fit_transform(df_model['Outcome'].str.capitalize())

# Feature lists
bool_features = ['Fever', 'Headache', 'Muscle_Pain', 'Rash', 'Vomiting']
num_features = [col for col in df_model.columns if col not in ['Outcome'] + bool_features]

# Prepare inputs
X = df_model.drop('Outcome', axis=1)
y = df_model['Outcome']


# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('scale', StandardScaler(), num_features)
], remainder='passthrough')

# ✅ Complete pipeline with SMOTE (applied only to training folds/splits)
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('smote', SMOTE(sampling_strategy='minority', random_state=42)),
    ('select', SelectKBest(score_func=f_classif, k='all')),
    ('classifier', RandomForestClassifier(random_state=42))
])



# Hyperparameter tuning grid
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

# 5-fold Stratified Cross Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
grid.fit(X, y)

# Save the best estimator (trained pipeline) to disk
pickle.dump(grid.best_estimator_, open('model.pkl', 'wb'))

# Final results
print("✅ Best CV Accuracy: {:.2f}%".format(grid.best_score_ * 100))
print("📌 Best Hyperparameters:", grid.best_params_)
print("\n📊 Classification Report on Full Dataset:")
print(classification_report(y, grid.best_estimator_.predict(X), target_names=le_outcome.classes_))


prediction = grid.predict(X)
accuracy = grid.score(X,y)
print("Accuracy: ",accuracy)

new_features_list = [1,19,294321,6500,False,0,False,True,False,True]
# Create a DataFrame for prediction, ensuring correct column order and structure
new_features_df = pd.DataFrame([new_features_list], columns=X.columns)
predict_deng = grid.predict(new_features_df)
print(predict_deng)