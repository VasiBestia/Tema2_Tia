import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================================
# 1. PROCESAREA DATELOR
# ==========================================


# Incarcarea setului de date
file_path = "salary.csv"
df = pd.read_csv(file_path)

# ---------------------------------------------------------
# Curatarea Datelor
# ---------------------------------------------------------

df_clean = df.dropna().copy()

# Verificam daca exista valori duplicate si le eliminam pentru a evita overfitting-ul pe date identice
df_clean = df_clean.drop_duplicates()


# ---------------------------------------------------------
# Transformarea Variabilelor Categorice (Encoding)
# ---------------------------------------------------------
# 'Education Level' este o variabila ordinala (exista o ierarhie clara).
# Folosim o mapare manuala pentru a pastra ordinea importantei: Bachelor < Master < PhD.
education_mapping = {"Bachelor's": 1, "Master's": 2, "PhD": 3}
df_clean["Education Level"] = df_clean["Education Level"].map(education_mapping)

# Pentru 'Gender' si 'Job Title' folosim LabelEncoder
le_gender = LabelEncoder()
df_clean["Gender"] = le_gender.fit_transform(df_clean["Gender"])

le_job = LabelEncoder()
df_clean["Job Title"] = le_job.fit_transform(df_clean["Job Title"])

# ---------------------------------------------------------
# Pregatirea datelor pentru antrenare
# ---------------------------------------------------------
# Definim X (features) si y (target - Salariul)
X = df_clean.drop("Salary", axis=1)
y = df_clean["Salary"]

# Impartim datele: 80% antrenare, 20% testare.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Dimensiune set antrenare: {X_train.shape}")
print(f"Dimensiune set testare: {X_test.shape}")
print("-" * 50)


# ==========================================
# 2. ANTRENAREA SI EVALUAREA MODELELOR
# ==========================================
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"--- Performanta Model: {model_name} ---")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
    print(f"MAE (Mean Absolute Error): {mae:.2f}")
    print(
        f"R2 Score: {r2:.4f}"
    )  # Cat de bine explica modelul variatia datelor (aproape de 1 e ideal)
    print("-" * 50)
    return r2


# ---------------------------------------------------------
# Model A: Decision Tree Regressor (Arbore de Decizie Simplu)
# ---------------------------------------------------------
# Vom cauta hiperparametrii optimi (adancimea arborelui, split-ul minim) pentru a evita overfitting-ul.
dt_params = {
    "max_depth": [None, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

# GridSearchCV antreneaza modelul pe toate combinatiile din 'dt_params' folosind Cross-Validation.
dt_grid = GridSearchCV(
    DecisionTreeRegressor(random_state=42), dt_params, cv=5, scoring="r2", n_jobs=-1
)
dt_grid.fit(X_train, y_train)

print(f"Cei mai buni parametri pentru Decision Tree: {dt_grid.best_params_}")
best_dt_model = dt_grid.best_estimator_
evaluate_model(best_dt_model, X_test, y_test, "Decision Tree Optimizat")

# ---------------------------------------------------------
# Model B: Random Forest Regressor (Ensemble)
# ---------------------------------------------------------

rf_params = {
    "n_estimators": [50, 100, 200],  # Numarul de arbori
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
}

rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42), rf_params, cv=3, scoring="r2", n_jobs=-1
)
rf_grid.fit(X_train, y_train)

print(f"Cei mai buni parametri pentru Random Forest: {rf_grid.best_params_}")
best_rf_model = rf_grid.best_estimator_
evaluate_model(best_rf_model, X_test, y_test, "Random Forest Optimizat")

# ---------------------------------------------------------
# Vizualizare Importanta Caracteristici (Feature Importance)
# ---------------------------------------------------------
# Observam care variabile au contribuit cel mai mult la predictia salariului (bazat pe Random Forest)
importances = best_rf_model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Importanta Caracteristicilor (Random Forest)")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=45)
plt.tight_layout()
plt.show()
