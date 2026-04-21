import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np
import joblib
import os



def train_or_predict(new_projects):
    model_path = 'logistic_model.joblib'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        scaler = joblib.load(model_path.replace(".joblib", "_scaler.joblib"))
    else:
        df_projects = pd.read_csv("projects_data.csv")

        #separa variaveis
        X = df_projects[["investment", "expected_return", "impact_score"]]
        y = df_projects["viability"]

        #Normalizar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        #Divide treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )

        #treina
        model = LogisticRegression()
        model.fit(X_train, y_train)

        #Avalia
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        #salva o modelo
        joblib.dump(model, model_path)
        joblib.dump(scaler, model_path.replace(".joblib", "_scaler.joblib"))
        joblib.dump(report, model_path.replace(".joblib", "_metrics.joblib"))

    #previsao
    if new_projects:
        df_new_projects = pd.DataFrame(new_projects)
        X_new_scaled = scaler.transform(df_new_projects)
        predictions = model.predict(X_new_scaled)
        #probabilidade de 1
        probabilities = model.predict_proba(X_new_scaled)[:,1]
        df_new_projects["probability"] = probabilities
        df_new_projects["viability"] = predictions

        return df_new_projects, joblib.load(model_path.replace(".joblib", "_metrics.joblib"))
    
    return None, joblib.load(model_path.replace(".joblib", "_metrics.joblib"))



#TESTARR
new_projects = [
    #{"investment": 13000, "expected_return":69000, "impact_score": 7}
    {"investment": 40000, "expected_return":60000, "impact_score": 6}
]

predictions, metrics = train_or_predict(new_projects)

if predictions is not None:
    print("\n Novos projetos e Viabilidade: ")
    print(predictions)

print("\n Métricas do modelo: ")
print(metrics)
