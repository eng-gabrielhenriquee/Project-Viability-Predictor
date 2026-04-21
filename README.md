## Project Viability Predictor
Sistema baseado em Machine Learning para prever a viabilidade de projetos com base em indicadores financeiros e de impacto.

## Problema

#Tomadores de decisão frequentemente precisam avaliar se um projeto é viável ou não, considerando:

Investimento necessário
Retorno esperado
Impacto do projeto

Essa análise, quando feita manualmente, pode ser:

subjetiva
inconsistente
difícil de escalar

Construir um modelo de classificação capaz de prever automaticamente se um projeto é:

✅ Viável
❌ Não viável

## Dataset

O modelo utiliza as seguintes variáveis:
| Variável        | Descrição                  |
| --------------- | -------------------------- |
| investment      | Valor investido no projeto |
| expected_return | Retorno esperado           |
| impact_score    | Impacto (escala de 1 a 10) |
| viability       | Classe alvo (0 ou 1)       |

## Tecnologias Utilizadas:

Python
Pandas
Scikit-learn
Joblib


## Modelo Utilizado

Foi utilizado o algoritmo:

Regressão Logística (Logistic Regression)

## Por quê?

A regressão logística é adequada para:

problemas de classificação binária

interpretação probabilística

rapidez e eficiência em datasets pequenos

🔄 Pipeline do Modelo

Leitura dos dados (pandas)
Separação de variáveis (X, y)
Normalização (StandardScaler)
Divisão treino/teste
Treinamento (LogisticRegression)
Avaliação (classification_report)
Salvamento do modelo (joblib)
Predição de novos dados

📊 Métricas:

O modelo gera automaticamente:

Precision
Recall
F1-score

Essas métricas ficam salvas para reutilização.
