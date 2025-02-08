# Ciência de Dados - EBAC
# Projeto Semantix - Previsão do Nível da Crise de Ansiedade
# Aluno: Lucas Antonio de Sousa Ribeiro

# Imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from io import BytesIO
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# Função para ler os dados
@st.cache_data
def load_data(file_data):
    return pd.read_csv(file_data, sep=',')

# Função para converter o df para csv
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')


# Função principal da aplicação
def main():
    # Configuração inicial da página da aplicação
    st.set_page_config(page_title = 'Nível de Ansiedade', \
        page_icon = ':thermometer',
        layout="wide",
        initial_sidebar_state='expanded'
    )

    # Título principal da aplicação
    st.title("Projeto - Previsão da Crise de Ansiedade")
    st.write("---")

    # Introdução
    st.write('## Entendimento do Negócio')
    st.markdown('''
                Na sociedade atual, as pessoas estão cada vez mais propensas a terem crises de ansiedade, 
                devido a fatores como o nível de stress, noites mal-dormidas, fumo, alto consumo de cafeína, 
                além de outros fatores prejudiciais no dia-a-dia. Neste projeto serão analisados os níveis 
                das crises de ansiedade informadas por 12.000 pessoas de acordo com os fatores mencionados. 
                Além de realizar uma previsão do nível da crise que a pessoa está propensa a ter por meio 
                da utilização de Machine Learning. Os dados foram obtidos no site kaggle, disponíveis neste 
                link: [Anxiety-Attack-Factors](https://www.kaggle.com/datasets/ashaychoudhary/anxiety-attack-factors-symptoms-and-severity/data).
                ''')
    
    st.write('## Entendimento dos Dados')
    st.markdown('''
                Cada pessoa informou sobre 20 variáveis relativas a ela. Tais variáveis estão presentes na 
                tabela abaixo com seus respectivos significados e tipos.
                ''')
    
    st.write('### Dicionário de dados')
    st.markdown('''
                | Variável | Descrição | Tipo |
                |  ---     | ---       | ---  |
                |ID | Número de identificação da pessoa| Númerica|
                |Age| Idade |  Númerica|
                |Gender | Gênero| Categórica|
                |Occupation | Profissão| Catgórica|
                |Sleep Hours | Horas de Sono|Numérica |
                | Physical Activity (hrs/week)| Quantidade de horas de atividade física semanal | Numérica|
                |Caffeine Intake (mg/day) | Quantidade de cafeína diária | Numérica |
                |Alcohol Consumption (drinks/week) | Consumo de álcool semanal| Numérica|
                | Smoking| Informa se a pessoa Fuma| Booleana | 
                |Family History of Anxiety | Histórico familiar de ansiedade| Booleana|
                |Stress Level (1-10) | Nível de Stress do paciente| Numérica|
                |Heart Rate (bpm during attack) | Palpitações durante a crise| Numérica|
                | Breathing Rate (breaths/min)| Taxa de respiração durante a crise | Numérica|
                | Sweating Level (1-5)| Nível de suor|Numérica |
                |Dizziness | Tontura durante a crise|Booleana |
                |Medication | Se a pessoa toma medicamentos para a ansiedade|Booleana |
                |Therapy Sessions (per month) | Número de sessoes de terapia mensais | Numérica|
                |Recent Major Life Event | Informa se houve algum evento recente traumático | Booleana|
                |Diet Quality (1-10) | Qualidade da dieta da pessoa | Numérica|
                |Severity of Anxiety Attack (1-10) | Grau de severidade da crise de ansiedade|Numérica |
                ''')
    
    st.write('## Visualização da base de Dados')
    df = pd.read_csv('https://raw.githubusercontent.com/LucRib9/Projeto-Previsao_Ansiedade/refs/heads/main/anxiety_attack_dataset.csv', sep=',')
    st.write(df)
    dim = df.shape
    st.write(f'Número de linhas: {dim[0]}')
    st.write(f'Número de colunas: {dim[1]}')

    st.write('## Análise Descritiva Univariada')

    # Separação das colunas numéricas e categóricas do dataframe
    quant = df.select_dtypes(include=['int64', 'float64']).columns
    quali = df.select_dtypes(include=['object']).columns

    # Tabela com análises das variáveis
    st.write('### Variáveis Numéricas')
    st.write(df.describe())
    st.write('### Variáveis Categóricas')
    st.write(df.describe(include='object'))

    # Gráfico Univariada
    st.write('### Visualização Gráfica')
    
    univariada = pd.DataFrame({
    'variavel': ['ID', 'Age', 'Gender', 'Occupation', 'Sleep Hours',
                 'Physical Activity (hrs/week)', 'Caffeine Intake (mg/day)',
                 'Alcohol Consumption (drinks/week)', 'Smoking',
                 'Family History of Anxiety', 'Stress Level (1-10)',
                 'Heart Rate (bpm during attack)', 'Breathing Rate (breaths/min)',
                 'Sweating Level (1-5)', 'Dizziness', 'Medication',
                 'Therapy Sessions (per month)', 'Recent Major Life Event',
                 'Diet Quality (1-10)', 'Severity of Anxiety Attack (1-10)']
    })

    variavel = st.selectbox(
    'Qual variável você quer visualizar?',
    univariada['variavel'])

    if variavel in quant:
         # Gráfico boxplot e contagem da variável 'Age' 
        plt.rc('figure', figsize=(10, 4))
        fig, ax = plt.subplots(1, 2)
        # Calcula os gráficos
        sns.histplot(x=variavel, data=df, ax = ax[0])
        sns.boxplot(y=variavel, data=df, ax = ax[1])
        # Mostra os gráficos
        st.pyplot(fig)
    
    elif variavel in quali:
        # Prepara a figura
        plt.rc('figure', figsize=(2,2))
        fig, ax = plt.subplots(1, 1)
        # Prepara o gráfico de contagem
        sns.countplot(x=variavel, data=df, hue=variavel)
        # Imprime o gráfico
        st.pyplot(fig)

    # Analise Bivariada     
    st.write('## Análise Descritiva Bivariada')

    # Analise de variaveis quantitativas 
    st.write('### Matriz de Correlação')
    # Função que imprime a matriz de correlação
    @st.cache_data
    def correlacao(dados):
        plt.rc('figure', figsize=(20, 17))
        fig, ax = plt.subplots(1, 1)
        heatmap = sns.heatmap(dados, vmax=1, annot=True, cmap='coolwarm')
        st.pyplot(fig)
         
    # Imprime a matriz de correlação
    corr = df[quant].drop(columns='ID').corr()
    correlacao(corr)

    # Bivariada de qualitativas
    st.write('Visualização do Nível de Ansiedade em Função das Categóricas')
    bivariada = pd.DataFrame({
    'variavel': ['Gender', 'Occupation',
                 'Smoking', 'Family History of Anxiety', 
                 'Dizziness', 'Medication',
                 'Recent Major Life Event',
                 ]
    })
    escolhida = st.selectbox(
    'Qual variável qualitativa você quer escolher para analisar o nível de ansiedade?',
    bivariada['variavel'])

    # Gráfico boxplot e contagem da variável 'Age' 
    plt.rc('figure', figsize=(10, 4))
    fig, ax = plt.subplots(1, 1)
    sns.pointplot(x=escolhida, y='Severity of Anxiety Attack (1-10)', 
              data=df, hue=escolhida, dodge=True, errorbar=('ci',95))
    # Mostra os gráficos
    st.pyplot(fig)


    # Preparação e Modelagem dos Dados
    st.write('## Limpeza e Modelagem de Dados')
    st.markdown('''Nesta seção, será utilizado um pipeline para remover os valores nulos e outliers de cada 
                variável, além de transformar as converter as categóricas em booleanas e normalizar todas as variáveis.
                Após a utilização do pipeline, a base é dividida em treino e teste, sobre as quais é utilizado o modelo 
                de Regressão Logística para predizer o Nível de Ansiedade de cada indivíduo.''')

    quant = quant.drop(['ID','Severity of Anxiety Attack (1-10)'])

    # Pipeline de preprocessamento dos dados
    # Pipeline que corrige as numéricas
    numericas = Pipeline(steps=[
        ('nulos', SimpleImputer(strategy='mean')),
        ('outliers', RobustScaler())
    ])

    # Pipeline que converte as categoricas em dummies
    categoricas = Pipeline(steps=[
        ('nulos', SimpleImputer(strategy='most_frequent')),
        ('dummies', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    # Corrige todas as variáveis 
    conversor = ColumnTransformer([
        ('num', numericas, quant),
        ('cat', categoricas, quali)
    ])

    # Faz todo o preprocessamento 
    preprocessamento = Pipeline(steps=[
        ('conversor', conversor),
        ('pca', PCA(n_components=10, random_state=42))
    ])

    # Separa as variáveis explicativas da variável resposta
    X = df.drop(columns=['ID','Severity of Anxiety Attack (1-10)'])
    y = df['Severity of Anxiety Attack (1-10)'].copy()
    

    # Ajusta as variáveis explicativas ao Pipeline
    prep_ajuste = preprocessamento.fit_transform(X)
    X_prep = pd.DataFrame(data=prep_ajuste, columns=['PC' + str(i) for i in range(1,11)])

    # Separa a base em treino (70%) e teste (30%)
    X_train, X_test, y_train, y_test = train_test_split(X_prep, y, 
                                                    test_size=0.3, 
                                                    train_size=0.7, 
                                                    random_state=42)
    
    # Aplica a Regressão Logísitica sobre a base de treino
    reg = LogisticRegression(random_state=42)

    # Ajusta a base de treino à regressão Logística
    reg = reg.fit(X_train, y_train)
    
    # Faz a predição na base de teste
    y_pred = reg.predict(X_test)

    # Separa o df em treino e teste
    df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    train_size=0.7, 
                                                    random_state=42)

    # Mostra os resultados na base de teste
    df_X_test['Predict'] = y_pred.copy()
    st.write('### Resultados do ajuste do modelo na base de teste')
    st.write(df_X_test)

    # Avalia a acurácia do ajuste
    st.write(f'##### A acurácia do ajuste foi de: {100*reg.score(X_test, y_test):.2f}%.')
        
if __name__ == '__main__':
	main()
