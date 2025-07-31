import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import numpy as np
import xgboost as xgb
import os

def carregar_e_unir_dados():
    """
    Carrega os dados de classificação e corrida de F1 de arquivos CSV,
    os une e os prepara para o processamento.

    A função localiza os arquivos 'f1_classificacao_limpo.csv' e
    'f1_corrida_limpo.csv' em um diretório de dados estruturado,
    lida com possíveis erros de arquivo não encontrado e realiza um merge
    dos dois DataFrames com base nas colunas 'Ano', 'GP' e 'Piloto'.
    As colunas de posição e pontos são renomeadas para evitar conflitos.

    Returns:
        pd.DataFrame or None: Um DataFrame do Pandas contendo os dados unidos
                              se os arquivos forem carregados com sucesso,
                              caso contrário, retorna None.
    """
    diretorio_script = os.path.dirname(__file__)
    caminho_quali = os.path.normpath(os.path.join(diretorio_script, '..', '..', 'dados', 'limpos', 'f1_classificacao_limpo.csv'))
    caminho_corrida = os.path.normpath(os.path.join(diretorio_script, '..', '..', 'dados', 'limpos', 'f1_corrida_limpo.csv'))

    try:
        df_quali = pd.read_csv(caminho_quali)
        df_corrida = pd.read_csv(caminho_corrida)
    except FileNotFoundError:
        print("ERRO: Arquivos de dados limpos não encontrados!")
        print(f"Verifique se '{os.path.basename(caminho_quali)}' e '{os.path.basename(caminho_corrida)}' existem na pasta 'dados/limpos'.")
        return None

    df_quali = df_quali.rename(columns={'Pos': 'Pos_Quali', 'Grid': 'Grid_Final'})
    df_corrida = df_corrida.rename(columns={'Pos': 'Pos_Corrida', 'Pontos': 'Pontos_Ganhos'})
    
    df_completo = pd.merge(df_quali, df_corrida, on=['Ano', 'GP', 'Piloto'], suffixes=('_quali', '_corrida'))
    
    return df_completo

def tempo_para_segundos(tempo_str):
    """
    Converte uma string de tempo no formato 'M:S.ms' ou 'S.ms' para segundos.

    Args:
        tempo_str (str): A string de tempo a ser convertida. Pode conter
                         minutos, segundos e milissegundos.

    Returns:
        float or np.nan: O tempo total em segundos como um número de ponto
                         flutuante. Retorna np.nan se a entrada for inválida,
                         nula ou não for uma string.
    """
    if pd.isna(tempo_str) or not isinstance(tempo_str, str):
        return np.nan
    partes = tempo_str.replace(':', '.').split('.')
    try:
        if len(partes) == 3:
            return (int(partes[0]) * 60) + int(partes[1]) + (int(partes[2]) / 1000)
        elif len(partes) == 2:
            return int(partes[0]) + (int(partes[1]) / 1000)
    except (ValueError, IndexError):
        return np.nan
    return np.nan

def preparar_dados_final(df):
    """
    Executa a engenharia de features e o pré-processamento final no DataFrame.

    As etapas incluem:
    1.  Conversão de colunas numéricas.
    2.  Conversão de tempos de qualificação para segundos.
    3.  Criação de features como 'Punicao_Grid' e gaps de tempo.
    4.  Criação de features de momentum (média móvel de resultados anteriores).
    5.  Aplicação de one-hot encoding para construtores.
    6.  Tratamento de valores ausentes.

    Args:
        df (pd.DataFrame): O DataFrame com os dados brutos unidos.

    Returns:
        pd.DataFrame: O DataFrame processado e pronto para o treinamento do modelo.
    """
    df_proc = df.copy()
    if 'Construtor_corrida' in df_proc.columns:
        df_proc.drop(columns=['Construtor_corrida'], inplace=True)
    
    for col_num in ['Pos_Quali', 'Grid_Final', 'Pos_Corrida', 'Pontos_Ganhos']:
        if col_num in df_proc.columns:
            df_proc[col_num] = pd.to_numeric(df_proc[col_num], errors='coerce')
            
    for col_tempo in ['Q1', 'Q2', 'Q3']:
        df_proc[f'{col_tempo}_s'] = df_proc[col_tempo].apply(tempo_para_segundos)
        
    df_proc['Punicao_Grid'] = df_proc['Grid_Final'] - df_proc['Pos_Quali']
    df_proc['Gap_Q1_Q2'] = df_proc['Q1_s'] - df_proc['Q2_s']
    df_proc['Gap_Q2_Q3'] = df_proc['Q2_s'] - df_proc['Q3_s']
    
    df_proc.fillna({'Q1_s': 999, 'Q2_s': 999, 'Q3_s': 999, 'Gap_Q1_Q2': 0, 'Gap_Q2_Q3': 0}, inplace=True)
    
    df_proc['race_id'] = df_proc.groupby(['Ano', 'GP']).ngroup()
    df_proc.sort_values(['Piloto', 'race_id'], inplace=True)

    print("Criando features de momentum...")
    window_size = 3
    df_proc['momentum_pos_3r'] = df_proc.groupby('Piloto')['Pos_Corrida'].shift(1).rolling(window=window_size, min_periods=1).mean()
    df_proc['momentum_pts_3r'] = df_proc.groupby('Piloto')['Pontos_Ganhos'].shift(1).rolling(window=window_size, min_periods=1).mean()
    df_proc['momentum_quali_3r'] = df_proc.groupby('Piloto')['Pos_Quali'].shift(1).rolling(window=window_size, min_periods=1).mean()
    
    df_proc.fillna({
        'momentum_pos_3r': df_proc['momentum_pos_3r'].median(),
        'momentum_pts_3r': df_proc['momentum_pts_3r'].median(),
        'momentum_quali_3r': df_proc['momentum_quali_3r'].median()
    }, inplace=True)
    
    df_proc.sort_values('race_id', inplace=True)
    df_proc.reset_index(drop=True, inplace=True)
    
    df_proc = pd.get_dummies(df_proc, columns=['Construtor_quali'], prefix='Construtor')
    df_proc.dropna(subset=['Pos_Corrida', 'Grid_Final'], inplace=True)
    return df_proc


print("Iniciando a OTIMIZAÇÃO DE HIPERPARÂMETROS")

df_completo = carregar_e_unir_dados()
if df_completo is None:
    exit()
df_processado = preparar_dados_final(df_completo)
print("Dados carregados e todas as features criadas.")

features_base = [
    'Pos_Quali', 'Grid_Final', 'Q1_s', 'Q2_s', 'Q3_s', 
    'Punicao_Grid', 'Gap_Q1_Q2', 'Gap_Q2_Q3',
    'momentum_pos_3r', 'momentum_pts_3r', 'momentum_quali_3r'
]
features_construtores = [col for col in df_processado.columns if col.startswith('Construtor_')]
features_finais = features_base + features_construtores

X = df_processado[features_finais]
y = df_processado['Pos_Corrida']
X.columns = X.columns.str.replace(r"\[|\]|<", "_", regex=True)

param_dist = {
    'n_estimators': [100, 300, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2]
}

tss = TimeSeriesSplit(n_splits=5)
modelo_base = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

random_search = RandomizedSearchCV(
    estimator=modelo_base,
    param_distributions=param_dist,
    n_iter=50,
    scoring='r2',
    cv=tss,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

print("\nIniciando a busca pelos melhores hiperparâmetros com o dataset final")
random_search.fit(X, y)

print("\n--- RESULTADOS DA OTIMIZAÇÃO FINAL ---")
print("Busca concluída!")
print(f"\nO melhor R² médio encontrado foi: {random_search.best_score_:.4f} ({random_search.best_score_:.2%})")
print("\nA melhor combinação de hiperparâmetros encontrada foi:")
print(random_search.best_params_)
print("---------------------------------------")