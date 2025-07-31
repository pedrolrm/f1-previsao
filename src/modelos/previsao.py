import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
    Converte um tempo em string no formato 'M:S.ms' ou 'S.ms' para um valor numérico em segundos.

    Args:
        tempo_str (str): Tempo representado como string.

    Returns:
        float: Tempo convertido para segundos, ou NaN se inválido.
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
    Realiza o pré-processamento e a engenharia de atributos no DataFrame de corridas da F1.

    Isso inclui limpeza de colunas, conversões numéricas, cálculo de gaps e momentum, 
    normalização de dados e codificação de variáveis categóricas.

    Args:
        df (pd.DataFrame): DataFrame original contendo os dados brutos combinados.

    Returns:
        pd.DataFrame: DataFrame preparado com atributos tratados e novas features.
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

    window_size = 3
    df_proc['momentum_pos_3r'] = df_proc.groupby('Piloto')['Pos_Corrida'].shift(1).rolling(window=window_size, min_periods=1).mean()
    df_proc['momentum_pts_3r'] = df_proc.groupby('Piloto')['Pontos_Ganhos'].shift(1).rolling(window=window_size, min_periods=1).mean()
    df_proc['momentum_quali_3r'] = df_proc.groupby('Piloto')['Pos_Quali'].shift(1).rolling(window=window_size, min_periods=1).mean()
    
    median_pos = df_proc['momentum_pos_3r'].median()
    median_pts = df_proc['momentum_pts_3r'].median()
    median_quali = df_proc['momentum_quali_3r'].median()
    df_proc.fillna({
        'momentum_pos_3r': median_pos,
        'momentum_pts_3r': median_pts,
        'momentum_quali_3r': median_quali
    }, inplace=True)
    
    df_proc.sort_values('race_id', inplace=True)
    df_proc.reset_index(drop=True, inplace=True)
    
    df_proc = pd.get_dummies(df_proc, columns=['Construtor_quali'], prefix='Construtor')
    df_proc.dropna(subset=['Pos_Corrida', 'Grid_Final'], inplace=True)
    return df_proc

print("1. Carregando e processando todos os dados (2014-2024)...")
df_completo = carregar_e_unir_dados()
if df_completo is None:
    exit()
df_processado = preparar_dados_final(df_completo)
print("Dados carregados e processados.")

df_treino = df_processado[df_processado['Ano'] < 2024].copy()
df_teste = df_processado[df_processado['Ano'] == 2024].copy()

print(f"\nTamanho do conjunto de treino: {len(df_treino)} registros")
print(f"Tamanho do conjunto de teste: {len(df_teste)} registros")

features_base = [
    'Pos_Quali', 'Grid_Final', 'Q1_s', 'Q2_s', 'Q3_s', 
    'Punicao_Grid', 'Gap_Q1_Q2', 'Gap_Q2_Q3',
    'momentum_pos_3r', 'momentum_pts_3r', 'momentum_quali_3r'
]
features_construtores = [col for col in df_processado.columns if col.startswith('Construtor_')]
features_finais = features_base + features_construtores

X_treino = df_treino[features_finais]
y_treino = df_treino['Pos_Corrida']

X_teste = df_teste[features_finais]
y_teste = df_teste['Pos_Corrida']

X_treino.columns = X_treino.columns.str.replace(r"\[|\]|<", "_", regex=True)
X_teste.columns = X_teste.columns.str.replace(r"\[|\]|<", "_", regex=True)

print("\n4. Iniciando a busca de hiperparâmetros no conjunto de treino (2014-2023)...")
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
    verbose=1,
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_treino, y_treino)
best_params = random_search.best_params_
print("\nMelhores hiperparâmetros encontrados:", best_params)

print("\n5. Treinando modelo final com os melhores parâmetros no conjunto de treino...")
modelo_final = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1, **best_params)
modelo_final.fit(X_treino, y_treino)
print("Modelo final treinado.")

print("\n6. Fazendo previsões para a temporada de 2024...")
previsoes_2024 = modelo_final.predict(X_teste)

print("\n--- AVALIAÇÃO DO MODELO NA TEMPORADA 2024 ---")

df_resultados = df_teste[['Ano', 'GP', 'Piloto', 'Pos_Corrida']].copy()
df_resultados['Posicao_Prevista'] = previsoes_2024
df_resultados['Posicao_Prevista_Arr'] = np.round(df_resultados['Posicao_Prevista'])

mae = mean_absolute_error(y_teste, previsoes_2024)
rmse = np.sqrt(mean_squared_error(y_teste, previsoes_2024))
r2 = r2_score(y_teste, previsoes_2024)

print("\n--- Métricas de Regressão ---")
print(f"Erro Médio Absoluto (MAE): {mae:.4f} (Em média, o modelo erra a posição por ~{mae:.1f} posições)")
print(f"Raiz do Erro Quadrático Médio (RMSE): {rmse:.4f}")
print(f"Coeficiente de Determinação (R²): {r2:.4f} ({r2:.2%})")

total_corridas = df_resultados['GP'].nunique()
acertos_vencedor = 0
acertos_podio = 0
total_membros_podio = 0
acertos_top10 = 0
total_membros_top10 = 0

for gp in df_resultados['GP'].unique():
    df_gp = df_resultados[df_resultados['GP'] == gp].copy()
    df_gp_real = df_gp.sort_values('Pos_Corrida')
    df_gp_previsto = df_gp.sort_values('Posicao_Prevista')
    
    vencedor_real = df_gp_real.iloc[0]['Piloto']
    vencedor_previsto = df_gp_previsto.iloc[0]['Piloto']
    if vencedor_real == vencedor_previsto:
        acertos_vencedor += 1
        
    podio_real = set(df_gp_real.head(3)['Piloto'])
    podio_previsto = set(df_gp_previsto.head(3)['Piloto'])
    acertos_podio += len(podio_real.intersection(podio_previsto))
    total_membros_podio += 3
    
    top10_real = set(df_gp_real.head(10)['Piloto'])
    top10_previsto = set(df_gp_previsto.head(10)['Piloto'])
    acertos_top10 += len(top10_real.intersection(top10_previsto))
    total_membros_top10 += 10

print("\n--- Métricas de Acurácia de Corrida ---")
print(f"Total de Corridas em 2024: {total_corridas}")
print(f"Acurácia do Vencedor: {acertos_vencedor}/{total_corridas} = {(acertos_vencedor/total_corridas):.2%}")
print(f"Acurácia de Pódio (membros corretos no pódio): {acertos_podio}/{total_membros_podio} = {(acertos_podio/total_membros_podio):.2%}")
print(f"Acurácia de Top 10 (membros corretos na zona de pontos): {acertos_top10}/{total_membros_top10} = {(acertos_top10/total_membros_top10):.2%}")

print("\n--- Exemplo de Previsão vs. Real (Top 5) ---")
exemplo_gp = df_resultados['GP'].unique()[0]
df_exemplo = df_resultados[df_resultados['GP'] == exemplo_gp]

print(f"\nResultado para: {exemplo_gp}")
print(df_exemplo[['Piloto', 'Pos_Corrida', 'Posicao_Prevista']].sort_values('Pos_Corrida').head(5).to_string(index=False))

print("\nPrevisão do Modelo:")
print(df_exemplo[['Piloto', 'Pos_Corrida', 'Posicao_Prevista']].sort_values('Posicao_Prevista').head(5).to_string(index=False))