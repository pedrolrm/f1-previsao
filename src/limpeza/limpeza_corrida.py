import pandas as pd
import numpy as np
import ast
import os

def get_clean_column_name(col):
    """
    Limpa nomes de colunas que estão formatados como tuplas em formato string.

    Esta função tenta interpretar a string `col` como uma tupla. Se a conversão for bem-sucedida 
    e o resultado for uma tupla com mais de um elemento, retorna o segundo elemento da tupla 
    (caso ele exista); caso contrário, retorna o primeiro. Se a conversão falhar ou o valor 
    não for uma tupla válida, retorna o valor original.

    Parâmetros:
    col (str): Nome da coluna, possivelmente representado como string de uma tupla.

    Retorna:
    str: Nome da coluna limpo (segundo elemento da tupla, se aplicável), ou o valor original.
    """
    try:
        col_tuple = ast.literal_eval(str(col))
        if isinstance(col_tuple, tuple) and len(col_tuple) > 1:
            return col_tuple[1] if col_tuple[1] else col_tuple[0]
    except (ValueError, SyntaxError):
        return col
    return col

diretorio_script = os.path.dirname(__file__)
ARQUIVO_BRUTO = os.path.normpath(os.path.join(diretorio_script, '..', '..', 'dados', 'brutos', 'f1_corrida_bruto.csv'))
ARQUIVO_LIMPO = os.path.normpath(os.path.join(diretorio_script, '..', '..', 'dados', 'limpos', 'f1_corrida_limpo.csv'))

print(f"Lendo dados brutos de: {ARQUIVO_BRUTO}")
df = pd.read_csv(ARQUIVO_BRUTO, header=0, dtype=str)

df.columns = [get_clean_column_name(c) for c in df.columns]

RENAME_MAP = {
    'Pos.': 'Pos',
    'Nu.': 'No', 'No.': 'No', 'Nº': 'No', 'N.º': 'No', 'N°': 'No', 'Num.': 'No', 'Não.': 'No',
    'Pilotos': 'Piloto', 'Driver': 'Piloto', 'Motorista': 'Piloto',
    'Construtora': 'Construtor', 'Equipe': 'Construtor', 'Constructor': 'Construtor',
    'Voltas\'': 'Voltas', 'Laps': 'Voltas',
    'Tempo/retirada': 'Tempo/Retirado', 'Tempo/Retirada': 'Tempo/Retirado', 'Tempo/Aposentado': 'Tempo/Retirado',
    'Tempo/Abandono': 'Tempo/Retirado', 'Tempo/Diferença': 'Tempo/Retirado', 'Time/Retired': 'Tempo/Retirado', 'Tempo': 'Tempo/Retirado',
    'Points': 'Pontos', 'Pts.': 'Pontos',
    'Grade': 'Grid', 'Grid final': 'Grid', 'Final grid': 'Grid', 'Grid 1': 'Grid', 'Grid 2': 'Grid'
}
df.rename(columns=RENAME_MAP, inplace=True)
df = df.T.groupby(level=0).first().T

COLUNAS_DESEJADAS = ['Ano', 'GP', 'Pos', 'No', 'Piloto', 'Construtor', 'Voltas', 'Tempo/Retirado', 'Pontos', 'Grid']
for col in COLUNAS_DESEJADAS:
    if col not in df.columns:
        df[col] = np.nan
df = df[COLUNAS_DESEJADAS]

df.dropna(subset=['Piloto', 'Pos'], how='all', inplace=True)
df = df[~df['Piloto'].str.contains('Piloto|Driver', na=False)]

df['No'] = df.groupby(['Ano', 'Piloto'])['No'].transform(lambda x: x.ffill().bfill())
df['Construtor'] = df.groupby(['Ano', 'Piloto'])['Construtor'].transform(lambda x: x.ffill().bfill())

df['Ano'] = pd.to_numeric(df['Ano'], errors='coerce').astype('Int64').astype(str)
df['Ano'] = df['Ano'].replace('<NA>', np.nan)

os.makedirs(os.path.dirname(ARQUIVO_LIMPO), exist_ok=True)
df.to_csv(ARQUIVO_LIMPO, index=False, encoding='utf-8-sig')

print(f"Arquivo limpo salvo em: {ARQUIVO_LIMPO}")
