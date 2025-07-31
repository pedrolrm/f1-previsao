import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO
import time
import os

RACES_BY_YEAR = {
    'Grande_Prêmio_da_Austrália': [2014, 2015, 2016, 2017, 2018, 2019, 2022, 2023, 2024],
    'Grande_Prêmio_da_Malásia': [2014, 2015, 2016, 2017],
    'Grande_Prêmio_do_Barém': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'Grande_Prêmio_da_China': [2014, 2015, 2016, 2017, 2018, 2019, 2024],
    'Grande_Prêmio_da_Espanha': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'Grande_Prêmio_de_Mônaco': [2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024],
    'Grande_Prêmio_do_Canadá': [2014, 2015, 2016, 2017, 2018, 2019, 2022, 2023, 2024],
    'Grande_Prêmio_da_Áustria': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'Grande_Prêmio_da_Grã-Bretanha': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'Grande_Prêmio_da_Alemanha': [2014, 2016, 2018, 2019],
    'Grande_Prêmio_da_Hungria': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'Grande_Prêmio_da_Bélgica': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'Grande_Prêmio_da_Itália': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'Grande_Prêmio_de_Singapura': [2014, 2015, 2016, 2017, 2018, 2019, 2022, 2023, 2024],
    'Grande_Prêmio_do_Japão': [2014, 2015, 2016, 2017, 2018, 2019, 2022, 2023, 2024],
    'Grande_Prêmio_da_Rússia': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
    'Grande_Prêmio_dos_Estados_Unidos': [2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024],
    'Grande_Prêmio_do_Brasil': [2014, 2015, 2016, 2017, 2018, 2019],
    'Grande_Prêmio_de_São_Paulo': [2021, 2022, 2023, 2024],
    'Grande_Prêmio_do_México': [2015, 2016, 2017, 2018, 2019],
    'Grande_Prêmio_da_Cidade_do_México': [2021, 2022, 2023, 2024],
    'Grande_Prêmio_de_Abu_Dhabi': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'Grande_Prêmio_da_Europa': [2016],
    'Grande_Prêmio_do_Azerbaijão': [2017, 2018, 2019, 2021, 2022, 2023, 2024],
    'Grande_Prêmio_da_França': [2018, 2019, 2021, 2022],
    'Grande_Prêmio_da_Estíria': [2020, 2021],
    'Grande_Prêmio_do_70.º_Aniversário': [2020],
    'Grande_Prêmio_da_Toscana': [2020],
    'Grande_Prêmio_de_Eifel': [2020],
    'Grande_Prêmio_de_Portugal': [2020, 2021],
    'Grande_Prêmio_da_Emília-Romanha': [2020, 2021, 2022, 2024],
    'Grande_Prêmio_da_Turquia': [2020, 2021],
    'Grande_Prêmio_de_Sakhir': [2020],
    'Grande_Prêmio_dos_Países_Baixos': [2021, 2022, 2023, 2024],
    'Grande_Prêmio_do_Catar': [2021, 2023, 2024],
    'Grande_Prêmio_da_Arábia_Saudita': [2021, 2022, 2023, 2024],
    'Grande_Prêmio_de_Miami': [2022, 2023, 2024],
    'Grande_Prêmio_de_Las_Vegas': [2023, 2024]
}

lista_dfs_classificacao = []

for gp, anos_gp in RACES_BY_YEAR.items():
    for ano in anos_gp:
        url = f'https://pt.wikipedia.org/wiki/{gp.replace(" ", "_")}_de_{ano}'

        try:
            response = requests.get(url, headers={'User-Agent': 'Meu-Projeto-F1/Final-v2'})
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                tabelas = soup.find_all('table', class_='wikitable')
                tabela_certa = None

                for tabela in tabelas:
                    for sup in tabela.find_all('sup'):
                        sup.decompose()

                    cabecalhos = [th.get_text(strip=True) for th in tabela.find_all('th')]
                    if 'Q1' in cabecalhos and 'Q2' in cabecalhos and ('Piloto' in cabecalhos or 'Driver' in cabecalhos):
                        tabela_certa = tabela
                        break

                if tabela_certa:
                    df = pd.read_html(StringIO(str(tabela_certa)))[0]
                    df['Ano'] = ano
                    df['GP'] = gp
                    lista_dfs_classificacao.append(df)
        except:
            pass

        time.sleep(0.1)

if lista_dfs_classificacao:
    df_historico_bruto = pd.concat(lista_dfs_classificacao, ignore_index=True)
    diretorio_script = os.path.dirname(__file__)
    caminho_arquivo = os.path.normpath(os.path.join(diretorio_script, '..', '..', 'dados', 'brutos', 'f1_classificacao_bruto.csv'))
    os.makedirs(os.path.dirname(caminho_arquivo), exist_ok=True)
    df_historico_bruto.to_csv(caminho_arquivo, index=False, encoding='utf-8-sig')
    print(f"Arquivo salvo em: {caminho_arquivo}")
else:
    print("Nenhum dado de corrida foi coletado.")
