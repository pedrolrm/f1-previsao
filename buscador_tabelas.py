import requests
from bs4 import BeautifulSoup
import time

FIRST_RACE_OF_YEAR = {
    2014: 'Grande_Prêmio_da_Austrália',
    2015: 'Grande_Prêmio_da_Austrália',
    2016: 'Grande_Prêmio_da_Austrália',
    2017: 'Grande_Prêmio_da_Austrália',
    2018: 'Grande_Prêmio_da_Austrália',
    2019: 'Grande_Prêmio_da_Austrália',
    2020: 'Grande_Prêmio_da_Áustria',
    2021: 'Grande_Prêmio_do_Barém',
    2022: 'Grande_Prêmio_do_Barém',
    2023: 'Grande_Prêmio_do_Barém',
    2024: 'Grande_Prêmio_do_Barém',
}


for ano, gp in FIRST_RACE_OF_YEAR.items():
    url = f"https://pt.wikipedia.org/wiki/{gp}_de_{ano}"
    
    print(f"\n===================================================================")
    print(f"Analisando página para o ano de {ano}: {gp.replace('_', ' ')}")
    print(f"URL: {url}")
    print(f"===================================================================")
    
    try:
        response = requests.get(url, headers={'User-Agent': 'Meu-Projeto-de-Dados-F1-Explorer/2.0'})
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            todas_as_tabelas = soup.find_all('table', class_='wikitable')
            
            if not todas_as_tabelas:
                print("Nenhuma tabela foi encontrada nesta página")
                continue

            print(f"Encontradas {len(todas_as_tabelas)} tabelas")

            
            for i, tabela in enumerate(todas_as_tabelas):
                print(f"\n----------- Tabela {i + 1} -----------")
                titulo_secao = tabela.find_previous(['h2', 'h3', 'h4'])
                if titulo_secao and titulo_secao.span:
                    print(f"Contexto (Título da Seção): '{titulo_secao.span.get_text(strip=True)}'")
                else:
                    print("Contexto (Título da Seção): Não encontrado, tabela possivelmente aninhada.")

                cabecalhos_th = tabela.find_all('th')
                
                if cabecalhos_th:
                    lista_cabecalhos = [th.get_text(strip=True).replace('\n', ' ') for th in cabecalhos_th]
                    print("Estrutura das colunas (Cabeçalhos):")
                    print(lista_cabecalhos)
                else:
                    print("Esta tabela não possui cabeçalhos (<th>).")

        else:
            print(f"--> ERRO: Página não encontrada (Status: {response.status_code}).")
    
    except Exception as e:
        print(f"--> ERRO: Falha ao processar a página. Detalhes: {e}")
    
    time.sleep(1) 

print("\n===================================================================")
print("Exploração geral de tabelas finalizada.")