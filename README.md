# Análise e Previsão de Resultados da Fórmula 1

## Resumo

Este projeto implementa um pipeline de ponta a ponta para prever os resultados de corridas da Fórmula 1. O processo automatizado inclui web scraping, limpeza de dados, engenharia de features de momentum e treinamento de um modelo XGBoost. O modelo é treinado com dados históricos (2014-2023) para prever a posição final dos pilotos e é avaliado contra os resultados da temporada de 2024.

## Funcionalidades

- **Web Scraping Dinâmico**: Extrai dados de resultados de corridas e classificações da Wikipédia.
- **Pipeline de Limpeza Automatizado**: Processa e padroniza dados brutos para modelagem.
- **Engenharia de Features de Momentum**: Cria métricas baseadas no desempenho recente de um piloto (últimas 3 corridas).
- **Modelo Preditivo com XGBoost**: Utiliza XGBoost para prever a posição de chegada.
- **Validação Cruzada para Séries Temporais**: Emprega `TimeSeriesSplit` para uma validação cronologicamente correta.
- **Otimização de Hiperparâmetros**: Usa `RandomizedSearchCV` para encontrar a melhor configuração do modelo.
- **Avaliação de Performance de Corrida**: Mede a acurácia na previsão do vencedor, pódio e top 10.

## Tecnologias

- Python 3.9+
- Pandas & NumPy
- Requests & BeautifulSoup4
- Scikit-Learn
- XGBoost
- lxml

## Estrutura do Projeto

```
.
├── dados/
│   ├── brutos/
│   └── limpos/
├── src/
│   ├── scrapers/
│   ├── limpeza/
│   └── modelos/
│	└── modelo_momentum.py
│       └── previsao.py
├── .gitignore
├── buscador_tabelas.py
├── README.md
└── requirements.txt
```

## Instalação e Execução

### 1. Clone o repositório:

```bash
git clone [https://github.com/pedrolrm/f1-previsao.git](https://github.com/pedrolrm/f1-previsao.git)
cd f1-previsao
```

### 2. Crie um ambiente virtual e instale as dependências:

```bash
# Crie o ambiente
python -m venv venv

# Ative o ambiente
# Windows: .\venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Instale as dependências
pip install -r requirements.txt
```

### 3. Execute o pipeline:

Os scripts devem ser executados na ordem correta para gerar os dados e treinar o modelo.

```bash
# Passo 1: Coleta de Dados
python src/scrapers/scraper_corrida.py
python src/scrapers/scraper_quali.py

# Passo 2: Limpeza dos Dados
python src/limpeza/limpeza_corrida.py
python src/limpeza/limpeza_quali.py

# Passo 3: Treinamento, Previsão e Avaliação
python src/modelos/modelo_momentum.py
python src/modelos/previsao.py
```

## Contribuição

Contribuições são bem-vindas. Para contribuir, por favor, faça um fork do repositório, crie uma nova branch e abra um Pull Request com suas alterações.

## Licença

Este projeto está licenciado sob a Licença MIT.
