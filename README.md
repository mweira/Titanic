# Titanic Kaggle - EDA e Machine Learning

Projeto para explorar o dataset Titanic do Kaggle, comparar modelos de machine learning e gerar um arquivo de submissao no formato esperado pela competicao.

## Arquivos principais

- `titanic_study.ipynb`: notebook principal, com EDA, graficos, engenharia de atributos, comparacao de modelos e geracao da submissao.
- `titanic_models_explained.ipynb`: notebook didatico com explicacoes visuais detalhadas de cada modelo.
- `titanic_model.py`: script rapido para treinar os modelos, escolher o melhor por validacao cruzada e gerar `submission.csv`.
- `titanic_eda.py`: script separado para rodar apenas a analise exploratoria com graficos.
- `submission.csv`: arquivo gerado para envio ao Kaggle.

## Modelos utilizados

O projeto compara tres modelos de classificacao:

- `LogisticRegression`: modelo linear simples, bom como baseline interpretavel.
- `RandomForestClassifier`: ensemble de arvores, robusto para relacoes nao lineares e bom para importancia de variaveis.
- `GradientBoostingClassifier`: ensemble sequencial de arvores, geralmente forte em datasets tabulares pequenos.

Todos os modelos usam a mesma preparacao de dados:

- imputacao de valores faltantes;
- padronizacao de variaveis numericas;
- one-hot encoding de variaveis categoricas;
- validacao cruzada com 5 folds usando acuracia.

## Features criadas

- `Title`: titulo extraido do nome do passageiro, como `Mr`, `Mrs`, `Miss`, `Officer` e `Royalty`.
- `FamilySize`: soma de `SibSp`, `Parch` e o proprio passageiro.
- `IsAlone`: indica se o passageiro viajava sozinho.
- `Deck`: primeira letra da cabine, quando disponivel.

## Como usar

Verifique se `train.csv` e `test.csv` estao na raiz do workspace.

Instale as dependencias:

```powershell
python -m pip install -r Titanic\requirements.txt
```

Para rodar o notebook, abra:

```text
Titanic\titanic_study.ipynb
```

Para estudar os modelos com explicacoes visuais, abra:

```text
Titanic\titanic_models_explained.ipynb
```

Para rodar apenas a analise exploratoria:

```powershell
python Titanic\titanic_eda.py
```

Para treinar os modelos e gerar a submissao:

```powershell
python Titanic\titanic_model.py
```

O arquivo final sera salvo em:

```text
Titanic\submission.csv
```
