# Avaliação de sentimentos de comentários com IMDB

Código acadêmico desenvolvido para a disciplina de **Inteligência Artificial (IA)**, ministrada pelo **Prof. Dr. Gustavo Andrade**.

**Aluno:** Roberto Pereira de Freitas Neto.

O projeto utiliza Keras, TensorFlow e o dataset IMDB para classificar comentários em inglês como positivos ou negativos. A rede recebe vetores de presença de palavras com vocabulário limitado a 10.000 índices e produz uma estimativa de sentimento. O código também classifica os dez comentários de `comentarios.json` e calcula precisão, recall, F1 e acurácia.

## Executar no Google Colab

1. Abra o [Google Colab](https://colab.research.google.com/) e importe o arquivo [colab/notebook.ipynb](colab/notebook.ipynb).
2. Execute as células na ordem apresentada para instalar as dependências, definir as funções e preparar o dataset.
3. Execute as células de treinamento e avaliação para obter as previsões e métricas.

O notebook é independente: basta baixar o arquivo `.ipynb` e abri-lo no Google Colab. As funções dos módulos e os dez comentários de teste estão diretamente nas células, sem uso de `%%writefile`, imports locais ou leitura de `comentarios.json`. A lógica, os comentários e os parâmetros originais foram preservados, adaptando os imports e a entrada de dados para execução no notebook. Alterações futuras nos arquivos Python precisam ser reproduzidas no notebook.

Quando `text_model.keras` existe no diretório de execução, o código carrega esse modelo. Caso contrário, inicia o treinamento. Para reutilizar um modelo no Colab, envie esse arquivo para o diretório de execução antes da célula de treinamento. Baixe o modelo gerado se quiser guardá-lo após o encerramento da sessão.

## Executar localmente

Na raiz do projeto, com Python e as dependências instaladas:

```bash
pip install -r requirements.txt
python main.py
```

O primeiro uso precisa de acesso à internet para baixar o dataset IMDB e o índice de palavras.

## Arquivos do projeto

- `main.py`: carregamento dos dados, treinamento ou leitura do modelo e avaliação dos comentários.
- `modules/`: vetorização, construção da rede, preparação de texto, inferência e treinamento.
- `comentarios.json`: cinco comentários negativos e cinco positivos usados na atividade.
- `colab/notebook.ipynb`: versão em notebook para execução no Google Colab.
- [docs/metricas_exercicio.md](docs/metricas_exercicio.md): enunciado da atividade de avaliação.
- [docs/tabela_metricas.png](docs/tabela_metricas.png): imagem da tabela de métricas.

As métricas e a análise presentes no código fazem parte do exercício acadêmico. O notebook preserva a implementação original, incluindo suas limitações; a análise escrita no final de `main.py` não é recalculada automaticamente a cada execução.

---

O código foi totalmente implementado pelo Autor de positório (Eu =D), a documentação e versão do arquivo do colab foram transcritas utilizando ferramentas de IA Generativa.
