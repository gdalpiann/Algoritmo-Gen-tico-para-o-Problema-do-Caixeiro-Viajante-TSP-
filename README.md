# Algoritmo Genético para o Problema do Caixeiro Viajante (TSP)

Este projeto foi desenvolvido para a disciplina de **Computação Evolutiva e Conexionista**. O objetivo é aplicar conceitos de Computação Evolutiva na resolução de um problema clássico de otimização combinatória: o **Problema do Caixeiro Viajante**, ou **Traveling Salesman Problem (TSP)**.

O TSP consiste em encontrar a menor rota possível que visite todas as cidades exatamente uma vez e retorne à cidade inicial. Como o problema pertence à classe dos problemas **NP-difíceis**, a busca exaustiva se torna inviável para instâncias maiores. Por isso, este trabalho utiliza um **Algoritmo Genético (AG)** para encontrar soluções aproximadas de boa qualidade.

---

## Objetivo do projeto

Implementar e analisar um Algoritmo Genético capaz de resolver instâncias do TSP de diferentes tamanhos, avaliando:

- convergência do algoritmo ao longo das gerações;
- tempo médio de execução;
- custo computacional, considerando tempo e uso de memória;
- qualidade da solução em comparação com ótimos conhecidos;
- escalabilidade conforme o número de cidades aumenta.

---

## Base de dados

As instâncias utilizadas foram obtidas da **TSPLIB**, uma biblioteca clássica de benchmarks para problemas de roteamento, especialmente o TSP.

Foram utilizadas instâncias simétricas no formato `.tsp`, com coordenadas bidimensionais e ótimo conhecido.

Instâncias utilizadas:

| Instância | Número de cidades | Ótimo conhecido | Categoria |
|---|---:|---:|---|
| `berlin52` | 52 | 7542 | Pequena |
| `eil76` | 76 | 538 | Pequena/Média |
| `kroA100` | 100 | 21282 | Média |
| `ch150` | 150 | 6528 | Média/Grande |
| `a280` | 280 | 2579 | Grande |

Os arquivos `.tsp` são arquivos de texto estruturados contendo metadados da instância e as coordenadas das cidades.

---

## Estrutura do projeto

```text
.
├── tsp_ga.py
├── README.md
├── tsplib/
│   ├── berlin52.tsp
│   ├── eil76.tsp
│   ├── kroA100.tsp
│   ├── ch150.tsp
│   └── a280.tsp
└── resultados/
    ├── convergencia_*.png
    ├── rota_*.png
    ├── runs_*.csv
    ├── resumo_geral.csv
    └── escalabilidade.png
````

A pasta `resultados/` é gerada automaticamente após a execução do script.

---

## Bibliotecas utilizadas

Conforme as restrições do trabalho, o núcleo do algoritmo utiliza apenas:

* `numpy`
* `pandas`
* `matplotlib`

Além disso, são utilizados módulos da biblioteca padrão do Python:

* `os`
* `time`
* `tracemalloc`

Esses módulos são usados para manipulação de caminhos, medição de tempo e estimativa de uso de memória.

---

## Como executar

Primeiro, certifique-se de que os arquivos `.tsp` estejam dentro da pasta `tsplib/`.

Depois, execute:

```bash
python tsp_ga.py
```

Ao final da execução, o programa cria automaticamente a pasta `resultados/`, contendo:

* arquivos `.csv` com os resultados de cada execução;
* gráficos de convergência;
* gráficos das melhores rotas encontradas;
* gráfico de escalabilidade;
* tabela resumo geral dos resultados.

---

## Funcionamento do algoritmo

O Algoritmo Genético implementado utiliza os principais componentes da Computação Evolutiva.

### Representação

Cada indivíduo da população representa uma rota do TSP.

O genótipo é uma permutação dos índices das cidades:

```text
[3, 0, 4, 1, 2]
```

Essa permutação representa a ordem em que as cidades são visitadas. Ao final, o caixeiro retorna para a cidade inicial, formando um ciclo.

---

### Função de fitness

A função de fitness corresponde à distância total da rota.

Como o objetivo do TSP é minimizar a distância percorrida, indivíduos com menor distância são considerados melhores.

A distância total é calculada somando as distâncias entre cidades consecutivas e adicionando o retorno da última cidade para a primeira.

---

### Seleção

Foi utilizada **seleção por torneio**.

Nesse método, um conjunto de indivíduos é escolhido aleatoriamente da população, e o melhor entre eles é selecionado como pai. A seleção por torneio foi escolhida por ser simples, eficiente e adequada para problemas de minimização.

---

### Crossover

O operador de crossover utilizado foi o **Order Crossover (OX)**.

Esse operador é adequado para problemas baseados em permutações, como o TSP, pois preserva a validade da rota e evita cidades repetidas ou ausentes.

---

### Mutação

Foi utilizada **mutação por inversão**.

Nesse operador, dois pontos da rota são escolhidos aleatoriamente, e o segmento entre eles é invertido. Essa mutação é adequada para o TSP porque pode melhorar a ordem local das cidades visitadas.

---

### Elitismo

O algoritmo utiliza elitismo, preservando os melhores indivíduos de uma geração para a próxima.

Isso garante que a melhor solução encontrada até o momento não seja perdida durante o processo evolutivo.

---

## Métricas avaliadas

Para cada instância, o algoritmo é executado múltiplas vezes. As seguintes métricas são coletadas:

### Convergência

São registrados, ao longo das gerações:

* melhor fitness da população;
* fitness médio da população.

Essas informações são usadas para gerar os gráficos de convergência.

### Tempo de execução

O tempo de cada execução é medido em segundos. Ao final, é calculado o tempo médio por instância.

### Custo computacional

O custo computacional é analisado por meio de:

* tempo médio de execução;
* pico de memória alocada durante a execução.

A memória é medida com o módulo `tracemalloc`.

### Qualidade da solução

A qualidade da solução é avaliada comparando a melhor distância encontrada com o ótimo conhecido da TSPLIB.

O gap percentual é calculado por:

```text
gap (%) = 100 * (distância encontrada - ótimo conhecido) / ótimo conhecido
```

Quanto menor o gap, mais próxima a solução está do ótimo conhecido.

---

## Parâmetros principais

Os parâmetros do Algoritmo Genético podem ser alterados diretamente no arquivo `tsp_ga.py`.

Configuração padrão:

```python
params = {
    "pop_size":        100,
    "n_generations":   500,
    "crossover_rate":  0.8,
    "mutation_rate":   0.2,
    "tournament_size": 5,
    "elite_size":      2,
    "n_runs":          10,
}
```

Esses parâmetros controlam:

* tamanho da população;
* número de gerações;
* taxa de crossover;
* taxa de mutação;
* tamanho do torneio;
* quantidade de indivíduos preservados por elitismo;
* número de execuções por instância.

---

## Saídas geradas

Após a execução, o projeto gera os seguintes arquivos:

### Resultados por instância

```text
runs_berlin52.csv
runs_eil76.csv
runs_kroA100.csv
runs_ch150.csv
runs_a280.csv
```

Esses arquivos contêm os resultados de cada execução individual.

### Tabela resumo

```text
resumo_geral.csv
```

Contém as estatísticas agregadas por instância, como:

* melhor distância encontrada;
* média das distâncias;
* desvio padrão;
* melhor gap;
* gap médio;
* tempo médio;
* memória média.

### Gráficos

São gerados gráficos de:

* convergência por instância;
* melhor rota encontrada;
* escalabilidade em função do número de cidades.

---

## Observações

O projeto não utiliza bibliotecas prontas de algoritmos genéticos, otimização ou resolução de TSP. Todos os operadores evolutivos foram implementados manualmente em Python.

As instâncias da TSPLIB foram utilizadas por possuírem tamanhos variados e ótimos conhecidos, permitindo uma análise mais consistente da qualidade das soluções e da escalabilidade do algoritmo.

Os arquivos `.tsp` são mantidos no formato original da TSPLIB, pois são arquivos de texto estruturados e representam diretamente a base de dados utilizada no projeto.

---

## Autores

* Gabriel de Oliveira Dalpian
* Alexandre Keiti Fukamati

