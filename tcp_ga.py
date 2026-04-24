"""
tsp_ga.py - Algoritmo Genético para o Problema do Caixeiro Viajante (TSP)
===========================================================================
Trabalho Prático: Otimização Evolutiva em Problemas Complexos
Disciplina: Computação Evolutiva e Conexionista

Bibliotecas utilizadas (conforme restrição do professor):
  - numpy:      operações matemáticas e manipulação de arrays
  - pandas:     organização de dados e tabelas de resultados
  - matplotlib: visualização de resultados e gráficos

Módulos da biblioteca padrão do Python (não são bibliotecas externas):
  - time:       medição de tempo de execução
  - os:         manipulação de caminhos de arquivo
  - tracemalloc: medição de consumo de memória

Instâncias: TSPLIB (formato EUC_2D)
  https://github.com/mastqe/tsplib/tree/master
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import tracemalloc


# ============================================================
# 1. LEITURA DE ARQUIVOS TSPLIB
# ============================================================
def read_tsp_file(filepath):
    """
    Lê um arquivo .tsp no formato TSPLIB e extrai as coordenadas.

    O formato TSPLIB é um padrão acadêmico para instâncias do TSP.
    Um arquivo .tsp contém um cabeçalho com metadados (nome, dimensão,
    tipo de distância) seguido de uma seção NODE_COORD_SECTION com as
    coordenadas (id, x, y) de cada cidade.

    Parâmetros
    ----------
    filepath : str
        Caminho para o arquivo .tsp.

    Retorna
    -------
    name : str
        Nome da instância.
    dimension : int
        Número de cidades.
    coords : np.ndarray, shape (n, 2)
        Coordenadas (x, y) de cada cidade.
    edge_weight_type : str
        Tipo de cálculo de distância (ex: "EUC_2D").
    """
    name = ""
    dimension = 0
    edge_weight_type = "EUC_2D"
    coords = []
    reading_coords = False

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            # --- Cabeçalho ---
            if line.startswith("NAME"):
                name = line.split(":")[1].strip()
            elif line.startswith("DIMENSION"):
                dimension = int(line.split(":")[1].strip())
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                edge_weight_type = line.split(":")[1].strip()
            elif line.startswith("NODE_COORD_SECTION"):
                reading_coords = True
                continue
            elif line in ("EOF", "DISPLAY_DATA_SECTION", "DEMAND_SECTION",
                          "DEPOT_SECTION", "EDGE_WEIGHT_SECTION"):
                reading_coords = False
                continue

            # --- Coordenadas ---
            if reading_coords and line:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        coords.append([x, y])
                    except ValueError:
                        reading_coords = False

    coords = np.array(coords)

    if len(coords) != dimension:
        print(f"  [Aviso] Esperado {dimension} cidades, lido {len(coords)}.")
        dimension = len(coords)

    return name, dimension, coords, edge_weight_type


# ============================================================
# 2. CÁLCULO DA MATRIZ DE DISTÂNCIAS
# ============================================================
def compute_distance_matrix(coords, edge_weight_type="EUC_2D"):
    """
    Calcula a matriz de distâncias entre todas as cidades.

    Para o tipo EUC_2D (Euclidiana 2D), a distância entre duas cidades
    é calculada como:
        d(i,j) = nint( sqrt( (xi - xj)^2 + (yi - yj)^2 ) )
    onde nint() arredonda para o inteiro mais próximo.

    Parâmetros
    ----------
    coords : np.ndarray, shape (n, 2)
        Coordenadas das cidades.
    edge_weight_type : str
        Tipo de distância. Apenas "EUC_2D" é suportado.

    Retorna
    -------
    dist_matrix : np.ndarray, shape (n, n)
        Matriz simétrica de distâncias inteiras.
    """
    if edge_weight_type != "EUC_2D":
        raise ValueError(
            f"Tipo de distância '{edge_weight_type}' não suportado. "
            f"Use apenas instâncias EUC_2D da TSPLIB."
        )

    # Broadcasting: diff[i, j, :] = coords[i] - coords[j]
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]

    # Distância euclidiana arredondada para inteiro
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    dist_matrix = np.rint(dist_matrix).astype(int)

    return dist_matrix


# ============================================================
# 3. REPRESENTAÇÃO E FUNÇÃO DE FITNESS
# ============================================================
#
# GENÓTIPO:  permutação de inteiros [0, 1, 2, ..., n-1]
#            Cada inteiro representa o índice de uma cidade.
#            Exemplo para 5 cidades: [3, 0, 4, 1, 2]
#
# FENÓTIPO:  rota completa que visita todas as cidades exatamente
#            uma vez e retorna à cidade de origem.
#            Exemplo: 3 → 0 → 4 → 1 → 2 → 3
#
# FITNESS:   distância total da rota (MINIMIZAR).
#            Quanto MENOR a distância, MELHOR a solução.
# ============================================================

def calculate_tour_distance(tour, dist_matrix):
    """
    Calcula a distância total de um tour (ciclo hamiltoniano).

    A distância é a soma de todas as arestas do ciclo:
        dist(tour[0], tour[1]) + dist(tour[1], tour[2]) + ...
        + dist(tour[n-1], tour[0])

    Parâmetros
    ----------
    tour : np.ndarray
        Permutação representando a ordem de visita.
    dist_matrix : np.ndarray
        Matriz de distâncias.

    Retorna
    -------
    float
        Distância total da rota.
    """
    # np.roll desloca o array, criando pares (tour[i], tour[i+1])
    next_cities = np.roll(tour, -1)
    return int(np.sum(dist_matrix[tour, next_cities]))


def evaluate_population(population, dist_matrix):
    """Avalia todos os indivíduos da população e retorna vetor de fitness."""
    return np.array([
        calculate_tour_distance(ind, dist_matrix) for ind in population
    ])


# ============================================================
# 4. INICIALIZAÇÃO DA POPULAÇÃO
# ============================================================
def initialize_population(pop_size, num_cities):
    """
    Gera a população inicial como permutações aleatórias.

    Cada indivíduo é uma permutação válida de [0, 1, ..., n-1],
    garantindo que toda solução inicial seja viável (visita cada
    cidade exatamente uma vez).

    Parâmetros
    ----------
    pop_size : int
        Número de indivíduos na população.
    num_cities : int
        Número de cidades.

    Retorna
    -------
    list[np.ndarray]
        Lista de permutações aleatórias.
    """
    return [np.random.permutation(num_cities) for _ in range(pop_size)]


# ============================================================
# 5. SELEÇÃO POR TORNEIO
# ============================================================
def tournament_selection(population, fitness, tournament_size):
    """
    Seleciona um indivíduo usando seleção por torneio.

    Funcionamento:
    1. Sorteia 'tournament_size' indivíduos aleatoriamente da população.
    2. Retorna o indivíduo com MENOR fitness (menor distância = melhor).

    A pressão seletiva é controlada pelo tamanho do torneio:
    - Torneio pequeno (2-3): menor pressão, mais exploração.
    - Torneio grande (5-7):  maior pressão, mais exploitação.

    Parâmetros
    ----------
    population : list[np.ndarray]
        População atual.
    fitness : np.ndarray
        Valores de fitness correspondentes.
    tournament_size : int
        Número de competidores no torneio.

    Retorna
    -------
    np.ndarray
        Cópia do indivíduo vencedor.
    """
    # Sorteia índices dos competidores (sem repetição)
    candidates = np.random.choice(len(population), tournament_size, replace=False)

    # Vencedor: menor fitness (menor distância)
    winner = candidates[np.argmin(fitness[candidates])]

    return population[winner].copy()


# ============================================================
# 6. CROSSOVER - Order Crossover (OX)
# ============================================================
def order_crossover(parent1, parent2):
    """
    Implementa o Order Crossover (OX) para permutações.

    O OX é projetado especificamente para o TSP porque preserva
    a ordem relativa das cidades, respeitando a estrutura de permutação.

    Funcionamento:
    1. Seleciona dois pontos de corte aleatórios: start e end.
    2. Copia o segmento [start, end] do pai1 para o filho.
    3. Preenche as posições restantes com cidades do pai2,
       na ordem em que aparecem, começando após 'end' de forma circular.

    Exemplo:
        pai1 = [1, 2, |3, 4, 5|, 6, 7]    (segmento: 3,4,5)
        pai2 = [5, 3,  6, 7, 2,  1, 4]
        Cidades faltando (na ordem do pai2): 6, 7, 2, 1
        Preenchimento circular após posição 'end':
        filho = [2, 1, 3, 4, 5, 6, 7]

    Parâmetros
    ----------
    parent1, parent2 : np.ndarray
        Dois pais selecionados.

    Retorna
    -------
    np.ndarray
        Um filho resultante do crossover.
    """
    n = len(parent1)

    # Dois pontos de corte (start < end)
    start, end = sorted(np.random.choice(n, 2, replace=False))

    # Filho começa vazio (-1)
    child = np.full(n, -1, dtype=int)

    # Copia segmento do pai1
    child[start:end + 1] = parent1[start:end + 1]

    # Conjunto das cidades já inseridas (busca O(1))
    in_child = set(child[start:end + 1])

    # Cidades do pai2 ainda não presentes no filho, na ordem do pai2
    remaining = [city for city in parent2 if city not in in_child]

    # Preenche posições vazias de forma circular a partir de (end+1)
    pos = (end + 1) % n
    for city in remaining:
        child[pos] = city
        pos = (pos + 1) % n

    return child


# ============================================================
# 7. MUTAÇÃO POR INVERSÃO
# ============================================================
def inversion_mutation(individual):
    """
    Mutação por inversão: reverte um segmento aleatório do tour.

    Escolhe dois pontos aleatórios e inverte a ordem das cidades
    entre eles. A permutação continua válida após a inversão.

    A inversão é especialmente eficaz para o TSP porque pode
    desfazer cruzamentos de arestas (2-opt move), o que tende
    a encurtar a rota.

    Exemplo:
        antes:  [1, 2, |3, 4, 5|, 6, 7]
        depois: [1, 2, |5, 4, 3|, 6, 7]

    Parâmetros
    ----------
    individual : np.ndarray
        Indivíduo (permutação) a ser mutado.

    Retorna
    -------
    np.ndarray
        Nova permutação com segmento invertido.
    """
    mutant = individual.copy()
    n = len(mutant)

    # Dois pontos de corte
    start, end = sorted(np.random.choice(n, 2, replace=False))

    # Inverte o segmento
    mutant[start:end + 1] = mutant[start:end + 1][::-1]

    return mutant


# ============================================================
# 8. ALGORITMO GENÉTICO PRINCIPAL
# ============================================================
def genetic_algorithm(dist_matrix, params):
    """
    Executa o Algoritmo Genético completo para o TSP.

    Ciclo evolutivo por geração:
    1. ELITISMO: copia os melhores indivíduos diretamente.
    2. SELEÇÃO: escolhe pais via torneio.
    3. CROSSOVER (OX): gera filhos a partir dos pais.
    4. MUTAÇÃO (inversão): aplica perturbações nos filhos.
    5. SUBSTITUIÇÃO: nova população substitui a anterior (geracional + elitismo).

    Parâmetros
    ----------
    dist_matrix : np.ndarray
        Matriz de distâncias (n x n).
    params : dict
        Hiperparâmetros do AG:
        - pop_size (int): tamanho da população.
        - n_generations (int): número máximo de gerações.
        - crossover_rate (float): probabilidade de crossover [0, 1].
        - mutation_rate (float): probabilidade de mutação [0, 1].
        - tournament_size (int): tamanho do torneio.
        - elite_size (int): número de indivíduos preservados por elitismo.

    Retorna
    -------
    best_tour : np.ndarray
        Melhor tour encontrado ao longo de toda a execução.
    best_distance : int
        Distância do melhor tour.
    history : dict
        Histórico com "best" e "avg" fitness por geração.
    """
    num_cities = dist_matrix.shape[0]

    # Extrai parâmetros com valores padrão
    pop_size        = params.get("pop_size", 100)
    n_generations   = params.get("n_generations", 500)
    crossover_rate  = params.get("crossover_rate", 0.8)
    mutation_rate   = params.get("mutation_rate", 0.2)
    tournament_size = params.get("tournament_size", 5)
    elite_size      = params.get("elite_size", 2)

    # --- Inicialização ---
    population = initialize_population(pop_size, num_cities)
    fitness = evaluate_population(population, dist_matrix)

    # Histórico de convergência
    best_history = []
    avg_history = []

    # Melhor solução encontrada (global, ao longo de todas as gerações)
    best_idx = np.argmin(fitness)
    global_best_tour = population[best_idx].copy()
    global_best_distance = fitness[best_idx]

    # --- Loop evolutivo ---
    for gen in range(n_generations):
        new_population = []

        # ---- ELITISMO ----
        # Copia os 'elite_size' melhores diretamente para a próxima geração.
        # Isso garante que a melhor solução nunca se perca.
        elite_indices = np.argsort(fitness)[:elite_size]
        for idx in elite_indices:
            new_population.append(population[idx].copy())

        # ---- GERAÇÃO DE NOVOS INDIVÍDUOS ----
        while len(new_population) < pop_size:
            # Seleção de dois pais por torneio
            parent1 = tournament_selection(population, fitness, tournament_size)
            parent2 = tournament_selection(population, fitness, tournament_size)

            # Crossover (OX)
            if np.random.random() < crossover_rate:
                child1 = order_crossover(parent1, parent2)
                child2 = order_crossover(parent2, parent1)
            else:
                # Sem crossover: filhos são cópias dos pais
                child1 = parent1.copy()
                child2 = parent2.copy()

            # Mutação (inversão)
            if np.random.random() < mutation_rate:
                child1 = inversion_mutation(child1)
            if np.random.random() < mutation_rate:
                child2 = inversion_mutation(child2)

            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        # ---- SUBSTITUIÇÃO GERACIONAL ----
        population = new_population[:pop_size]
        fitness = evaluate_population(population, dist_matrix)

        # Atualiza melhor solução global
        gen_best_idx = np.argmin(fitness)
        if fitness[gen_best_idx] < global_best_distance:
            global_best_distance = fitness[gen_best_idx]
            global_best_tour = population[gen_best_idx].copy()

        # Registra histórico
        best_history.append(global_best_distance)
        avg_history.append(np.mean(fitness))

    history = {"best": best_history, "avg": avg_history}

    return global_best_tour, global_best_distance, history


# ============================================================
# 9. EXECUÇÃO DO EXPERIMENTO (MÚLTIPLAS RODADAS)
# ============================================================
def run_experiment(filepath, params, optimal_distance=None):
    """
    Executa o AG múltiplas vezes para uma instância TSPLIB.

    Cada execução usa uma semente aleatória diferente, garantindo
    relevância estatística dos resultados. São coletadas métricas de:
    - Qualidade da solução (distância, gap percentual)
    - Tempo de execução
    - Consumo de memória

    Parâmetros
    ----------
    filepath : str
        Caminho para o arquivo .tsp.
    params : dict
        Parâmetros do AG (inclui "n_runs" para nº de execuções).
    optimal_distance : int or None
        Distância ótima conhecida para cálculo do gap.

    Retorna
    -------
    df_results : pd.DataFrame
        Resultados detalhados de cada execução.
    stats : dict
        Estatísticas agregadas (média, desvio, melhor, gap).
    all_histories : list[dict]
        Históricos de convergência de todas as execuções.
    best_overall_tour : np.ndarray
        Melhor tour encontrado entre todas as execuções.
    coords : np.ndarray
        Coordenadas das cidades (para plotagem).
    name : str
        Nome da instância.
    """
    # Leitura da instância
    name, dimension, coords, edge_weight_type = read_tsp_file(filepath)
    dist_matrix = compute_distance_matrix(coords, edge_weight_type)

    n_runs = params.get("n_runs", 10)

    print(f"\n{'=' * 60}")
    print(f"  Instância: {name} | Cidades: {dimension}")
    if optimal_distance:
        print(f"  Ótimo conhecido: {optimal_distance:,}")
    print(f"  Execuções: {n_runs}")
    print(f"{'=' * 60}")

    results = []
    all_histories = []
    best_overall_tour = None
    best_overall_distance = float("inf")

    for run in range(n_runs):
        # Medição de memória
        tracemalloc.start()
        start_time = time.time()

        # Executa o AG
        best_tour, best_distance, history = genetic_algorithm(dist_matrix, params)

        elapsed_time = time.time() - start_time
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Gap percentual
        gap = None
        if optimal_distance:
            gap = ((best_distance - optimal_distance) / optimal_distance) * 100

        results.append({
            "run":            run + 1,
            "best_distance":  best_distance,
            "time_seconds":   round(elapsed_time, 2),
            "peak_memory_mb": round(peak_memory / (1024 * 1024), 2),
            "gap_percent":    round(gap, 2) if gap is not None else None,
        })

        all_histories.append(history)

        if best_distance < best_overall_distance:
            best_overall_distance = best_distance
            best_overall_tour = best_tour.copy()

        # Progresso no terminal
        gap_str = f"Gap: {gap:.2f}%" if gap is not None else ""
        print(f"  Run {run + 1:2d}/{n_runs}  |  "
              f"Distância: {best_distance:>10,}  |  "
              f"{gap_str:>14s}  |  "
              f"Tempo: {elapsed_time:.2f}s")

    # --- Monta DataFrame de resultados ---
    df_results = pd.DataFrame(results)

    # --- Estatísticas agregadas ---
    stats = {
        "instance":       name,
        "dimension":      dimension,
        "optimal":        optimal_distance,
        "best_found":     int(df_results["best_distance"].min()),
        "worst_found":    int(df_results["best_distance"].max()),
        "mean_distance":  round(df_results["best_distance"].mean(), 2),
        "std_distance":   round(df_results["best_distance"].std(), 2),
        "mean_time":      round(df_results["time_seconds"].mean(), 2),
        "mean_memory_mb": round(df_results["peak_memory_mb"].mean(), 2),
        "best_gap":       round(df_results["gap_percent"].min(), 2)  if optimal_distance else None,
        "mean_gap":       round(df_results["gap_percent"].mean(), 2) if optimal_distance else None,
    }

    print(f"\n  --- Resumo {name} ---")
    print(f"  Melhor: {stats['best_found']:,}  |  "
          f"Média: {stats['mean_distance']:,.0f} ± {stats['std_distance']:,.0f}")
    if optimal_distance:
        print(f"  Melhor gap: {stats['best_gap']:.2f}%  |  "
              f"Gap médio: {stats['mean_gap']:.2f}%")
    print(f"  Tempo médio: {stats['mean_time']:.2f}s  |  "
          f"Memória pico média: {stats['mean_memory_mb']:.2f} MB")

    return df_results, stats, all_histories, best_overall_tour, coords, name


# ============================================================
# 10. FUNÇÕES DE VISUALIZAÇÃO
# ============================================================
def plot_convergence(all_histories, instance_name, optimal=None, save_path=None):
    """
    Plota curvas de convergência: best fitness e average fitness.

    Cada execução é plotada com transparência (alpha baixo) e
    a média de todas as execuções é destacada.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Best Fitness ---
    for history in all_histories:
        ax1.plot(history["best"], alpha=0.25, linewidth=0.8, color="steelblue")
    avg_best = np.mean([h["best"] for h in all_histories], axis=0)
    ax1.plot(avg_best, color="darkred", linewidth=2, label="Média das execuções")
    if optimal:
        ax1.axhline(y=optimal, color="green", linestyle="--",
                     linewidth=1.5, label=f"Ótimo = {optimal:,}")
    ax1.set_xlabel("Geração")
    ax1.set_ylabel("Melhor Distância")
    ax1.set_title(f"{instance_name} — Best Fitness")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Average Fitness ---
    for history in all_histories:
        ax2.plot(history["avg"], alpha=0.25, linewidth=0.8, color="steelblue")
    avg_avg = np.mean([h["avg"] for h in all_histories], axis=0)
    ax2.plot(avg_avg, color="darkblue", linewidth=2, label="Média das execuções")
    ax2.set_xlabel("Geração")
    ax2.set_ylabel("Distância Média da População")
    ax2.set_title(f"{instance_name} — Average Fitness")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_best_route(tour, coords, distance, instance_name,
                    optimal=None, save_path=None):
    """
    Plota a melhor rota encontrada sobre as coordenadas das cidades.
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    # Cidades
    ax.scatter(coords[:, 0], coords[:, 1],
               c="crimson", s=25, zorder=5, edgecolors="black", linewidths=0.5)

    # Rota (fecha o ciclo)
    tour_closed = np.append(tour, tour[0])
    ax.plot(coords[tour_closed, 0], coords[tour_closed, 1],
            "b-", linewidth=0.8, alpha=0.7)

    title = f"{instance_name} — Melhor Rota (Distância: {distance:,})"
    if optimal:
        gap = ((distance - optimal) / optimal) * 100
        title += f"  [Gap: {gap:.2f}%]"
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_scalability(all_stats, save_path=None):
    """
    Plota análise de escalabilidade: tempo e gap vs número de cidades.
    """
    df = pd.DataFrame(all_stats).sort_values("dimension")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(df["dimension"], df["mean_time"], "o-", color="teal", linewidth=2)
    ax1.set_xlabel("Número de Cidades")
    ax1.set_ylabel("Tempo Médio (s)")
    ax1.set_title("Escalabilidade — Tempo de Execução")
    ax1.grid(True, alpha=0.3)

    ax2.plot(df["dimension"], df["mean_gap"], "s-", color="orangered", linewidth=2)
    ax2.set_xlabel("Número de Cidades")
    ax2.set_ylabel("Gap Médio (%)")
    ax2.set_title("Escalabilidade — Qualidade da Solução")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def print_summary_table(all_stats):
    """
    Exibe e retorna tabela resumo com os resultados de todas as instâncias.
    """
    df = pd.DataFrame(all_stats)
    cols = ["instance", "dimension", "optimal", "best_found",
            "mean_distance", "std_distance", "best_gap", "mean_gap",
            "mean_time", "mean_memory_mb"]
    display_names = [
        "Instância", "Cidades", "Ótimo", "Melhor Encontrado",
        "Média", "Desvio Padrão", "Melhor Gap (%)", "Gap Médio (%)",
        "Tempo Médio (s)", "Memória (MB)"
    ]
    df = df[cols]
    df.columns = display_names

    print("\n" + "=" * 110)
    print("  TABELA RESUMO DE RESULTADOS")
    print("=" * 110)
    print(df.to_string(index=False))
    print("=" * 110)
    return df


# ============================================================
# 11. EXECUÇÃO PRINCIPAL
# ============================================================
def main():
    """Ponto de entrada principal: configura e executa todos os experimentos."""

    # ---------- CONFIGURAÇÃO ----------

    # Diretório com os arquivos .tsp baixados da TSPLIB
    TSP_DIR = "tsplib"

    # Parâmetros do Algoritmo Genético
    params = {
        "pop_size":        100,      # Tamanho da população
        "n_generations":   500,      # Número de gerações
        "crossover_rate":  0.8,      # Probabilidade de crossover
        "mutation_rate":   0.2,      # Probabilidade de mutação
        "tournament_size": 5,        # Tamanho do torneio
        "elite_size":      2,        # Indivíduos preservados por elitismo
        "n_runs":          10,       # Número de execuções por instância
    }

    # Instâncias a executar: {nome_arquivo: ótimo_conhecido}
    instances = {
        "berlin52": 7542,
        "eil76":    538,
        "kroA100":  21282,
        "ch150":    6528,
        "a280":     2579,
    }

    # Diretório de saída
    OUTPUT_DIR = "resultados"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------- CABEÇALHO ----------
    print("=" * 60)
    print("  ALGORITMO GENÉTICO PARA O TSP")
    print("  Computação Evolutiva e Conexionista")
    print("=" * 60)
    print("\n  Parâmetros do AG:")
    for k, v in params.items():
        print(f"    {k:>16s} = {v}")
    print()

    # ---------- EXECUÇÃO ----------
    all_stats = []

    for inst_name, optimal in instances.items():
        filepath = os.path.join(TSP_DIR, f"{inst_name}.tsp")

        if not os.path.exists(filepath):
            print(f"\n  [ERRO] Arquivo não encontrado: {filepath}")
            print(f"         Baixe a instância em:")
            print(f"         http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/")
            continue

        # Executa o experimento
        df_results, stats, histories, best_tour, coords, name = \
            run_experiment(filepath, params, optimal)

        all_stats.append(stats)

        # Gráficos de convergência
        plot_convergence(
            histories, name, optimal=optimal,
            save_path=os.path.join(OUTPUT_DIR, f"convergencia_{inst_name}.png")
        )

        # Gráfico da melhor rota
        plot_best_route(
            best_tour, coords, stats["best_found"], name, optimal=optimal,
            save_path=os.path.join(OUTPUT_DIR, f"rota_{inst_name}.png")
        )

        # Salva resultados individuais em CSV
        df_results.to_csv(
            os.path.join(OUTPUT_DIR, f"runs_{inst_name}.csv"), index=False
        )

    # ---------- RESUMO FINAL ----------
    if all_stats:
        df_summary = print_summary_table(all_stats)
        df_summary.to_csv(
            os.path.join(OUTPUT_DIR, "resumo_geral.csv"), index=False
        )

        # Gráfico de escalabilidade
        plot_scalability(
            all_stats,
            save_path=os.path.join(OUTPUT_DIR, "escalabilidade.png")
        )

    print(f"\n  Resultados salvos em: {os.path.abspath(OUTPUT_DIR)}/")
    print("  Concluído!\n")


# ============================================================
if __name__ == "__main__":
    main()