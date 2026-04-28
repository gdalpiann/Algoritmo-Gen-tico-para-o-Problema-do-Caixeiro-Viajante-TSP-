import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import os
import tracemalloc


# ===================== LEITURA TSPLIB =====================

def read_tsp_file(filepath):
    """Lê arquivo .tsp (TSPLIB) e retorna nome, dimensão, coordenadas e tipo."""
    name, dimension, edge_weight_type = "", 0, "EUC_2D"
    coords = []
    reading_coords = False

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
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

            if reading_coords and line:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        coords.append([float(parts[1]), float(parts[2])])
                    except ValueError:
                        reading_coords = False

    coords = np.array(coords)
    if len(coords) != dimension:
        dimension = len(coords)

    return name, dimension, coords, edge_weight_type


# ===================== MATRIZ DE DISTÂNCIAS =====================

def compute_distance_matrix(coords, edge_weight_type="EUC_2D"):
    """Matriz de distâncias euclidianas (EUC_2D) arredondadas para inteiro."""
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    return np.rint(dist_matrix).astype(int)


# ===================== FITNESS =====================
# Genótipo:  permutação [0, 1, ..., n-1]
# Fenótipo:  rota cíclica visitando todas as cidades
# Fitness:   distância total da rota (minimizar)

def calculate_tour_distance(tour, dist_matrix):
    """Distância total do tour (ciclo fechado)."""
    next_cities = np.roll(tour, -1)
    return int(np.sum(dist_matrix[tour, next_cities]))


def evaluate_population(population, dist_matrix):
    """Avalia toda a população e retorna vetor de fitness."""
    return np.array([calculate_tour_distance(ind, dist_matrix) for ind in population])


# ===================== INICIALIZAÇÃO =====================

def initialize_population(pop_size, num_cities):
    """Gera população inicial como permutações aleatórias."""
    return [np.random.permutation(num_cities) for _ in range(pop_size)]


# ===================== SELEÇÃO POR TORNEIO =====================

def tournament_selection(population, fitness, tournament_size):
    """Seleciona o melhor entre 'tournament_size' candidatos aleatórios."""
    candidates = np.random.choice(len(population), tournament_size, replace=False)
    winner = candidates[np.argmin(fitness[candidates])]
    return population[winner].copy()


# ===================== ORDER CROSSOVER (OX) =====================

def order_crossover(parent1, parent2):
    """
    Crossover de ordem (OX) para permutações.
    Copia segmento do pai1 e preenche o restante com genes do pai2 em ordem circular.
    """
    n = len(parent1)
    start, end = sorted(np.random.choice(n, 2, replace=False))

    child = np.full(n, -1, dtype=int)
    child[start:end + 1] = parent1[start:end + 1]

    in_child = set(child[start:end + 1])
    remaining = [city for city in parent2 if city not in in_child]

    pos = (end + 1) % n
    for city in remaining:
        child[pos] = city
        pos = (pos + 1) % n

    return child


# ===================== MUTAÇÃO POR INVERSÃO =====================

def inversion_mutation(individual):
    """Inverte um segmento aleatório do tour (equivale a 2-opt move)."""
    mutant = individual.copy()
    n = len(mutant)
    start, end = sorted(np.random.choice(n, 2, replace=False))
    mutant[start:end + 1] = mutant[start:end + 1][::-1]
    return mutant


# ===================== ALGORITMO GENÉTICO =====================

def genetic_algorithm(dist_matrix, params):
    """
    AG completo: inicialização → [elitismo + seleção + crossover + mutação] × gerações.
    Retorna melhor tour, sua distância e histórico de convergência.
    """
    num_cities = dist_matrix.shape[0]
    pop_size        = params.get("pop_size", 120)
    n_generations   = params.get("n_generations", 500)
    crossover_rate  = params.get("crossover_rate", 0.8)
    mutation_rate   = params.get("mutation_rate", 0.4)
    tournament_size = params.get("tournament_size", 9)
    elite_size      = params.get("elite_size", 8)

    population = initialize_population(pop_size, num_cities)
    fitness = evaluate_population(population, dist_matrix)

    best_history, avg_history = [], []
    best_idx = np.argmin(fitness)
    global_best_tour = population[best_idx].copy()
    global_best_distance = fitness[best_idx]

    for gen in range(n_generations):
        new_population = []

        # Elitismo
        elite_indices = np.argsort(fitness)[:elite_size]
        for idx in elite_indices:
            new_population.append(population[idx].copy())

        # Gera filhos
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitness, tournament_size)
            parent2 = tournament_selection(population, fitness, tournament_size)

            if np.random.random() < crossover_rate:
                child1 = order_crossover(parent1, parent2)
                child2 = order_crossover(parent2, parent1)
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()

            if np.random.random() < mutation_rate:
                child1 = inversion_mutation(child1)
            if np.random.random() < mutation_rate:
                child2 = inversion_mutation(child2)

            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        # Substituição geracional
        population = new_population[:pop_size]
        fitness = evaluate_population(population, dist_matrix)

        gen_best_idx = np.argmin(fitness)
        if fitness[gen_best_idx] < global_best_distance:
            global_best_distance = fitness[gen_best_idx]
            global_best_tour = population[gen_best_idx].copy()

        best_history.append(global_best_distance)
        avg_history.append(np.mean(fitness))

    return global_best_tour, global_best_distance, {"best": best_history, "avg": avg_history}


# ===================== MÚLTIPLAS EXECUÇÕES =====================

def run_experiment(filepath, params, optimal_distance=None):
    """Executa o AG várias vezes e coleta métricas (distância, gap, tempo, memória)."""
    name, dimension, coords, edge_weight_type = read_tsp_file(filepath)
    dist_matrix = compute_distance_matrix(coords, edge_weight_type)
    n_runs = params.get("n_runs", 25)

    print(f"\n{'=' * 60}")
    print(f"  Instância: {name} | Cidades: {dimension}")
    if optimal_distance:
        print(f"  Ótimo conhecido: {optimal_distance:,}")
    print(f"  Execuções: {n_runs}")
    print(f"{'=' * 60}")

    results, all_histories = [], []
    best_overall_tour = None
    best_overall_distance = float("inf")

    for run in range(n_runs):
        tracemalloc.start()
        start_time = time.time()

        best_tour, best_distance, history = genetic_algorithm(dist_matrix, params)

        elapsed_time = time.time() - start_time
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        gap = ((best_distance - optimal_distance) / optimal_distance * 100) if optimal_distance else None

        results.append({
            "run": run + 1,
            "best_distance": best_distance,
            "time_seconds": round(elapsed_time, 2),
            "peak_memory_mb": round(peak_memory / (1024 * 1024), 2),
            "gap_percent": round(gap, 2) if gap is not None else None,
        })
        all_histories.append(history)

        if best_distance < best_overall_distance:
            best_overall_distance = best_distance
            best_overall_tour = best_tour.copy()

        gap_str = f"Gap: {gap:.2f}%" if gap is not None else ""
        print(f"  Run {run + 1:2d}/{n_runs}  |  "
              f"Dist: {best_distance:>10,}  |  "
              f"{gap_str:>14s}  |  "
              f"Tempo: {elapsed_time:.2f}s")

    df_results = pd.DataFrame(results)

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
        "best_gap":       round(df_results["gap_percent"].min(), 2) if optimal_distance else None,
        "mean_gap":       round(df_results["gap_percent"].mean(), 2) if optimal_distance else None,
    }

    print(f"\n  Melhor: {stats['best_found']:,}  |  "
          f"Média: {stats['mean_distance']:,.0f} ± {stats['std_distance']:,.0f}")
    if optimal_distance:
        print(f"  Melhor gap: {stats['best_gap']:.2f}%  |  Gap médio: {stats['mean_gap']:.2f}%")

    return df_results, stats, all_histories, best_overall_tour, coords, name


# ===================== VISUALIZAÇÃO =====================

def plot_convergence(all_histories, instance_name, optimal=None, save_path=None):
    """Curvas de convergência: best fitness e average fitness por geração."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for h in all_histories:
        ax1.plot(h["best"], alpha=0.25, linewidth=0.8, color="steelblue")
    avg_best = np.mean([h["best"] for h in all_histories], axis=0)
    ax1.plot(avg_best, color="darkred", linewidth=2, label="Média das execuções")
    if optimal:
        ax1.axhline(y=optimal, color="green", linestyle="--", linewidth=1.5,
                     label=f"Ótimo = {optimal:,}")
    ax1.set_xlabel("Geração"); ax1.set_ylabel("Melhor Distância")
    ax1.set_title(f"{instance_name} — Best Fitness")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    for h in all_histories:
        ax2.plot(h["avg"], alpha=0.25, linewidth=0.8, color="steelblue")
    avg_avg = np.mean([h["avg"] for h in all_histories], axis=0)
    ax2.plot(avg_avg, color="darkblue", linewidth=2, label="Média das execuções")
    ax2.set_xlabel("Geração"); ax2.set_ylabel("Distância Média da População")
    ax2.set_title(f"{instance_name} — Average Fitness")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_best_route(tour, coords, distance, instance_name, optimal=None, save_path=None):
    """Plota a melhor rota sobre as coordenadas 2D das cidades."""
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(coords[:, 0], coords[:, 1], c="crimson", s=25, zorder=5,
               edgecolors="black", linewidths=0.5)

    tour_closed = np.append(tour, tour[0])
    ax.plot(coords[tour_closed, 0], coords[tour_closed, 1], "b-", linewidth=0.8, alpha=0.7)

    title = f"{instance_name} — Melhor Rota (Distância: {distance:,})"
    if optimal:
        gap = ((distance - optimal) / optimal) * 100
        title += f"  [Gap: {gap:.2f}%]"
    ax.set_title(title); ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_convergence_grid(instances, output_dir):
    """Monta grid com as imagens de convergência salvas."""
    available = [n for n in instances if os.path.exists(
        os.path.join(output_dir, f"convergencia_{n}.png"))]
    if not available:
        return

    n_inst = len(available)
    fig, axes = plt.subplots(n_inst, 2, figsize=(14, 4.0 * n_inst), squeeze=False)

    for i, inst_name in enumerate(available):
        img = mpimg.imread(os.path.join(output_dir, f"convergencia_{inst_name}.png"))
        h, w = img.shape[:2]
        mid = w // 2
        axes[i, 0].imshow(img[:, :mid]); axes[i, 0].axis("off")
        axes[i, 1].imshow(img[:, mid:]); axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "convergencia_grid.png"), dpi=150, bbox_inches="tight")
    plt.show()


def plot_scalability(all_stats, save_path=None):
    """Tempo e gap vs número de cidades."""
    df = pd.DataFrame(all_stats).sort_values("dimension")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(df["dimension"], df["mean_time"], "o-", color="teal", linewidth=2)
    ax1.set_xlabel("Número de Cidades"); ax1.set_ylabel("Tempo Médio (s)")
    ax1.set_title("Escalabilidade — Tempo de Execução"); ax1.grid(True, alpha=0.3)

    ax2.plot(df["dimension"], df["mean_gap"], "s-", color="orangered", linewidth=2)
    ax2.set_xlabel("Número de Cidades"); ax2.set_ylabel("Gap Médio (%)")
    ax2.set_title("Escalabilidade — Qualidade da Solução"); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def print_summary_table(all_stats):
    """Monta e retorna DataFrame com tabela resumo."""
    df = pd.DataFrame(all_stats)
    cols = ["instance", "dimension", "optimal", "best_found",
            "mean_distance", "std_distance", "best_gap", "mean_gap",
            "mean_time", "mean_memory_mb"]
    display_names = [
        "Instância", "Cidades", "Ótimo", "Melhor",
        "Média", "Desvio", "Melhor Gap (%)", "Gap Médio (%)",
        "Tempo (s)", "Memória (MB)"]
    df = df[cols]
    df.columns = display_names
    return df


# ===================== EXECUÇÃO PRINCIPAL =====================

def main():
    TSP_DIR = "tsplib"
    OUTPUT_DIR = "resultados"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Parâmetros do AG
    params = {
        "pop_size":        120,
        "n_generations":   500,
        "crossover_rate":  0.8,
        "mutation_rate":   0.4,
        "tournament_size": 9,
        "elite_size":      8,
        "n_runs":          25,
    }

    # Instâncias TSPLIB e seus ótimos conhecidos
    instances = {
        "berlin52": 7542,
        "eil76":    538,
        "kroA100":  21282,
        "ch150":    6528,
        "a280":     2579,
    }

    print("=" * 60)
    print("  ALGORITMO GENÉTICO PARA O TSP")
    print("=" * 60)
    print("\n  Parâmetros:")
    for k, v in params.items():
        print(f"    {k:>16s} = {v}")
    print()

    all_stats = []

    for inst_name, optimal in instances.items():
        filepath = os.path.join(TSP_DIR, f"{inst_name}.tsp")

        df_results, stats, histories, best_tour, coords, name = \
            run_experiment(filepath, params, optimal)

        all_stats.append(stats)

        plot_convergence(histories, name, optimal=optimal,
                         save_path=os.path.join(OUTPUT_DIR, f"convergencia_{inst_name}.png"))
        plot_best_route(best_tour, coords, stats["best_found"], name, optimal=optimal,
                        save_path=os.path.join(OUTPUT_DIR, f"rota_{inst_name}.png"))
        df_results.to_csv(os.path.join(OUTPUT_DIR, f"runs_{inst_name}.csv"), index=False)

    if all_stats:
        df_summary = print_summary_table(all_stats)
        print("\n" + "=" * 110)
        print("  TABELA RESUMO")
        print("=" * 110)
        print(df_summary.to_string(index=False))
        print("=" * 110)
        df_summary.to_csv(os.path.join(OUTPUT_DIR, "resumo_geral.csv"), index=False)

        plot_convergence_grid(instances, OUTPUT_DIR)

        if len(all_stats) >= 2:
            plot_scalability(all_stats,
                             save_path=os.path.join(OUTPUT_DIR, "escalabilidade.png"))

    print(f"\n  Resultados salvos em: {os.path.abspath(OUTPUT_DIR)}/")


if __name__ == "__main__":
    main()
