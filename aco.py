import random
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# =====================================================
# 1. CREATE A MORE COMPLEX COMMUNICATION NETWORK
# =====================================================

random.seed(42)
np.random.seed(42)

NUM_NODES = 25
SOURCE = 0
DESTINATION = 21

# Connected small-world network with many alternative paths
G = nx.connected_watts_strogatz_graph(
    n=NUM_NODES,
    k=4,
    p=0.35,
    seed=17
)

# Assign different communication costs to the links
for u, v in G.edges():
    G[u][v]["weight"] = random.randint(1, 20)

# Keep the graph layout fixed throughout the simulation
pos = nx.spring_layout(G, seed=12, iterations=100)

print("Number of routers:", G.number_of_nodes())
print("Number of communication links:", G.number_of_edges())


# =====================================================
# 2. ACO PARAMETERS
# =====================================================

NUM_ANTS = 35
ITERATIONS = 150

ALPHA = 1.0          # Pheromone importance
BETA = 3.0           # Link-cost heuristic importance
EVAPORATION = 0.25    # Pheromone evaporation rate
Q = 100.0            # Pheromone deposit constant
ELITE_WEIGHT = 3.0   # Extra reinforcement for best route

# Initial pheromone on every link
pheromone = {
    (min(u, v), max(u, v)): 1.0
    for u, v in G.edges()
}


# =====================================================
# 3. ROUTE COST
# =====================================================

def calculate_cost(route):
    return sum(
        G[u][v]["weight"]
        for u, v in zip(route[:-1], route[1:])
    )


# =====================================================
# 4. CONSTRUCT A ROUTE USING AN ANT
# =====================================================

def construct_route():
    route = [SOURCE]
    visited = {SOURCE}
    current = SOURCE

    # Prevent infinite loops
    while current != DESTINATION:

        candidates = [
            node for node in G.neighbors(current)
            if node not in visited
        ]

        if not candidates:
            return None, float("inf")

        scores = []

        for nxt in candidates:
            edge = (min(current, nxt), max(current, nxt))

            tau = pheromone[edge]
            distance = G[current][nxt]["weight"]

            # Higher pheromone and lower link cost
            # increase the probability of selecting a link
            score = (
                tau ** ALPHA
                * (1.0 / distance) ** BETA
            )

            scores.append(score)

        total = sum(scores)

        if total <= 0 or not np.isfinite(total):
            nxt = random.choice(candidates)
        else:
            probabilities = [s / total for s in scores]

            nxt = random.choices(
                candidates,
                weights=probabilities,
                k=1
            )[0]

        route.append(nxt)
        visited.add(nxt)
        current = nxt

    return route, calculate_cost(route)


# =====================================================
# 5. FIND THE REFERENCE OPTIMAL ROUTE
# =====================================================

optimal_route = nx.shortest_path(
    G,
    source=SOURCE,
    target=DESTINATION,
    weight="weight"
)

optimal_cost = calculate_cost(optimal_route)

print("\nDijkstra reference route:", optimal_route)
print("Dijkstra reference cost:", optimal_cost)


# =====================================================
# 6. PREPARE LIVE VISUALIZATION
# =====================================================

plt.ion()

fig, (ax_network, ax_convergence) = plt.subplots(
    1, 2, figsize=(16, 7)
)

best_route = None
best_cost = float("inf")
history = []
improvements = 0


def draw_network(iteration, new_improvement=False):

    ax_network.clear()
    ax_convergence.clear()

    # Draw all network links
    nx.draw_networkx_edges(
        G, pos, ax=ax_network,
        edge_color="lightgray",
        width=1.5
    )

    # Draw all routers
    nx.draw_networkx_nodes(
        G, pos, ax=ax_network,
        node_color="skyblue",
        node_size=500
    )

    # Highlight source and destination
    nx.draw_networkx_nodes(
        G, pos, ax=ax_network,
        nodelist=[SOURCE],
        node_color="limegreen",
        node_size=650
    )

    nx.draw_networkx_nodes(
        G, pos, ax=ax_network,
        nodelist=[DESTINATION],
        node_color="tomato",
        node_size=650
    )

    nx.draw_networkx_labels(
        G, pos, ax=ax_network,
        font_size=8
    )

    # Display the cost of each link
    labels = nx.get_edge_attributes(G, "weight")

    nx.draw_networkx_edge_labels(
        G, pos, ax=ax_network,
        edge_labels=labels,
        font_size=7
    )

    # Highlight the best route found so far
    if best_route is not None:

        route_edges = list(
            zip(best_route[:-1], best_route[1:])
        )

        nx.draw_networkx_edges(
            G, pos, ax=ax_network,
            edgelist=route_edges,
            edge_color="green",
            width=4
        )

        # Highlight intermediate routers on the best path
        intermediate = [
            node for node in best_route
            if node not in (SOURCE, DESTINATION)
        ]

        nx.draw_networkx_nodes(
            G, pos, ax=ax_network,
            nodelist=intermediate,
            node_color="gold",
            node_size=600
        )

        # Redraw source and destination on top
        nx.draw_networkx_nodes(
            G, pos, ax=ax_network,
            nodelist=[SOURCE],
            node_color="limegreen",
            node_size=650
        )

        nx.draw_networkx_nodes(
            G, pos, ax=ax_network,
            nodelist=[DESTINATION],
            node_color="tomato",
            node_size=650
        )

        nx.draw_networkx_labels(
            G, pos, ax=ax_network,
            font_size=8
        )

    status = "NEW BEST ROUTE!" if new_improvement else "Searching..."

    ax_network.set_title(
        f"CNRP — ACO Routing\n"
        f"Iteration {iteration}/{ITERATIONS} | "
        f"Best cost: {best_cost:.0f} | {status}"
    )

    ax_network.axis("off")

    # Plot best-so-far cost (non-increasing)
    ax_convergence.plot(
        range(1, len(history) + 1),
        history,
        color="green",
        linewidth=2,
        label="ACO best-so-far cost"
    )

    ax_convergence.axhline(
        optimal_cost,
        color="red",
        linestyle="--",
        label=f"Dijkstra optimum = {optimal_cost}"
    )

    ax_convergence.set_title("ACO Convergence")
    ax_convergence.set_xlabel("Ant / search step")
    ax_convergence.set_ylabel("Best route cost")
    ax_convergence.grid(True, alpha=0.3)
    ax_convergence.legend()

    fig.tight_layout()
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    # Pause longer on a genuine improvement
    plt.pause(0.35 if new_improvement else 0.03)


# =====================================================
# 7. RUN THE ACO ALGORITHM
# =====================================================

start_time = time.perf_counter()

draw_network(0)

for iteration in range(1, ITERATIONS + 1):

    iteration_routes = []

    for ant in range(NUM_ANTS):

        route, cost = construct_route()

        if route is None:
            continue

        iteration_routes.append((route, cost))

        improved = cost < best_cost

        if improved:
            best_route = route[:]
            best_cost = cost
            improvements += 1

            print(
                f"Improvement {improvements}: "
                f"iteration={iteration}, "
                f"ant={ant + 1}, "
                f"cost={best_cost}, "
                f"route={best_route}"
            )

        # Record best-so-far cost after every ant
        history.append(best_cost)

        # Redraw on every genuine improvement
        if improved:
            draw_network(
                iteration,
                new_improvement=True
            )

    # Evaporate pheromone on all links
    for edge in pheromone:
        pheromone[edge] *= (1.0 - EVAPORATION)

    # Deposit pheromone from routes explored this iteration
    for route, cost in iteration_routes:
        deposit = Q / cost

        for u, v in zip(route[:-1], route[1:]):
            edge = (min(u, v), max(u, v))
            pheromone[edge] += deposit

    # Elitist reinforcement: favor the global best route
    if best_route is not None:
        elite_deposit = ELITE_WEIGHT * Q / best_cost

        for u, v in zip(best_route[:-1], best_route[1:]):
            edge = (min(u, v), max(u, v))
            pheromone[edge] += elite_deposit

    # Update progress after each iteration
    draw_network(iteration)


# =====================================================
# 8. FINAL RESULTS
# =====================================================

elapsed = time.perf_counter() - start_time

print("\n========== FINAL RESULTS ==========")
print("ACO best route:", best_route)
print("ACO route cost:", best_cost)
print("Dijkstra optimal route:", optimal_route)
print("Dijkstra optimal cost:", optimal_cost)
print("Improvements:", improvements)
print(f"Elapsed time: {elapsed:.2f} seconds")

gap = best_cost - optimal_cost

print("Optimality gap:", gap)

if optimal_cost > 0:
    print(f"Optimality gap percentage: {100 * gap / optimal_cost:.2f}%")

# Display the final result without pausing for each update
draw_network(ITERATIONS)
plt.ioff()
plt.show()
#Preeth Rai
#1WN24CS205