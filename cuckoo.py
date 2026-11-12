
import random

# --------------------------------
# 1. Cloud infrastructure
# --------------------------------

SERVER_CPU = 16       # CPU units per physical server
SERVER_RAM = 32       # GB RAM per physical server

VMS = [
    {"name": "VM1", "cpu": 4, "ram": 8},
    {"name": "VM2", "cpu": 6, "ram": 12},
    {"name": "VM3", "cpu": 2, "ram": 4},
    {"name": "VM4", "cpu": 8, "ram": 16},
    {"name": "VM5", "cpu": 4, "ram": 8},
    {"name": "VM6", "cpu": 6, "ram": 8},
    {"name": "VM7", "cpu": 2, "ram": 4},
    {"name": "VM8", "cpu": 4, "ram": 12},
]

NUM_SERVERS = len(VMS)


# --------------------------------
# 2. Evaluate a placement
# --------------------------------

def evaluate(solution):
    cpu = [0] * NUM_SERVERS
    ram = [0] * NUM_SERVERS

    for i, server in enumerate(solution):
        cpu[server] += VMS[i]["cpu"]
        ram[server] += VMS[i]["ram"]

    # Reject placements exceeding server capacity
    for s in range(NUM_SERVERS):
        if cpu[s] > SERVER_CPU or ram[s] > SERVER_RAM:
            return 10000 + sum(cpu) + sum(ram)

    active = sum(
        1 for s in range(NUM_SERVERS)
        if cpu[s] > 0
    )

    # Primary objective: minimize active servers
    return active


# --------------------------------
# 3. Create a random nest
# --------------------------------

def new_nest():
    return [
        random.randrange(NUM_SERVERS)
        for _ in VMS
    ]


# --------------------------------
# 4. Cuckoo Search
# --------------------------------

def cuckoo_search(nests_count=30, iterations=300,
                  discovery_rate=0.25):

    nests = [new_nest() for _ in range(nests_count)]

    # Include a valid placement as a starting candidate
    nests[0] = list(range(len(VMS)))

    scores = [evaluate(n) for n in nests]

    best_index = scores.index(min(scores))
    best = nests[best_index][:]
    best_score = scores[best_index]

    for _ in range(iterations):

        # Generate a new candidate by modifying a nest
        i = random.randrange(nests_count)
        candidate = nests[i][:]

        # Change one or more VM assignments
        changes = random.randint(1, max(1, len(VMS) // 2))

        for _ in range(changes):
            vm = random.randrange(len(VMS))
            candidate[vm] = random.randrange(NUM_SERVERS)

        candidate_score = evaluate(candidate)

        # Keep the better solution
        j = random.randrange(nests_count)

        if candidate_score < scores[j]:
            nests[j] = candidate
            scores[j] = candidate_score

        # Abandon some nests and generate new ones
        for k in range(nests_count):
            if random.random() < discovery_rate:
                replacement = new_nest()
                replacement_score = evaluate(replacement)

                if replacement_score < scores[k]:
                    nests[k] = replacement
                    scores[k] = replacement_score

        # Update global best
        current_index = scores.index(min(scores))

        if scores[current_index] < best_score:
            best = nests[current_index][:]
            best_score = scores[current_index]

    return best, best_score


# --------------------------------
# 5. Simple First-Fit baseline
# --------------------------------

def first_fit():
    placement = []
    cpu = []
    ram = []

    for vm in VMS:
        assigned = False

        for s in range(len(cpu)):
            if (cpu[s] + vm["cpu"] <= SERVER_CPU
                    and ram[s] + vm["ram"] <= SERVER_RAM):
                placement.append(s)
                cpu[s] += vm["cpu"]
                ram[s] += vm["ram"]
                assigned = True
                break

        if not assigned:
            placement.append(len(cpu))
            cpu.append(vm["cpu"])
            ram.append(vm["ram"])

    return placement


# --------------------------------
# 6. Display results
# --------------------------------

def show_results(title, solution):
    print("\n" + title)
    print("-" * 45)

    for s in sorted(set(solution)):
        members = [
            VMS[i]["name"]
            for i, server in enumerate(solution)
            if server == s
        ]

        cpu = sum(
            VMS[i]["cpu"]
            for i, server in enumerate(solution)
            if server == s
        )

        ram = sum(
            VMS[i]["ram"]
            for i, server in enumerate(solution)
            if server == s
        )

        print(f"Server {s + 1}: {', '.join(members)}")
        print(f"  CPU: {cpu}/{SERVER_CPU}")
        print(f"  RAM: {ram}/{SERVER_RAM} GB")

    print("Active servers:", len(set(solution)))


if __name__ == "__main__":
    baseline = first_fit()
    optimized, score = cuckoo_search()

    show_results("FIRST-FIT BASELINE", baseline)
    show_results("CUCKOO SEARCH", optimized)

    print("\nBaseline server count:", len(set(baseline)))
    print("Cuckoo Search server count:", score)
#Preeth Rai
#1WN24CS205