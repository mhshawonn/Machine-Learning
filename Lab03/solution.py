def calculate_timekeeper_energy(m: int, n: int, l: int, t: int) -> int:
    total_energy = 0
    for i in range(m):
        for j in range(n):
            energy = i ^ j
            if energy > l:
                total_energy += energy
    return total_energy % t

    