import os
from anneal import SimAnneal
import matplotlib.pyplot as plt
import random
import time

def read_coords(path):
    coords = []
    with open(path, "r") as f:
        for line in f.readlines():
            line = [float(x.replace("\n", "")) for x in line.split(" ")]
            coords.append(line)
    return coords


def generate_random_coords(num_nodes):
    return [[random.uniform(-1000, 1000), random.uniform(-1000, 1000)] for i in range(num_nodes)]


if __name__ == "__main__":
    
    path = r'/data/datos.txt'
    separador = os.path.sep
    dir_actual = os.path.dirname(os.path.abspath(__file__))
    dir = separador.join(dir_actual.split(separador)[:-1])
    newPath = dir + path
    
    coords = read_coords(newPath)  
    sa = SimAnneal(coords, stopping_iter=900)
    startTime = time.time()
    sa.anneal()
    
    endTime = time.time()
    print(f"Route: {sa.best_solution}")
    print(f"Tiempo de ejecución: {(endTime - startTime):.6f} segundos")

    sa.visualize_routes()
    sa.plot_learning()

