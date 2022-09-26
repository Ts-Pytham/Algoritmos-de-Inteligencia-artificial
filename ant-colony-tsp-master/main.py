import math
from aco import ACO, Graph
from plot import plot
import matplotlib.pyplot as plt
import time, os

def distance(city1: dict, city2: dict):
    return math.sqrt((city1['x'] - city2['x']) ** 2 + (city1['y'] - city2['y']) ** 2)


def main():
    cities = []
    points = []
    #using the exact, or absolute path.
    i = 1

    path = r'/data/datos.txt'
    separador = os.path.sep
    dir_actual = os.path.dirname(os.path.abspath(__file__))
    dir = separador.join(dir_actual.split(separador)[:-1])
    newPath = dir + path

    with open(newPath) as f:
        for line in f.readlines():

            city = line.split(' ')
            cities.append(dict(index=i, x=float(city[0]), y=float(city[1])))
            points.append((float(city[0]), float(city[1])))
            i += 1
    startTime = time.time()

    cost_matrix = []
    rank = len(cities)
    for i in range(rank):
        row = []
        for j in range(rank):
            row.append(distance(cities[i], cities[j]))
        cost_matrix.append(row)
    aco = ACO(10, 1000, 1.0, 10.0, 0.5, 10, 2)
    graph = Graph(cost_matrix, rank)
    path, cost = aco.solve(graph)
    endTime = time.time()
    print('cost: {}, path: {}'.format(cost, path))
    print(f'Time: {(endTime - startTime):.6f}')
    plot(points, path)
    aco.plot_learning()
if __name__ == '__main__':
    main()
