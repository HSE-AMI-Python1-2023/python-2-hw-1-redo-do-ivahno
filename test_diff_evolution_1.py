import pytest
import numpy as np
import random
from scipy.stats import qmc
import math

def test_diff_evolution_part_1():
    from diff_evolution import differential_evolution
    SEED = 7
    random.seed(SEED)
    np.random.seed(SEED)

    def rastrigin(array, A=10):
        return A * 2 + (array[0] ** 2 - A * np.cos(2 * np.pi * array[0])) + (array[1] ** 2 - A * np.cos(2 * np.pi * array[1]))
  
    def griewank(array):
        term_1 = (array[0] ** 2 + array[1] ** 2) / 2
        term_2 = np.cos(array[0]/ np.sqrt(2)) * np.cos(array[1]/ np.sqrt(2))
        return 1 + term_1 - term_2
  
    def rosenbrock(array):
        return (1 - array[0]) ** 2 + 100 * (array[1] - array[0] ** 2) ** 2

    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='random', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  3.379606654175286e-05
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting="LatinHypercube", mutation_setting='rand1', selection_setting='current'))[-1][1] ==  1.7068164604891756e-05
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting="Halton", mutation_setting='rand1', selection_setting='current'))[-1][1] ==  2.2428120991787372e-05
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting="Sobol", mutation_setting='rand1', selection_setting='current'))[-1][1] ==  0.00016929239603726042
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  1.197346524151044e-07
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  1.2930886263973207e-06
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  4.044373724809702e-10
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  1.5938252939662334e-06
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  1.0215162049576065e-12
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  9.044764937016225e-12
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  1.8017032310524428e-11
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  2.5156543514981422e-12
