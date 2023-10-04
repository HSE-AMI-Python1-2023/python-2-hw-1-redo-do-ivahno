import pytest
import numpy as np
import random
from scipy.stats import qmc
import math

def test_diff_evolution_part_3():
    from diff_evolution import differential_evolution
    SEED = 228
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
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='random', mutation_setting='rand1', selection_setting='worst'))[-1][1] ==  1.623440975814363e-06
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='random', mutation_setting='rand1', selection_setting='random_among_worst'))[-1][1] ==  2.6353992512128422e-05
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='random', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] ==  5.0291536127361185e-05
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  1.197346524151044e-07
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='worst'))[-1][1] ==  5.329070518200751e-15
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='random_among_worst'))[-1][1] ==  3.451623165062756e-08
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] ==  2.028812673415814e-10
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  1.0215162049576065e-12
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='random_among_worst'))[-1][1] ==  2.8119728767705965e-12
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] ==  2.404743071338089e-13



