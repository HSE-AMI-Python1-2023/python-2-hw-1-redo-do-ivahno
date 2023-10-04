import pytest
import numpy as np
import random
from scipy.stats import qmc
import math

def test_diff_evolution_part_4():
    from diff_evolution import differential_evolution
    SEED = 21
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


    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  1.0215162049576065e-12
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='random_among_worst'))[-1][1] ==  2.8119728767705965e-12
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] ==  2.404743071338089e-13
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand2', selection_setting='current'))[-1][1] ==  1.1605779670631478e-09
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand2', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand2', selection_setting='random_among_worst'))[-1][1] ==  5.415138337738767e-10
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand2', selection_setting='random_selection'))[-1][1] ==  8.273381979506667e-13
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='best1', selection_setting='current'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='best1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='best1', selection_setting='random_among_worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='best1', selection_setting='random_selection'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='current'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='random_among_worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='random_selection'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  9.044764937016225e-12
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='random_among_worst'))[-1][1] ==  5.42512701429132e-11
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] ==  8.626432901337466e-14
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='current'))[-1][1] ==  7.747535946123207e-10
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='random_among_worst'))[-1][1] ==  7.956492131810933e-10
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='random_selection'))[-1][1] ==  1.529509852105093e-11
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='best1', selection_setting='current'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='best1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='best1', selection_setting='random_among_worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='best1', selection_setting='random_selection'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='current'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='random_among_worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='random_selection'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  1.8017032310524428e-11
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand1', selection_setting='random_among_worst'))[-1][1] ==  7.780442956573097e-13
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] ==  5.21249710061511e-13
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand2', selection_setting='current'))[-1][1] ==  4.449752566415555e-09
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand2', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand2', selection_setting='random_among_worst'))[-1][1] ==  4.352338711655079e-09
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand2', selection_setting='random_selection'))[-1][1] ==  8.940181928096536e-12
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='best1', selection_setting='current'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='best1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='best1', selection_setting='random_among_worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='best1', selection_setting='random_selection'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='current'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='random_among_worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='random_selection'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  2.5156543514981422e-12
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand1', selection_setting='random_among_worst'))[-1][1] ==  2.3953061756287752e-12
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] ==  4.696243394164412e-14
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand2', selection_setting='current'))[-1][1] ==  8.138583140748779e-10
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand2', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand2', selection_setting='random_among_worst'))[-1][1] ==  3.817779425929757e-11
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand2', selection_setting='random_selection'))[-1][1] ==  1.7663348561569592e-10
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='best1', selection_setting='current'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='best1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='best1', selection_setting='random_among_worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='best1', selection_setting='random_selection'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='current'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='random_among_worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='random_selection'))[-1][1] ==  0.0

    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  1.197346524151044e-07
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='worst'))[-1][1] ==  5.329070518200751e-15
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='random_among_worst'))[-1][1] ==  3.451623165062756e-08
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] ==  2.028812673415814e-10
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand2', selection_setting='current'))[-1][1] ==  0.00041887219276759424
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand2', selection_setting='worst'))[-1][1] ==  4.796163466380676e-13
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand2', selection_setting='random_among_worst'))[-1][1] ==  0.00043792941555587106
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand2', selection_setting='random_selection'))[-1][1] ==  6.479763019484608e-07
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='best1', selection_setting='current'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='best1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='best1', selection_setting='random_among_worst'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='best1', selection_setting='random_selection'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='current'))[-1][1] ==  6.927791673660977e-14
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='random_among_worst'))[-1][1] ==  2.4868995751603507e-14
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='random_selection'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  1.2930886263973207e-06
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='random_among_worst'))[-1][1] ==  4.697851807122788e-06
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] ==  7.902087872935226e-10
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='current'))[-1][1] ==  0.00033858769831063285
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='worst'))[-1][1] ==  1.580957587066223e-13
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='random_among_worst'))[-1][1] ==  0.00021997450961919185
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='random_selection'))[-1][1] ==  5.7356261962127064e-08
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='best1', selection_setting='current'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='best1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='best1', selection_setting='random_among_worst'))[-1][1] ==  7.105427357601002e-15
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='best1', selection_setting='random_selection'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='current'))[-1][1] ==  1.5987211554602254e-14
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='random_among_worst'))[-1][1] ==  1.7763568394002505e-15
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='random_selection'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  4.044373724809702e-10
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand1', selection_setting='worst'))[-1][1] ==  1.7763568394002505e-15
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand1', selection_setting='random_among_worst'))[-1][1] ==  2.2562766872624707e-06
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] ==  9.022457447827037e-09
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand2', selection_setting='current'))[-1][1] ==  0.0001792338154231743
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand2', selection_setting='worst'))[-1][1] ==  8.189005029635155e-13
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand2', selection_setting='random_among_worst'))[-1][1] ==  4.563925637768307e-05
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand2', selection_setting='random_selection'))[-1][1] ==  9.324457650450313e-09
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='best1', selection_setting='current'))[-1][1] ==  1.9539925233402755e-14
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='best1', selection_setting='worst'))[-1][1] ==  0.9949590570932898
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='best1', selection_setting='random_among_worst'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='best1', selection_setting='random_selection'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='current'))[-1][1] ==  1.7763568394002505e-15
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='random_among_worst'))[-1][1] ==  1.7763568394002505e-15
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='random_selection'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  1.5938252939662334e-06
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand1', selection_setting='random_among_worst'))[-1][1] ==  5.267711600254188e-09
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] ==  3.1586555593321464e-09
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand2', selection_setting='current'))[-1][1] ==  0.0004433015054683409
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand2', selection_setting='worst'))[-1][1] ==  2.34123831432953e-12
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand2', selection_setting='random_among_worst'))[-1][1] ==  0.0006425692265228378
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand2', selection_setting='random_selection'))[-1][1] ==  1.2351379012898178e-05
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='best1', selection_setting='current'))[-1][1] ==  1.7763568394002505e-14
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='best1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='best1', selection_setting='random_among_worst'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='best1', selection_setting='random_selection'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='current'))[-1][1] ==  5.329070518200751e-15
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='random_among_worst'))[-1][1] ==  1.3145040611561853e-13
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='random_selection'))[-1][1] ==  0.0

    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='random', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  3.379606654175286e-05
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='random', mutation_setting='rand1', selection_setting='worst'))[-1][1] ==  1.623440975814363e-06
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='random', mutation_setting='rand1', selection_setting='random_among_worst'))[-1][1] ==  2.6353992512128422e-05
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='random', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] ==  5.0291536127361185e-05
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='random', mutation_setting='rand2', selection_setting='current'))[-1][1] ==  0.0007829271144532441
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='random', mutation_setting='rand2', selection_setting='worst'))[-1][1] ==  4.3670919820206185e-06
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='random', mutation_setting='rand2', selection_setting='random_among_worst'))[-1][1] ==  0.0005061350363791974
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='random', mutation_setting='rand2', selection_setting='random_selection'))[-1][1] ==  0.0001115965206988795
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='random', mutation_setting='best1', selection_setting='current'))[-1][1] ==  2.9372280506324224e-08
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='random', mutation_setting='best1', selection_setting='worst'))[-1][1] ==  8.947500696426115e-16
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='random', mutation_setting='best1', selection_setting='random_among_worst'))[-1][1] ==  8.628961748372131e-09
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='random', mutation_setting='best1', selection_setting='random_selection'))[-1][1] ==  8.781066846358215e-09
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='current'))[-1][1] ==  3.459298758896539e-08
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='worst'))[-1][1] ==  1.8467453932420408e-12
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='random_among_worst'))[-1][1] ==  4.827414434779271e-09
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='random_selection'))[-1][1] ==  2.3673602669422268e-08
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  1.7068164604891756e-05
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='worst'))[-1][1] ==  9.345939532357008e-07
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='random_among_worst'))[-1][1] ==  6.291421812443823e-06
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] ==  3.960062842304544e-06
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='current'))[-1][1] ==  7.618182936255569e-05
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='worst'))[-1][1] ==  5.341116664177106e-05
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='random_among_worst'))[-1][1] ==  0.002990757048292397
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='random_selection'))[-1][1] ==  0.00037806397695296307
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='LatinHypercube', mutation_setting='best1', selection_setting='current'))[-1][1] ==  1.7612832371109777e-09
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='LatinHypercube', mutation_setting='best1', selection_setting='worst'))[-1][1] ==  1.2910767433003021e-14
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='LatinHypercube', mutation_setting='best1', selection_setting='random_among_worst'))[-1][1] ==  1.590405024650817e-10
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='LatinHypercube', mutation_setting='best1', selection_setting='random_selection'))[-1][1] ==  1.5452104130545194e-09
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='current'))[-1][1] ==  3.671149271202803e-08
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='worst'))[-1][1] ==  4.0295585790760335e-12
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='random_among_worst'))[-1][1] ==  3.4886110201264687e-10
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='random_selection'))[-1][1] ==  9.220667801561064e-09
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Halton', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  2.2428120991787372e-05
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Halton', mutation_setting='rand1', selection_setting='worst'))[-1][1] ==  6.8610005020808336e-09
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Halton', mutation_setting='rand1', selection_setting='random_among_worst'))[-1][1] ==  4.54963851291326e-05
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Halton', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] ==  0.008667212694634824
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Halton', mutation_setting='rand2', selection_setting='current'))[-1][1] ==  0.0003583405017178267
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Halton', mutation_setting='rand2', selection_setting='worst'))[-1][1] ==  5.476435354155614e-06
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Halton', mutation_setting='rand2', selection_setting='random_among_worst'))[-1][1] ==  3.088107357810844e-05
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Halton', mutation_setting='rand2', selection_setting='random_selection'))[-1][1] ==  9.659293658574029e-06
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Halton', mutation_setting='best1', selection_setting='current'))[-1][1] ==  7.689410044606886e-09
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Halton', mutation_setting='best1', selection_setting='worst'))[-1][1] ==  3.77437687019181e-14
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Halton', mutation_setting='best1', selection_setting='random_among_worst'))[-1][1] ==  3.035989650540948e-09
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Halton', mutation_setting='best1', selection_setting='random_selection'))[-1][1] ==  7.681663761139406e-10
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='current'))[-1][1] ==  2.4542272386157863e-08
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='worst'))[-1][1] ==  3.5119405241796733e-13
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='random_among_worst'))[-1][1] ==  2.316908229883164e-08
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='random_selection'))[-1][1] ==  4.803442283500619e-10
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Sobol', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  0.00016929239603726042
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Sobol', mutation_setting='rand1', selection_setting='worst'))[-1][1] ==  7.953064645964002e-05
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Sobol', mutation_setting='rand1', selection_setting='random_among_worst'))[-1][1] ==  4.797248952328879e-06
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Sobol', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] ==  9.222862222860833e-05
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Sobol', mutation_setting='rand2', selection_setting='current'))[-1][1] ==  0.0023816855039101303
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Sobol', mutation_setting='rand2', selection_setting='worst'))[-1][1] ==  2.5684822712800203e-05
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Sobol', mutation_setting='rand2', selection_setting='random_among_worst'))[-1][1] ==  0.005407317348194391
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Sobol', mutation_setting='rand2', selection_setting='random_selection'))[-1][1] ==  0.00037558037968324175
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Sobol', mutation_setting='best1', selection_setting='current'))[-1][1] ==  2.8051519111737967e-08
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Sobol', mutation_setting='best1', selection_setting='worst'))[-1][1] ==  3.988311496010582e-15
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Sobol', mutation_setting='best1', selection_setting='random_among_worst'))[-1][1] ==  2.970985647484177e-08
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Sobol', mutation_setting='best1', selection_setting='random_selection'))[-1][1] ==  1.2695345835865277e-09
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='current'))[-1][1] ==  4.172793337981942e-07
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='worst'))[-1][1] ==  3.0160103711769958e-12
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='random_among_worst'))[-1][1] ==  8.29268805783664e-08
    assert list(differential_evolution(rosenbrock, [[0, 2], [0, 2]], init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='random_selection'))[-1][1] ==  3.2886353169544126e-09
