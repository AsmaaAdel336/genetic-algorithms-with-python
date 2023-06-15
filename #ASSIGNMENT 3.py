# ASSIGNMENT 3
import numpy as np
import random as rd
import math

rd.seed(10)


def create_initial_pop(pop_size, variables_min, variables_max):

    number_variables = len(variables_min)
    init_pop = np.random.uniform(
        variables_min, variables_max, (pop_size, number_variables))
    return init_pop


def arithmetic_crossover(two_parents, pcross):

    two_children = two_parents.copy()

    x = rd.random()

    if(x <= pcross):
        alpha = rd.random()
        two_children[0, :] = alpha*two_parents[0, :] + \
            (1-alpha)*two_parents[1, :]
        alpha = rd.random()
        two_children[1, :] = alpha*two_parents[0, :] + \
            (1-alpha)*two_parents[1, :]
    return two_children


def gaussian_mutate(individual, variables_min, variables_max, sigma=0.5, pmute=0.05):
    mutated_individual = individual.copy()
    if(rd.random() <= pmute):
        m = np.random.normal(0, sigma)
        mutated_individual = np.array(individual) + m
    return mutated_individual


def fitness(real_pop, pop_size):  # using the objective function in assignment 2
    fit = np.zeros((pop_size, 1))

    for i in range(pop_size):
        fit[i] = 8-(real_pop[i][0]+0.0317)**2 + (real_pop[i][1])**2

    return fit


def tournament_selection(pop, k):
    num_of_INDs = np.shape(pop)[0]
    two_parents = np.zeros((2, np.size(pop, 1)))
    selected_parets = np.zeros((k, np.size(pop, 1)))

    for i in range(2):
        for j in range(k):
            rand_selected_INDX = rd.randint(0, num_of_INDs-1)
            selected_parets[j] = pop[rand_selected_INDX]

        fitness = fitness(selected_parets, k)
        parent_indx = np.argmax(fitness)
        two_parents[i] = pop[parent_indx]

    return two_parents


def Elitism(pop=[], fitness=[]):
    elitism = np.zeros((2, np.size(pop, 1)))
    fit_copy = fitness.copy()
    fit_copy.sort(reverse=True)
    max_fitness = fit_copy[0]
    second_max_fitness = fit_copy[1]
    for i in range(len(fitness)):
        if fitness[i] == max_fitness:
            indx1 = i
        if fitness[i] == second_max_fitness:
            indx2 = i
    elitism[0, :] = pop[indx1]
    elitism[1, :] = pop[indx2]
    fitness.remove(max_fitness)
    fitness.remove(second_max_fitness)
    # to delete row(axis 0 = horizontal = row) of indx1 in matrix pop
    pop = np.delete(pop, indx1, 0)
    pop = np.delete(pop, indx2-1, 0)

    return elitism


# without elitism
def run_real_GA(npop, x_min, x_max, ngen, pcross, pmute, sigma):

    best_hist = []
    avg_best_hist = 0
    pop = create_initial_pop(npop, x_min, x_max)
    new_generation = np.zeros_like(pop)
    for i in range(ngen):
        for j in range(npop, 2):
            twoparents = tournament_selection(pop, k)
            twochildren = arithmetic_crossover(twoparents, pcross)
            new_generation[i, :] = twochildren[0, :]
            new_generation[i+1, :] = twochildren[1, :]
        for k in range(npop):
            mutedInd = gaussian_mutate(
                new_generation[k, :], x_min, x_max, sigma, pmute)
            new_generation[k, :] = mutedInd
        pop = new_generation.copy()
        fit = fitness(pop, npop)
        best_hist.append(max(fit))
        avg_best_hist = (sum(best_hist)/len(best_hist))
    return pop, best_hist, avg_best_hist


# with elitism
def run_real_GA_with_Elitism(npop, x_min, x_max, ngen, pcross, pmute, sigma):

    best_hist = []
    avg_best_hist = 0
    pop = create_initial_pop(npop, x_min, x_max)
    new_generation = np.zeros_like(pop)
    fit = fitness(pop, npop)
    elitism = Elitism(pop, fit)
    new_generation[0, :] = elitism[0, :]
    new_generation[1, :] = elitism[1, :]
    for i in range(ngen):
        for j in range(npop-2, 2):
            twoparents = tournament_selection(pop, k)
            twochildren = arithmetic_crossover(twoparents, pcross)
            new_generation[i+2, :] = twochildren[0, :]
            new_generation[i+3, :] = twochildren[1, :]
        for k in range(npop-2):
            mutedInd = gaussian_mutate(
                new_generation[k, :], x_min, x_max, sigma, pmute)
            new_generation[k+2, :] = mutedInd
        pop = new_generation.copy()
        fit = fitness(pop, npop)
        best_hist.append(max(fit))
        avg_best_hist = math.floor(sum(best_hist)/len(best_hist))
        elitism = Elitism(pop, fit)
        new_generation[0, :] = elitism[0, :]
        new_generation[1, :] = elitism[1, :]
    return pop, best_hist, avg_best_hist


# to run the code without elitism
print(run_real_GA(20, [-2, -2], [2, 2], 100, 0.6, 0.05, 0.5))

# to run the code with elitism
#print(run_real_GA_with_Elitism(20, [-2, -2], [2, 2], 100, 0.6, 0.05, 0.5))
