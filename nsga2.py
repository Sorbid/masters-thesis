import numpy
import random
from collections import deque
from multiprocessing import Event, Pipe, Process, Queue
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import pymop.factory
from math import factorial
from deap.benchmarks.tools import igd
import matplotlib.pyplot as plt
from pymop.factory import get_problem, get_uniform_weights
import time

start_time = time.time()
PROBLEM = "dtlz1"
NOBJ = 3
NDIM = NOBJ + 10 - 1
P = 30
BOUND_LOW, BOUND_UP = 0.0, 1.0
problem = pymop.factory.get_problem(PROBLEM, n_var=NDIM, n_obj=NOBJ)

# параметры алгоритма
MU = 60
NGEN = 500
CXPB = 1.0
MUTPB = 1.0

ref_points = get_uniform_weights(P, NOBJ)
pf = get_problem("dtlz1", n_var=NDIM, n_obj=NOBJ).pareto_front(ref_points)

# ref_points = tools.uniform_reference_points(NOBJ, P)
# pf = problem.pareto_front(ref_points)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * NOBJ)
creator.create("Individual", list, fitness=creator.FitnessMin)


def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]


toolbox = base.Toolbox()
toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", problem.evaluate, return_values_of=["F"])
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=30.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0 / NDIM)
toolbox.register("select", tools.selNSGA2)
toolbox.register("get_best", tools.selBest)
toolbox.register("get_worst", tools.selWorst)


def mig_parallel(deme, k, pipein, pipeout, selection, replacement):
    emigrants = selection(deme, k)
    if replacement is None:
        immigrants = emigrants
    else:
        immigrants = replacement(deme, k)

    pipeout.send(emigrants)
    buf = pipein.recv()

    for place, immigrant in zip(immigrants, buf):
        indx = deme.index(place)
        deme[indx] = immigrant


def main(queue, procid, pipein, pipeout, sync):
    toolbox.register("migrate", mig_parallel, k=1, pipein=pipein, pipeout=pipeout,
                     selection=toolbox.get_best, replacement=toolbox.get_worst)
    MIG_RATE = 13

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)
    deme = toolbox.population(n=MU)

    logbook = tools.Logbook()
    logbook.header = "gen", "deme", "evals", "std", "min", "avg", "max"

    invalid_ind = [ind for ind in deme if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    deme = toolbox.select(deme, len(deme))

    record = stats.compile(deme)
    logbook.record(gen=0, deme=procid, evals=len(deme), **record)

    if procid == 0:
        temp_stream = logbook.stream
        print(temp_stream)
        sync.set()
    else:
        logbook.log_header = False
        sync.wait()
        temp_stream = logbook.stream
        print(temp_stream)

    # for gen in range(1, NGEN):
        # offspring = algorithms.varAnd(deme, toolbox, cxpb=CXPB, mutpb=MUTPB)
        #
        # invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        # for ind, fit in zip(invalid_ind, fitnesses):
        #     ind.fitness.values = fit
        #
        # deme = toolbox.select(deme + offspring, MU)
        #
        # record = stats.compile(deme)
        # logbook.record(gen=gen, deme=procid, evals=len(deme), **record)
        # temp_stream = logbook.stream
        # print(temp_stream)
        #
        # if gen % MIG_RATE == 0 and gen > 0:
        #     toolbox.migrate(deme)
    for gen in range(1, NGEN):
        # Vary the population
        offspring = tools.selTournamentDCD(deme, len(deme))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        deme = toolbox.select(deme + offspring, MU)
        record = stats.compile(deme)
        logbook.record(gen=gen, deme=procid, evals=len(deme), **record)
        print(logbook.stream)

        if gen % MIG_RATE == 0 and gen > 0:
            toolbox.migrate(deme)

    queue.put([deme, logbook])
    # df_log = pd.DataFrame(logbook)


if __name__ == "__main__":
    NBR_DEMES = 6
    pipes = [Pipe(False) for _ in range(NBR_DEMES)]
    pipes_in = deque(p[0] for p in pipes)
    pipes_out = deque(p[1] for p in pipes)
    pipes_in.rotate(1)
    pipes_out.rotate(-1)
    e = Event()
    q = Queue()
    result = []
    processes = [Process(target=main, args=(q, i, ipipe, opipe, e)) for i, (ipipe, opipe) in
                 enumerate(zip(pipes_in, pipes_out))]

    [proc.start() for proc in processes]

    for proc in processes:
        res = q.get()
        result.append(res)

    [proc.join() for proc in processes]

    solutions = []

    for deme, logbook in result:
        pop_fit = numpy.array([ind.fitness.values for ind in deme])
        solutions.append(igd(pop_fit, pf))
    best_solution = min(solutions)
    index_solution = solutions.index(best_solution)
    deme, logbook = result[index_solution]

    print(solutions)
    print('{:.6f}'.format(best_solution))
    print('{:.4f}'.format(MU*NGEN*NBR_DEMES/(time.time() - start_time)))

    # fig = plt.figure(figsize=(7, 7))
    # ax = fig.add_subplot(111, projection="3d")
    #
    # p = numpy.array([ind.fitness.values for ind in deme])
    # ax.scatter(p[:, 0], p[:, 1], p[:, 2], marker="o", s=24, label="Решение")
    #
    # ax.scatter(pf[:, 0], pf[:, 1], pf[:, 2], marker="x", c="k", s=32, label="Лучшее решение")
    #
    # ax.scatter(ref_points[:, 0], ref_points[:, 1], ref_points[:, 2], marker="o", s=24, label="Относительные точки")
    #
    # ax.view_init(elev=26, azim=9)
    # ax.autoscale(tight=True)
    # plt.legend()
    # plt.tight_layout()
    # #plt.show()
    # plt.savefig(f"nsga2.png")
