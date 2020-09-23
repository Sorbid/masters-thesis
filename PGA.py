#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.
from matplotlib import cm 
from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib.pyplot as plt
import numpy
import random

from collections import deque
from multiprocessing import Event, Pipe, Process
from math import pi
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

# Attribute generator
#Функция Растрыгина
#toolbox.register("attr_bool", random.uniform, -5.12, 5.12)
#toolbox.register("attr_bool", random.random)
#Функция Швефеля
#toolbox.register("attr_bool", random.uniform, -500, 500)
#Функция Стыбинского - Танга:
toolbox.register("attr_bool", random.randint, 0,1)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, 
                 toolbox.attr_bool, 100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalOneMax(individual):
    A = -4.189829
    return A * len(individual) + sum([(x * numpy.sin(numpy.sqrt(numpy.abs(x)))) for x in individual]),

    """
    Функция Швефеля:
    A = 4.189829
    return A*len(individual) + sum([(x * numpy.sin(numpy.sqrt(numpy.abs(x)))) for x in individual]),

    Функция Стыбинского - Танга:
    return sum([(x**4 - 16*x**2 + 5*x) for x in individual])/2,

    Функция Растрыгина:
    #A = 10
    #return A*len(individual) + sum([(x**2 - A * numpy.cos(2 * pi * x)) for x in individual]),
    """



def migPipe(deme, k, pipein, pipeout, selection, replacement=None):
    """Migration using pipes between initialized processes. It first selects
    *k* individuals from the *deme* and writes them in *pipeout*. Then it
    reads the individuals from *pipein* and replace some individuals in the
    deme. The replacement strategy shall not select twice the same individual.
    
    :param deme: A list of individuals on which to operate migration.
    :param k: The number of individuals to migrate.
    :param pipein: A :class:`~multiprocessing.Pipe` from which to read
                   immigrants.
    :param pipeout: A :class:`~multiprocessing.Pipe` in which to write
                    emigrants. 
    :param selection: The function to use for selecting the emigrants.
    :param replacement: The function to use to select which individuals will
                        be replaced. If :obj:`None` (default) the individuals
                        that leave the population are directly replaced.
    """
    emigrants = selection(deme, k)
    if replacement is None:
        # If no replacement strategy is selected, replace those who migrate
        immigrants = emigrants
    else:
        # Else select those who will be replaced
        immigrants = replacement(deme, k)
    
    pipeout.send(emigrants)
    buf = pipein.recv()
    
    for place, immigrant in zip(immigrants, buf):
        indx = deme.index(place)
        deme[indx] = immigrant

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main(procid, pipein, pipeout, sync):
    toolbox.register("migrate", migPipe, k=1, pipein=pipein, pipeout=pipeout,
                     selection=tools.selBest, replacement=random.sample)

    MU = 1000
    NGEN = 100
    CXPB = 0.5
    MUTPB = 0.2
    MIG_RATE = 10
    
    deme = toolbox.population(n=MU)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    logbook = tools.Logbook()
    logbook.header = "gen", "deme", "evals", "std", "min", "avg", "max"
    
    for ind in deme:
        ind.fitness.values = toolbox.evaluate(ind)
    record = stats.compile(deme)
    logbook.record(gen=0, deme=procid, evals=len(deme), **record)
    hof.update(deme)
    
    if procid == 0:
        # Synchronization needed to log header on top and only once
        print(logbook.stream)
        sync.set()
    else:
        logbook.log_header = False  # Never output the header
        sync.wait()
        print(logbook.stream)
    
    for gen in range(1, NGEN):
        deme = toolbox.select(deme, len(deme))
        deme = algorithms.varAnd(deme, toolbox, cxpb=CXPB, mutpb=MUTPB)
        
        invalid_ind = [ind for ind in deme if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind)
        
        hof.update(deme)
        record = stats.compile(deme)
        logbook.record(gen=gen, deme=procid, evals=len(deme), **record)
        print(logbook.stream)
            
        if gen % MIG_RATE == 0 and gen > 0:
            toolbox.migrate(deme)
    print(tools.selBest(deme, 1)[0], tools.selBest(deme, 1)[0].fitness.values)


if __name__ == "__main__":
    
    NBR_DEMES = 3
    
    pipes = [Pipe(False) for _ in range(NBR_DEMES)]
    pipes_in = deque(p[0] for p in pipes)
    pipes_out = deque(p[1] for p in pipes)
    pipes_in.rotate(1)
    pipes_out.rotate(-1)
    
    e = Event()
    
    processes = [Process(target=main, args=(i, ipipe, opipe, e)) for i, (ipipe, opipe) in enumerate(zip(pipes_in, pipes_out))]
    
    for proc in processes:
        proc.start()
    
    for proc in processes:
        proc.join()

    # График функции Растригина
    #X = numpy.linspace(-4, 4, 50)    
    #Y = numpy.linspace(-4, 4, 50)    
    #X, Y = numpy.meshgrid(X, Y)

    #Z = ((X**2 - 10 * numpy.cos(2 * pi * X)) +
    #     (Y**2 - 10 * numpy.cos(2 * pi * Y)) + 20)

    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #surf = ax.plot_surface(X, Y, Z, rstride=1,
    #       cstride=1, cmap=cm.jet)
    #plt.savefig('Растригин.png')

    # График функции Швефеля
    #X = numpy.linspace(-500, 500, 100)
    #Y = numpy.linspace(-500, 500, 100)
    #X, Y = numpy.meshgrid(X, Y)
    #Z = (X * numpy.sin(numpy.sqrt(numpy.abs(X))) +
    #     Y * numpy.sin(numpy.sqrt(numpy.abs(X))) + 418.9829)
    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #surf = ax.plot_surface(X, Y, Z, rstride=1,
    #       cstride=1, cmap=cm.jet)
    #plt.savefig('Швефель.png')

    # График функции Швефеля
    #X = numpy.linspace(-5, 5, 100)
    #Y = numpy.linspace(-5, 5, 100)
    #X, Y = numpy.meshgrid(X, Y)
    #Z = ((X ** 4 - 16 * X ** 2 + 5 * X) +
    #     (Y ** 4 - 16 * Y ** 2 + 5 * Y) /2)
    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #surf = ax.plot_surface(X, Y, Z, rstride=1,
    #       cstride=1, cmap=cm.jet)
    ##plt.savefig('Швефель.png')
    #plt.show()
