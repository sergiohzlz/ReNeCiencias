import numpy as np
import pmc
from pyevolve import G1DList
from pyevolve import GSimpleGA
from pyevolve import Selectors
from pyevolve import Initializators, Mutators
from pyevolve import Consts
from pyevolve import Crossovers

NENT, NMED, NSAL = 2, 5, 1
RMIN, RMAX = -1.5,1.5
MUT = 0.01
GENS = 500

ens, sals = pmc.leer_data_set('resta.dat',NENT,NSAL)
nn = pmc.PMC(NENT,NMED,NSAL)
def evaluacion(genotipo, ens, sals):
    nn = pmc.PMC(NENT,NMED,NSAL)
    nn.cromosoma = genotipo
    error = 0.0
    for e,s in zip(ens,sals):
        error += nn.error(e,s)
    return error

def eval_func(chromosome):
    return evaluacion(chromosome, ens, sals)


crorig = nn.cromosoma
gnm = G1DList.G1DList(len(crorig))
gnm.setParams(rangemin=RMIN, rangemax=RMAX, bestRawScore=0.00, roundDecimal=2)
gnm.crossover.set(Crossovers.G1DListCrossoverTwoPoint)
gnm.initializator.set(Initializators.G1DListInitializatorReal)
gnm.evaluator.set(eval_func)

ga = GSimpleGA.GSimpleGA(gnm)
ga.selector.set(Selectors.GTournamentSelector)
ga.setMutationRate(MUT)
ga.minimax = Consts.minimaxType['minimize']
ga.setGenerations(GENS)
#ga.terminationCriteria.set(GSimpleGA.RawScoreCriteria)
ga.evolve(10)


print ga.bestIndividual()
