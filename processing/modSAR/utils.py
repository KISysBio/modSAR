import igraph as ig
import numpy as np
import pandas as pd
from pyomo.core.base.param import IndexedParam
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity


# Print iterations progress
def print_progress_bar(iteration, total, prefix='', suffix='',
                       decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def print_var(var, *args):
    isParam = isinstance(var, IndexedParam)
    print("\n---- VARIABLE %s:" % var.name)
    if isParam:
        if len(args) == 0:
            print(var)
        elif len(args) == 1:
            for i in args[0]:
                print(i, var[i])
        else:
            for tup in tuple((i, j) for i in args[0] for j in args[1]):
                print(tup, var[tup])
    else:
        if len(args) == 0:
            print(var.value)
        elif len(args) == 1:
            if type(args[0]) is list:
                if type(args[0][0]) is str and all(element == str(args[0])[0] for element in map(lambda x: x[0], args[0])):
                    elements = sorted(args[0], key=lambda x: int(x[1:]))
                else:
                    elements = sorted(args[0])
            else:
                elements = sorted(args[0])
            for element in elements:
                print('{:10}'.format(element), var[element].value)
        else:
            samples = sorted(args[1])

            regions = sorted(args[0])
            print("          \t" + "\t".join('{:7}'.format(str(r)) for r in regions))
            for sample in samples:
                print('{:20}'.format(sample), end="\t")
                for region in regions:
                    value = float(var[(region, sample)].value)
                    print("%2.5f" % value, end="\t")
                print()


def round_decimal(number, decimal_places=2):
    decimal_value = Decimal(number)
    return decimal_value.quantize(Decimal(10) ** -decimal_places)
