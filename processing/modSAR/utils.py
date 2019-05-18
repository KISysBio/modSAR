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


def get_similarity_matrix(fingerprints):
    """Calculate the similarity matrix among all fingerprints

    """

    similarityMatrix = np.zeros((fingerprintDF.shape[0], fingerprintDF.shape[0]), dtype="f8")
    for i in range(fingerprintDF.shape[0]):
        for j in range(fingerprintDF.shape[0]):
            if i == j:
                similarityMatrix[i, j] = 0
            elif i < j:
                similarityMatrix[i, j] = TanimotoSimilarity(fingerprintDF.iloc[i]["FP_OBJ"], fingerprintDF.iloc[j]["FP_OBJ"])
                similarityMatrix[j, i] = similarityMatrix[i, j]
    return pd.DataFrame(similarityMatrix, index=fingerprintDF.index, columns=fingerprintDF.index)


def create_graph(adjMatrix, threshold, k=5, is_directed=True, is_weighted=True):

    thresholdMat = (adjMatrix >= threshold) * adjMatrix
    thresholdMat = thresholdMat.as_matrix().tolist()

    g = ig.Graph.Weighted_Adjacency(thresholdMat, loops=False)

    if not is_directed:
        if is_weighted:
            w = g.es["weight"]
            g.to_undirected()
            g.es["weight"] = w
        else:
            g.to_undirected()

    belowThreshold = [i for i, s in enumerate(g.strength(mode=1)) if s < k]
    edgeId = len(g.es())

    for n in belowThreshold:
        neighbours = np.flip(np.argsort(adjMatrix.iloc[n].values), axis=0)[0:k]
        for neigh in neighbours:
            shouldAdd = not g.es.select(_source=n, _target=neigh) and (is_directed or (not is_directed and not g.es.select(_source=neigh, _target=n)))
            if shouldAdd:
                g.add_edge(n, neigh)
                if is_weighted:
                    g.es[edgeId]["weight"] = adjMatrix.iloc[n, neigh]
                edgeId += 1
        g.vs[n]["usedKnn"] = True

    g.vs["degree"] = g.degree()

    if is_weighted:
        g["globalClusteringCoefficient"] = g.transitivity_avglocal_undirected(weights="weight", mode="zero")
    else:
        g["globalClusteringCoefficient"] = g.transitivity_avglocal_undirected(mode="zero")

    if is_weighted:
        vc = g.community_multilevel(weights='weight')
    else:
        vc = g.community_multilevel()
    g.vs["louvain"] = ['m%02d' % (x + 1) for x in vc.membership]
    g.vs["label"] = adjMatrix.columns
    g["threshold"] = threshold
    g["k"] = k

    # g["edgeDensity"] = g.density()
    # g["metric"] = g["globalClusteringCoefficient"] - 0.4 * g["edgeDensity"]
    return g
