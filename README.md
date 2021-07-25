Source code of our paper:

> Cardoso-Silva, J., Papageorgiou, L. G. & Tsoka, S. **Network-based piecewise linear regression for QSAR modelling.** J. Comput. Aided. Mol. Des. 33, 831–844 (2019). http://link.springer.com/10.1007/s10822-019-00228-6

(update 14/03/2021: We're re-organizing the notebooks, a new more didatic version of this source code should be available soon)

Checkout [this notebook](https://github.com/KISysBio/modSAR/blob/master/processing/notebooks/2021-07-jon-osm-s4-predict-evariste-compounds.ipynb) for a newer practical example.

## How to run the notebooks

1. Download the source code either from the `Code` button above to download zip or checkout project from Github.

2. Create a folder `data` and move `Master Chemical List - annotated.xlsx` to the data folder.

3. Install Docker. Read the instructions for [Mac](https://docs.docker.com/docker-for-mac/install/), [Ubuntu](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-20-04), [Windows](https://docs.docker.com/docker-for-windows/install/)

4. Install [`docker-compose`](https://docs.docker.com/compose/install/)

5. Open a terminal and type `docker-compose build` to let Docker download all the required dependencies and libraries automatically

6. Type `docker-compose up` to start Jupyter notebook. Click on the link that will show up on the terminal and run the notebooks:
    - `OSM-S4 - Notebook 01 - Preprocessing.ipynb`
    - `OSM-S4 - Notebook 02 - Running modSAR.ipynb`

By default, modSAR will solve the equations with [GLPK](https://www.gnu.org/software/glpk/) solver but it can also work with [CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio/pricing) if you have an academic or commercial license to run MIP algorithms. In that case, pass `solver_name="cplex"` when creating an instance `ModSAR` class.
