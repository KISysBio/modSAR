

## How to run the notebooks

1. Download the source code either from the `Code` button above to download zip or checkout project from Github

    a. If checking out, use Gitkraken to change to branch modsar-OSM or run `git clone --single-branch --branch modsar-OSM git@github.com:KISysBio/modSAR.git`

2. Create a folder `data` and move `Master Chemical List - annotated.xlsx` to the data folder.

3. Install Docker. Read the instructions for [Mac](https://docs.docker.com/docker-for-mac/install/), [Ubuntu](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-20-04), [Windows](https://docs.docker.com/docker-for-windows/install/) 

4. Install [`docker-compose`](https://docs.docker.com/compose/install/)

5. Open a terminal and type `docker-compose build` to let Docker download all the required dependencies and libraries automatically

6. Type `docker-compose up` to start Jupyter notebook. Click on the link that will show up on the terminal and run the notebooks:
    - `OSM-S4 - Notebook 01 - Preprocessing.ipynb`
    - `OSM-S4 - Notebook 02 - Running modSAR.ipynb`
