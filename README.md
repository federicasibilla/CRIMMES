# Consumer Resource Interactions Model for Microbial Ecology in Space
## Repository for Federica Sibilla's Master Thesis project
## Repository collaborators: Simon van Vliet and Alessia del Panta

### CONTENT

* MODEL

Contains material to use the model for different projects or questions. 

    ** shared_scripts
        * *definitions.py*: stores examples of functions to generate C and D matrices
        * *N_dynamics.py* : stores the function pertaining growth and reproduction on the grid
        * *R_dynamics.py* : stores the functions pertaining dynamics of resources on the grid
        * *SOR.py*        : stores the main algorithm for finding equilibrium concentration of resources on the grid
        * *update.py*     : stores the different option for grid update and spatial simulations
        * *well_mixed.py* : stores all the R and N functions in well wixed, and the main simulation function   
        * mapping
            * *mapping_F.py*  : stores functions for mapping in the no-auxotrophs no-regulation case
            * *mapping_O.py*  : stores functions for the mapping in the auxotrophs leakage-regulated case
    ** running_scripts
        * CR_networks_generation
            * *protocol_1.py* : example of main script to generate networks, follows generation protocol 1 
        * CR_to_LV
            * *CR_to_LV.py*   : script to map networks onto LV (input is CR networks in a pickled dataframe)
        * spatial
            * *spatial_CR.py* : script to run the spatial version of the CR networks (input is wm CR in a pickled dataframe)
    ** process_data
        * *protocol_1.ipynb*  : example of how to process data from network generation to use it for mapping
        * *CR_to_LV.ipynb*    : example of how to process data from mapping
        * *spatial_CR.ipynb*  : example of how to process data from spatial simulations
    ** slurms : examples of slurms scripts to run each of the different running_scripts

* SIMULATIONS

    ** networks_dataset       : scripts used for the networks dataset generation
    ** spatial_simulations    : scripts used for the spatial runs of the subsampled networks
    ** auxotrophs             : scripts used to run the WM and spatial experiment of auxotrophic communities

* ANALYSIS : notebooks to analyze the thesis data

COMPLETE DATA IS AVAILABLE UPON REQUEST