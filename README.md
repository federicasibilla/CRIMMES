# **Consumer Resource Interactions Model for Microbial Ecology in Space**  
**Repository for Federica Sibilla's Master Thesis Project**  
**Collaborators:** Simon van Vliet, Alessia del Panta  

Check the environment.yml file for requirements.

---

## **CONTENT**  

### **MODEL**  
Contains materials to use the model for different projects or research questions.  

#### **Shared Scripts**  
- **`definitions.py`**: Stores examples of functions to generate `C` and `D` matrices.  
- **`N_dynamics.py`**: Contains functions for growth and reproduction on the grid.  
- **`R_dynamics.py`**: Contains functions for resource dynamics on the grid.  
- **`SOR.py`**: Implements the main algorithm for finding equilibrium resource concentrations on the grid.  
- **`update.py`**: Provides different options for grid updates and spatial simulations.  
- **`well_mixed.py`**: Contains all resource (`R`) and population (`N`) functions for well-mixed systems, along with the main simulation function.  

##### **Mapping**  
- **`mapping_F.py`**: Functions for mapping in the no-auxotrophs, no-regulation case.  
- **`mapping_O.py`**: Functions for mapping in the auxotrophs leakage-regulated case.  

#### **Running Scripts**  
- **CR Networks Generation**  
  - **`protocol_1.py`**: Example script to generate networks following Generation Protocol 1.  
- **CR to LV Mapping**  
  - **`CR_to_LV.py`**: Script to map CR networks onto Lotka-Volterra (LV) models. Input: CR networks in a pickled dataframe.  
- **Spatial Simulations**  
  - **`spatial_CR.py`**: Script to run the spatial version of CR networks. Input: Well-mixed CR networks in a pickled dataframe.  

#### **Process Data**  
- **`protocol_1.ipynb`**: Example notebook for processing data from network generation for mapping.  
- **`CR_to_LV.ipynb`**: Example notebook for processing data from CR-to-LV mapping.  
- **`spatial_CR.ipynb`**: Example notebook for processing data from spatial simulations.  

#### **Slurm Scripts**  
Examples of Slurm scripts to run each of the different running scripts.  

---

### **SIMULATIONS**  

#### **Networks Dataset**  
Scripts used for generating the networks dataset.  

#### **Spatial Simulations**  
Scripts used for running spatial simulations of subsampled networks.  

#### **Auxotrophs**  
Scripts used to run well-mixed (WM) and spatial experiments for auxotrophic communities.  

---

### **ANALYSIS**  
Notebooks for analyzing thesis data.  

---

### **DATA AVAILABILITY**  
Complete data is available upon request.  
