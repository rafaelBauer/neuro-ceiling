# Neuro-CEILing

# Dependencies:

All the dependencies are in the environment.yml file. In order to 
have them installed, all you need is to have the following installed
 - Python
 - Conda

# How to

## Setup environment

To set up the environment, one can simply create a conda environment
based on the YAML file and activate it by running
 ```
 conda env create -f environment.yml
 conda activate neuro-ceiling
 ```

### Update YAML file from environment
```
conda env export > environment.yml
```

### Update the environment from YAML file

```
conda env update --prefix ./env --file environment.yml  --prune
```
