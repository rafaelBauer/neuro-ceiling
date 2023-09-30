## Setup environment

To set up the environment, one can simply create a conda environment
based on the YAML file and activate it by running

``` shell
conda env create -f environment.yml
conda activate neuro-ceiling
```

### Update YAML file from environment
```
conda env export > environment.yml
```

### Update the environment from YAML file

```
conda env update --name neuro-ceiling --file environment.yml  --prune
```
