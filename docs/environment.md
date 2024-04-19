## Setup Conda environment

To set up the environment, one can simply create a conda environment
based on the YAML file and activate it by running

``` shell
conda env create -f environment.yml
conda activate neuro-ceiling
```

### Update YAML file from environment
```
conda env export -f environment.yml --no-builds
```

### Update the environment from YAML file

```
conda env update --name neuro-ceiling --file environment.yml  --prune
```
