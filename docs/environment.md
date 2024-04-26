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

## Setup ManiSkill2 env

Download an example ReplicaCAD scene from Habitat
```
wget https://dl.fbaipublicfiles.com/habitat/ReplicaCAD/hab2_bench_assets.zip -P data
cd data && unzip -q hab2_bench_assets.zip -d hab2_bench_assets
```

## Remote Development X11 Forwarding
To use it make sure the ~/.ssh/config contains the following:

```
Host *
ForwardX11 yes
ForwardX11Trusted yes
```