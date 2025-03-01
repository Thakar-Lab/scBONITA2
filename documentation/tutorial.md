# scBONITA2 Tutorial

### Download and set up the environment for scBONITA2
1. [Download](../README.md#cloning-the-scbonita2-repository) the scBONITA2 repository and [set up the scBONITA2 environment](../README.md#setting-up-the-environment).
2. Navigate to `scBONITA` in your terminal.


### Docker
If you installed Docker, simply run:
```bash
bash bash_scripts/docker_tutorial.sh
```
The script will automatically create a Docker image and container to run scBONITA. The output will be saved locally to the `output` directory. 

### Conda environment
If you installed the scBonita conda environment, activate it using:
```bash
conda activate scBonita
```

and run the `local_tutorial.sh` bash file using:

```bash
bash bash_scripts/local_tutorial.sh
```

### Manual installation
If you installed the required packages manually:

```bash
bash bash_scripts/local_tutorial.sh
```

