# scBONITA2
Infers Boolean molecular signaling networks using scRNAseq data and prior knowledge networks, performs attractor analysis, and calculates the importance of each node in the network

## Setup:

### Cloning the scBONITA2 repository
In your terminal, navigate to the directory where you want to install this project and enter 
```bash
git clone https://github.com/Luminarada80/scBONITA2.git
```

Navigate to the `scBONITA2` directory
```bash
cd scBONITA2
```

## Setting up the environment
There are three options for setting up the environment for scBONITA2:
1. **Docker**: This is the easiest and most reliable method
2. **Conda**: This requires you to install a version of conda and create an scBonita conda environment. 
For Mac users with the M1 series chip, use the Docker instructions or download packages individually.
3. **Manually install packages**: This will take a bit longer, but will get around issues with creating the environment.

### Installation guides:
- [Docker installation guide](documentation/setup_instructions/docker_setup.md)
- [Conda installation guide](documentation/setup_instructions/conda_setup.md)
- [BlueHive installation guide](documentation/setup_instructions/bluehive_setup.md)
- [Manual installation guide](documentation/setup_instructions/manual_setup.md)

## Running scBONITA2
Once you have followed one of these guides to set up the environment, try running the [tutorial data](documentation/tutorial.md) and read the ["How scBONITA2 works"](documentation/running_scBONITA2.md) documentation. Each step has a more detailed guide explaining how it all works.