# Setting up scBONITA2 on BlueHive
BlueHive is the HPC at the University of Rochster. These instructions will only apply to individuals who have access and are using BlueHive. These instructions are similar to the instructions for [setting up a conda environment](conda_setup.md), but setting up conda environments on BlueHive is more restrictive and requires a few more steps.

## Installing Anaconda on BlueHive
1. Open the terminal and run `module load anaconda3`
2. Run `conda init`
3. Restart the terminal
4. Run `conda create -n myenv`
5. Run `conda activate myenv`
6. Run `conda install -c anaconda git`
7. Install mamba using `conda install mamba`
    - Note: This can take a significant amount of time, be patient
8. Run `conda install git`
8. Navigate to the directory into which you would like to install scBONITA
9. Clone the repository using `git clone https://github.com/Luminarada80/scBONITA2.git`
10. Navigate into the directory using `cd scBONITA2`
11. Use `mamba env create --file spec-file.txt --name scBonita`
12. Activate the environment using `conda activate scBonita`
    - This will switch you into the correct environment to work with scBONITA.
    - A conda environment is basically a pre-packaged python installation that has a specific python version and package list that works with the code. This makes it so that you don't have to install each required package one-by-one, and you can have different package versions by having different conda environments

## Testing that the scBonita conda environment is working
1. Ensure that the `scBonita` conda environment is active (or enter `conda activate scBonita` to activate)
2. Make sure you are in the project folder (`scBONITA2`) in your terminal
3. Run the test data for the scBONITA pipeline:
    - `bash bash_scripts/local_tutorial.sh`

The first time you run scBONITA2, it will download the KEGG pathway xml files from KEGG. This takes a while to download, but makes the runtime of the rule determination step much faster.

The `local_tutorial.sh` file can be copied and modified to run different datasets / conditions. When running scBONITA, place bash scripts into this folder to run. scBONITA will automatically create the necessary output files in a directory called `scBONITA_output`.

> If a package is missing, download it using the command `mamba install <PACKAGE_NAME>` or `mamba install -c conda-forge <PACKAGE_NAME>`

## Notes:
This method installs `git` and `mamba` in the `myenv` environment. If you need to use `git`, you can either try installing it to your `scBonita` environment or you can use your `myenv` environment using `conda activate myenv`.