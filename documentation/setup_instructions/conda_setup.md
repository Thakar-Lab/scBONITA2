# Setting up a conda environment

> NOTE: scBONITA only runs on Linux environments, if you use Windows please download and install Windows Subsystem for Linux (WSL) [here](https://learn.microsoft.com/en-us/windows/wsl/install)

## Installing Anaconda
1. Install Anaconda from https://www.anaconda.com/download
2. Run the Anaconda installer in the terminal on Linux or WSL (Windows Subsystem for Linux)
   - `bash Anaconda3-20204.09-0-Linux-x86_64.sh` (if the file you downloaded is different, use that file name)
   - Follow the prompts to install
3. Once Anaconda is installed, close and re-open your terminal. 
    - You should see `(base)` before your username

## Creating the scBonita conda environment
1. To create the correct conda environment, navigate to the `scBONITA2` directory that you cloned from before in the terminal. Once at the correct directory, enter:
    ```bash
    conda env create --file spec-file.txt --name scBonita
    ```

    > You can use mamba to help solve the conda environment faster. 
    > - Install mamba using `conda install mamba -n base -c conda-forge`
    > - Use  `mamba env create --file spec-file.txt --name scBonita` to create the environment

2. Once conda has finished working, you can confirm that the environment was created by entering `conda activate scBonita`. This will switch you into the correct environment to work with scBONITA.
   - A conda environment is basically a pre-packaged python installation that has a specific python version and package list that works with the code. This makes it so that you don't have to install each required package one-by-one, and you can have different package versions by having different conda environments

## Testing that the scBonita conda environment is working
1. Ensure that the `scBonita` conda environment is active:
    ```bash 
    conda activate scBonita
    ```
2. Make sure you are in the project folder (`scBONITA2`) in your terminal
3. Run the test data for the scBONITA pipeline:
    ```bash
    bash bash_scripts/local_george_hiv.sh
    ```

The `local_george_hiv.sh` file can be copied and modified to run different datasets / conditions. When running scBONITA, place bash scripts into this folder to run. scBONITA will automatically create the necessary output files in a directory called `scBONITA_output`. Make sure that you have the `george_data` directory downloaded

> If a package is missing, download it using the command `conda install <PACKAGE_NAME>` or `conda install -c conda-forge <PACKAGE_NAME>`