
# Installing scBONITA2 using Docker

### macOS:
1. Download Docker Desktop from [Docker's website](https://www.docker.com/products/docker-desktop).
2. Install Docker by dragging the Docker icon to the Applications folder.
3. Launch Docker from the Applications folder.
4. Follow the on-screen instructions to complete the setup.

### Linux:
1. Open a terminal.
2. Run the following commands to install Docker:
   ```bash
   sudo apt-get update
   sudo apt-get install docker.io
   sudo systemctl start docker
   sudo systemctl enable docker
   sudo usermod -aG docker $USER
   ```
3. Close and re-open the terminal

### Windows:
1. Download Docker Desktop from [Docker's official website](https://www.docker.com/products/docker-desktop).
2. Install Docker Desktop and follow the setup instructions.
3. Ensure that WSL 2 is installed (Docker Desktop will prompt you if it's not).

## Clone the `scBONITA2` Repository

1. Open a terminal.
2. Clone the `scBONITA2` repository from GitHub:
   ```bash
   git clone https://github.com/Luminarada80/scBONITA2.git
   ```
3. Navigate to the `scBONITA2` directory:
   ```bash
   cd scBONITA2
   ```

If you need to re-install the docker image without using a cached version, use:
```bash
docker build --no-cache -t scbonita .
```

## Running scBONITA2 with Docker
Use the `docker_george_hiv.sh` file to run scBONITA2 with Docker. This file will automatically configure the Docker environment and run the bash file in the environment.
```bash
sudo bash bash_scripts/docker_george_hiv.sh
```

