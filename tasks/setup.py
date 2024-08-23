# This script uses the Invoke library to automate several setup tasks on an Ubuntu LTS 20.04 or 22.04 system.
# Tasks include checking NVMe drives, setting them up, installing Docker, installing Anaconda,
# setting up a Conda environment, and downloading AzCopy for easy file transfer.
# If needed, install CUDA and drivers by following the instructions at https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.htm

from invoke import task

from pipelines.utils import get_logger

logger = get_logger(__name__)


@task
def setup_nvme(c):
    # See if your VM has an NVMe drive
    c.run("sudo apt install -y nvme-cli")

    nvme_output = c.run("sudo nvme list", hide=True, warn=True)
    if not nvme_output.ok:
        logger.info("Failed to list NVMe devices. Skipping.")
        return

    # Extract NVMe device names from the output
    nvme_devices = []
    for line in nvme_output.stdout.splitlines():
        if "/dev/nvme" in line:
            nvme_devices.append(line.split()[0])

    if not nvme_devices:
        logger.info("Did not find any NVMe disks. Skipping.")
        return

    # Use the first NVMe device found
    nvme_device = nvme_devices[0]

    logger.info(f"Formatting and mounting the NVMe drive: {nvme_device}")
    # Format it with XFS
    c.run(f"sudo mkfs.xfs {nvme_device}")

    # Mount it
    c.run("sudo mkdir -p /mnt/ephemeral_nvme")
    c.run(f"sudo mount {nvme_device} /mnt/ephemeral_nvme")

    # Enable read and write to this disk for the user
    c.run("sudo chown $USER /mnt/ephemeral_nvme")
    c.run("mkdir -p /mnt/ephemeral_nvme/workdir/")

    # Link ./workdir to point to a folder on the NVMe
    c.run("ln -sfn /mnt/ephemeral_nvme/workdir/ ./workdir")


@task
def install_docker(c):
    """
    Task to install Docker on an Ubuntu system by following https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository

    Args:
        c (invoke.context.Context): The context instance (passed automatically by the @task decorator).
    """
    logger.info("Installing Docker")

    # Add Docker's official GPG key:
    c.run("sudo apt-get update")
    c.run("sudo apt-get install ca-certificates curl")
    c.run("sudo install -m 0755 -d /etc/apt/keyrings")
    c.run(
        "sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc"
    )
    c.run("sudo chmod a+r /etc/apt/keyrings/docker.asc")

    # Add the repository to Apt sources:
    c.run(
        'echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] '
        'https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | '
        "sudo tee /etc/apt/sources.list.d/docker.list > /dev/null"
    )
    c.run("sudo apt-get update")

    # Install docker
    c.run(
        "sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin"
    )

    # Set the permissions for the current user
    c.run("sudo usermod -a -G docker $USER")
    c.run("newgrp docker")


@task
def install_anaconda(c):
    """
    Installs Anaconda if it is not already installed.

    This task checks if Anaconda (conda) is already installed on the system. If it is not installed,
    it downloads the Anaconda installer for Linux, runs the installer, and then removes the installer file.

    Args:
        c (invoke.context.Context): The context instance (passed automatically by the @task decorator).
    """
    if c.run("conda", hide=True, warn=True).ok:
        logger.info("Conda is already installed.")
        return
    logger.info("Installing Anaconda")
    c.run(
        "curl -o Anaconda-latest-Linux-x86_64.sh "
        "https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh"
    )
    c.run("bash Anaconda-latest-Linux-x86_64.sh")
    c.run("rm Anaconda-latest-Linux-x86_64.sh")


@task(pre=[install_anaconda])
def setup_conda_env(c):
    """
    Sets up the Conda environment using the environment file.

    This task creates a Conda environment based on the specifications in the 'conda_env.yml' file.
    After creating the environment, it activates the environment named 'wikichat' and downloads
    the 'en_core_web_sm' model for spaCy.

    Args:
        c (invoke.context.Context): The context instance (passed automatically by the @task decorator).
    """
    logger.info("Creating the Conda environment")
    c.run("conda env create --file conda_env.yml")
    c.run("conda activate wikichat")
    c.run("python -m spacy download en_core_web_sm")


@task
def download_azcopy(c):
    """
    Downloads and installs AzCopy, a command-line utility for copying data to and from Microsoft Azure.

    This task performs the following steps:
    1. Downloads the AzCopy tarball from the official Microsoft Azure link.
    2. Extracts the contents of the tarball.
    3. Removes the downloaded tarball to clean up.
    4. Creates a directory named 'bin' in the user's home directory if it doesn't already exist.
    5. Moves the AzCopy executable to the 'bin' directory.
    6. Removes the extracted directory to clean up.
    7. Adds the 'bin' directory to the user's PATH environment variable for the current session.
    8. Appends the PATH update to the user's .bashrc file to make the change permanent.

    Args:
        c (invoke.context.Context): The context instance (passed automatically by the @task decorator).
    """
    logger.info("Downloading Azcopy")
    c.run("wget https://aka.ms/downloadazcopy-v10-linux")
    c.run("tar -xvf downloadazcopy-v10-linux")
    c.run("rm downloadazcopy-v10-linux")
    c.run("mkdir -p ~/bin")
    c.run("mv ./azcopy_linux_amd64_*/azcopy ~/bin/")
    c.run("rm -r ./azcopy_linux_amd64_*/")
    c.run('export PATH="$PATH:~/bin/"')
    c.run("echo 'export PATH=$PATH:~/bin/' >> ~/.bashrc")


@task(
    post=[
        download_azcopy,
        setup_conda_env,
        install_anaconda,
        install_docker,
        setup_nvme,
    ]
)
def install(c):
    """
    Installs various tools and sets up the environment.

    This task orchestrates the installation and setup of several tools and environments required for the project.
    It performs the following steps in sequence:
    1. Downloads and installs AzCopy, a command-line utility for copying data to and from Microsoft Azure.
    2. Sets up the Conda environment using the specifications in the 'conda_env.yml' file and downloads the 'en_core_web_sm' model for spaCy.
    3. Downloads and installs the Anaconda distribution if it is not already installed.
    4. Installs Docker, a platform for developing, shipping, and running applications inside containers.
    5. Sets up NVMe (Non-Volatile Memory Express) storage.

    Args:
        c (invoke.context.Context): The context instance (passed automatically by the @task decorator).
    """
    logger.info("Started installing...")
