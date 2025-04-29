# This script uses the Invoke library to automate several setup tasks on an Ubuntu LTS 20.04 or 22.04 system.
# Tasks include checking NVMe drives, setting them up, installing Docker, installing Anaconda,
# setting up a Conda environment, and downloading AzCopy for easy file transfer.
# If needed, install CUDA and drivers by following the instructions at https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.htm

from invoke import task

from utils.logging import logger


@task
def setup_nvme(c):
    """
    Set up an NVMe drive on the VM by performing the following steps. Only works on certain Linux distributions.

    1. Installs the `nvme-cli` package to manage NVMe devices.
    2. Lists available NVMe devices on the system.
    3. Extracts NVMe device names from the listing output.
    4. Checks if any NVMe devices are found; if none, logs a message and exits.
    5. Formats the first NVMe device found with the XFS filesystem.
    6. Creates a mount point at `/mnt/ephemeral_nvme`.
    7. Mounts the NVMe device to the created mount point.
    8. Changes ownership of the mount point to the current user to enable read and write access.
    9. Creates a `workdir` directory on the NVMe drive.
    10. Creates a symbolic link `./workdir` pointing to the `workdir` directory on the NVMe drive.

    Args:
        c: The context instance (passed automatically by invoke).
    """
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

    logger.info(f"Checking if the NVMe drive {nvme_device} is already formatted")
    # Check if the NVMe device is already formatted
    format_check = c.run(f"sudo file -s {nvme_device}", hide=True).stdout
    if "filesystem" in format_check:
        logger.info(
            f"The NVMe drive {nvme_device} is already formatted. Skipping formatting."
        )
    else:
        logger.info(f"Formatting the NVMe drive: {nvme_device}")
        # Format it with XFS
        c.run(f"sudo mkfs.xfs {nvme_device}")

    # Mount it
    if not c.run("sudo test -d /mnt/ephemeral_nvme", warn=True).ok:
        c.run("sudo mkdir -p /mnt/ephemeral_nvme")
    c.run(f"sudo mount {nvme_device} /mnt/ephemeral_nvme")

    # Enable read and write to this disk for the user
    c.run("sudo chown $USER /mnt/ephemeral_nvme")
    if not c.run("sudo test -d /mnt/ephemeral_nvme/workdir", warn=True).ok:
        c.run("mkdir -p /mnt/ephemeral_nvme/workdir/")

    # Link ./workdir_nvme to point to a folder on the NVMe
    if not c.run("test -L ./workdir_nvme", warn=True).ok:
        c.run("ln -sfn /mnt/ephemeral_nvme/workdir/ ./workdir_nvme")
    else:
        logger.info("Symbolic link ./workdir_nvme already exists. Skipping.")


@task
def install_docker(c):
    """
    Install Docker on an Ubuntu system by following https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository

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
def install_pixi(c):
    """
    Install Pixi if it is not already installed.

    This task checks if Pixi is already installed on the system. If it is not installed,
    it downloads the Pixi installer for Linux and runs the installer.

    Args:
        c (invoke.context.Context): The context instance (passed automatically by the @task decorator).
    """
    if c.run("pixi", hide=True, warn=True).ok:
        logger.info("Pixi is already installed.")
        return
    logger.info("Installing Pixi")
    c.run("curl -fsSL https://pixi.sh/install.sh | sh")


@task
def download_azcopy(c):
    """
    Download and install AzCopy, a command-line utility for copying data to and from Microsoft Azure.

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
        install_pixi,
        install_docker,
        setup_nvme,
    ]
)
def install(c):
    """
    Install various tools and set up the environment.

    This task orchestrates the installation and setup of several tools and environments required for the project.
    It performs the following steps in sequence:
    1. Downloads and installs AzCopy, a command-line utility for copying data to and from Microsoft Azure.
    2. Sets up the Conda environment using the specifications in the 'conda_env.yaml' file and downloads the 'en_core_web_sm' model for spaCy.
    3. Downloads and installs the Anaconda distribution if it is not already installed.
    4. Installs Docker, a platform for developing, shipping, and running applications inside containers.
    5. Sets up NVMe (Non-Volatile Memory Express) storage.

    Args:
        c (invoke.context.Context): The context instance (passed automatically by the @task decorator).
    """
    logger.info("Started installing...")
