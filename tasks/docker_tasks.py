import grp  # Import grp module to get group info
import os
import pwd  # Import pwd module to get user info
import sys

import docker
from invoke.tasks import task

sys.path.insert(0, "./")
from tasks.defaults import (
    DEFAULT_DISTILLED_MODEL_PORT,
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_EMBEDDING_MODEL_PORT,
    DEFAULT_NUM_GPUS,
    DEFAULT_VECTORDB_PORT,  # rest port. grpc port is +1
    DEFAULT_WORKDIR,
    QDRANT_VERSION,
    TEI_DOCKER_CONTAINER_HARDWARE_ARCHITECTURE,
    TEI_VERSION,
    TGI_VERSION,
)
from utils.docker_utils import (
    check_if_docker_container_is_running,
    get_docker_container_by_name,
    wait_for_docker_container_to_be_ready,
)
from utils.logging import logger

QDRANT_USER_ID = 1000
QDRANT_GROUP_ID = 1000  # the same as user id, but can be different


def check_and_warn_ownership(path, required_uid, required_gid):
    """Checks directory ownership and warns if incorrect."""
    if not os.path.exists(path):
        try:
            # Create the directory if it doesn't exist.
            # This might fail if parent directories don't have write permissions
            # for the user running the script.
            os.makedirs(path, exist_ok=True)
            logger.info(f"Created directory: {path}")
            # Attempt to set initial ownership if just created, might require sudo
            # Note: This os.chown will likely fail if the script isn't run as root
            try:
                os.chown(path, required_uid, required_gid)
                logger.info(f"Set ownership of {path} to {required_uid}:{required_gid}")
            except PermissionError:
                logger.warning(
                    f"Created directory {path}, but failed to set ownership to {required_uid}:{required_gid}. Manual 'sudo chown' might be required."
                )
            except OSError as e:
                logger.warning(
                    f"Created directory {path}, but encountered OS error setting ownership: {e}. Manual 'sudo chown' might be required."
                )

        except OSError as e:
            logger.error(
                f"Failed to create directory {path}: {e}. Please ensure parent directories exist and have correct permissions."
            )
            return False  # Cannot proceed if directory cannot be created

    # Check ownership again after potential creation/chown attempt
    if os.path.exists(path):
        try:
            stat_info = os.stat(path)
            if stat_info.st_uid != required_uid or stat_info.st_gid != required_gid:
                try:
                    owner_name = pwd.getpwuid(stat_info.st_uid).pw_name
                except KeyError:
                    owner_name = str(stat_info.st_uid)
                try:
                    group_name = grp.getgrgid(stat_info.st_gid).gr_name
                except KeyError:
                    group_name = str(stat_info.st_gid)

                logger.error(
                    f"Ownership mismatch for {path}. "
                    f"Required: {required_uid}:{required_gid}, Found: {stat_info.st_uid}:{stat_info.st_gid} ({owner_name}:{group_name}). "
                    f"Run 'sudo chown -R {required_uid}:{required_gid} {path}' to fix."
                )
                return False
        except Exception as e:
            logger.error(f"Could not check ownership for {path}: {e}")
            return False  # Indicate failure if stat fails
    else:
        # If the directory still doesn't exist after trying to create it
        logger.error(f"Directory {path} does not exist and could not be created.")
        return False

    return True  # Ownership is correct or directory created successfully with correct ownership


@task
def stop_docker_container(c, container_name_prefix):
    """
    Stops a specified Docker container if it is running.

    This function checks if a Docker container whose given name starts with `container_name_prefix` is running. If the container is already
    stopped, it logs this information and takes no further action. If the container is running, this function
    proceeds to stop it. It requires the Docker SDK for Python (`docker` package) to interact with Docker.

    Args:
        c: The context parameter for Invoke tasks, automatically passed by Invoke.
        container_name_prefix (str): The name of the Docker container to stop starts with this string.

    Note:
        This function requires Docker to be installed and running on the host system.
    """
    client = docker.from_env()
    # List all running containers
    running_containers = client.containers.list()

    for container in running_containers:
        if container.name.startswith(container_name_prefix):
            container.stop()
            container.remove()

    client.close()


@task
def start_qdrant_docker_container(
    c, workdir=DEFAULT_WORKDIR, rest_port=DEFAULT_VECTORDB_PORT, grpc_port=None
):
    """
    Start a Qdrant docker container if it is not already running.

    This function checks if a Qdrant docker container named 'qdrant' is already running. If so, it logs that
    the container is already up and does nothing. If the container exists but is stopped, it simply restarts
    the container. If the container does not exist, a new Qdrant container is created and started with specified
    REST and gRPC ports, and a volume for Qdrant storage.

    Args:
        c: The context parameter for Invoke tasks, automatically passed by Invoke.
        workdir (str): The working directory where Qdrant storage will be located. Defaults to 'workdir'.
        rest_port (int): The port number for REST API of Qdrant service. Defaults to DEFAULT_VECTORDB_PORT.
        grpc_port (int): The port number for gRPC service of Qdrant. Defaults to DEFAULT_VECTORDB_PORT+1.

    Note:
        This function requires Docker to be installed and running on the host system. It also requires the
        Docker SDK for Python (`docker` package) to be installed.

    Raises:
        Exception: If there is any error during the creation and start of the Qdrant container, an exception
        will be logged with the failure reason.
    """
    if not grpc_port:
        grpc_port = rest_port + 1

    client = docker.from_env()
    container_name = f"qdrant-v{QDRANT_VERSION}"
    is_running = check_if_docker_container_is_running(client, container_name)
    if is_running:
        logger.info(f"'{container_name}' docker container is already running.")
        return

    container = get_docker_container_by_name(client, container_name)
    if container:
        # container already exists, just stopped
        container.start()
        client.close()
        return

    # Get the current working directory and construct the volume path
    current_directory = os.getcwd()
    qdrant_index_path = os.path.join(current_directory, workdir, "qdrant_index")
    qdrant_snapshots_path = os.path.join(current_directory, workdir, "qdrant_snapshots")

    # --- Check ownership before proceeding ---
    if not check_and_warn_ownership(qdrant_index_path, QDRANT_USER_ID, QDRANT_GROUP_ID):
        logger.error("Correct ownership required for qdrant_index. Aborting.")
        client.close()
        return
    if not check_and_warn_ownership(
        qdrant_snapshots_path, QDRANT_USER_ID, QDRANT_GROUP_ID
    ):
        logger.error("Correct ownership required for qdrant_snapshots. Aborting.")
        client.close()
        return
    # --- End ownership check ---

    # Convert the volume specification into a format the Docker SDK expects
    volumes = {
        qdrant_index_path: {
            "bind": "/qdrant/storage",
            "mode": "rw",
        },
        qdrant_snapshots_path: {
            "bind": "/qdrant/snapshots",
            "mode": "rw",
        },
    }

    try:
        # Several steps are taken here to limit the attack surface of the Qdrant container
        # We are using an unprivileged container, and running the container in read-only mode. Also, we are using a non-root user.
        # See https://qdrant.tech/documentation/guides/security/#hardening for more details
        container = client.containers.run(
            f"qdrant/qdrant:v{QDRANT_VERSION}-unprivileged",
            detach=True,
            ports={"6333": rest_port, "6334": grpc_port},
            volumes=volumes,
            name=container_name,
            oom_kill_disable=True,
            read_only=True,
            user=f"{QDRANT_USER_ID}:{QDRANT_GROUP_ID}",  # Pass both UID and GID
        )
        logger.info(f"Container '{container_name}' started, id={container.id}")
        wait_for_docker_container_to_be_ready(
            container, string_in_output="Qdrant HTTP listening on"
        )
    except Exception as e:
        logger.error(f"Failed to start container '{container_name}': {str(e)}")
    finally:
        client.close()


@task
def start_embedding_docker_container(
    c,
    workdir=DEFAULT_WORKDIR,
    port=DEFAULT_EMBEDDING_MODEL_PORT,
    embedding_model=DEFAULT_EMBEDDING_MODEL_NAME,
    num_gpus=DEFAULT_NUM_GPUS,
):
    """
    Start a Docker container for HuggingFace's text embedding inference (TEI) if it is not already running.

    See https://github.com/huggingface/text-embeddings-inference for TEI documentation.
    This function checks if a text-embedding-inference Docker container is already running and starts it if not.


    Args:
        c: The context parameter for Invoke tasks, automatically passed by Invoke.
        workdir (str): The working directory where any required files are stored. This directory is mounted to the container, and is used to save the embedding model to disk.
        port (int): The port number on which the container's web server is exposed.
        embedding_model (str): The HuggingFace ID of the embedding model to be used for embedding text.
        num_gpus (int): The number of GPUs to use for the container. Defaults to DEFAULT_NUM_GPUS.

    Note:
        Requires Docker to be installed and running on the host system, and the Docker SDK for Python (`docker` package) to be installed.
        GPU support is subject to Docker's GPU access and configuration on the host system.
        If the system does not have GPU (or has an incompatible GPU), you need to modify this task to use the CPU-only docker of TEI.
    """
    client = docker.from_env()
    for gpu_id in range(num_gpus):
        tei_container_name = f"text-embedding-inference-v{TEI_VERSION}-gpu{gpu_id}"
        is_running = check_if_docker_container_is_running(client, tei_container_name)
        if is_running:
            logger.info(
                f"'text-embedding-inference-{gpu_id}' docker container is already running."
            )
            continue

        container = get_docker_container_by_name(client, tei_container_name)
        if container:
            # container already exists, just stopped
            container.start()
            continue

        # Get the current working directory and construct the volume path
        current_directory = os.getcwd()

        # Convert the volume specification into a format the Docker SDK expects
        volumes = {
            os.path.join(current_directory, workdir): {
                "bind": "/data",
                "mode": "rw",
            },
        }

        command = [
            "--model-id",
            embedding_model,
            "--max-client-batch-size",
            "128",
            "--max-batch-tokens",
            "50000",
            "--max-concurrent-requests",
            "256",
            "--dtype",
            (
                "float16"
                if TEI_DOCKER_CONTAINER_HARDWARE_ARCHITECTURE != "cpu"
                else "float32"
            ),
            "--tokenization-workers",
            "8",
            "--hostname",
            "0.0.0.0",
        ]
        if embedding_model == "Alibaba-NLP/gte-multilingual-base":
            command.append(
                "--revision=refs/pr/7"
            )  # this model's main branch is not compatible with TEI

        try:
            if TEI_DOCKER_CONTAINER_HARDWARE_ARCHITECTURE:
                image = f"ghcr.io/huggingface/text-embeddings-inference:{TEI_DOCKER_CONTAINER_HARDWARE_ARCHITECTURE}-{TEI_VERSION}"
            else:
                image = f"ghcr.io/huggingface/text-embeddings-inference:{TEI_VERSION}"

            container = client.containers.run(
                image,
                detach=True,
                device_requests=(
                    [
                        docker.types.DeviceRequest(
                            capabilities=[["gpu"]], device_ids=[str(gpu_id)]
                        )
                    ]
                    if TEI_DOCKER_CONTAINER_HARDWARE_ARCHITECTURE != "cpu"
                    else None
                ),
                ports={"80": port + gpu_id},
                volumes=volumes,
                name=tei_container_name,
                command=command,
                remove=True,  # Automatically remove the container when it stops
            )
            logger.info(
                f"text-embeddings-inference-{gpu_id} docker container started, id={container.id}"
            )

            wait_for_docker_container_to_be_ready(container, string_in_output="Ready")
        except Exception as e:
            logger.error(
                f"Failed to start text-embeddings-inference-{gpu_id} container: {str(e)}"
            )
    client.close()


@task
def load_distilled_model(
    c,
    distilled_model_port=DEFAULT_DISTILLED_MODEL_PORT,
    workdir=DEFAULT_WORKDIR,
    model_id="stanford-oval/Llama-2-7b-WikiChat-fused",
):
    """Load a distilled model using Docker"""
    client = docker.from_env()
    container_name = f"distilled-model-tgi-v{TGI_VERSION}"

    is_running = check_if_docker_container_is_running(client, container_name)
    if is_running:
        logger.info("Distilled model docker container is already running.")
        return

    container = get_docker_container_by_name(client, container_name)
    if container:
        # container already exists, just stopped
        container.start()
        logger.info("Distilled model docker container started.")
        return

    # Get the current working directory and construct the volume path
    current_directory = os.getcwd()
    # Convert the volume specification into a format the Docker SDK expects
    volumes = {
        os.path.join(current_directory, workdir): {
            "bind": "/data",
            "mode": "rw",
        },
    }

    try:
        container = client.containers.run(
            f"ghcr.io/huggingface/text-generation-inference:{TGI_VERSION}",
            detach=True,
            device_requests=[docker.types.DeviceRequest(capabilities=[["gpu"]])],
            ports={"80": distilled_model_port},
            volumes=volumes,
            name=container_name,
            command=[
                "--model-id",
                model_id,
                "--hostname",
                "0.0.0.0",
                "--num-shard",
                "1",
            ],
        )
        logger.info(f"Distilled model docker container started, id={container.id}")
        wait_for_docker_container_to_be_ready(container, string_in_output="Ready")
    except Exception as e:
        logger.error(f"Failed to start distilled model container: {str(e)}")
    finally:
        client.close()


@task
def stop_all_docker_containers(c):
    """
    Stops all Docker containers used in the project.

    This task stops the text embedding inference containers for each GPU,
    the Qdrant container and the distilled model container.

    Args:
        c: The context parameter for Invoke tasks, automatically passed by Invoke.
    """

    # Stop text embedding inference containers
    for gpu_id in range(DEFAULT_NUM_GPUS):
        container_name = f"text-embedding-inference-v{TEI_VERSION}-gpu{gpu_id}"
        stop_docker_container(c, container_name)

    # Stop Vector DB containers
    stop_docker_container(c, f"qdrant-v{QDRANT_VERSION}")

    # Stop distilled model container
    stop_docker_container(c, f"distilled-model-tgi-v{TGI_VERSION}")

    logger.info("All Docker containers have been stopped.")
