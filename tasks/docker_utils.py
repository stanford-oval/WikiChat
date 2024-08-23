import os
from time import sleep

import docker
from invoke import task
import sys

sys.path.insert(0, "./")
from pipelines.utils import get_logger
from tasks.defaults import (
    DEFAULT_DISTILLED_MODEL_PORT,
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_EMBEDDING_MODEL_PORT,
    DEFAULT_NUM_GPUS,
    DEFAULT_WORKDIR,
    QDRANT_VERSION,
    TEI_VERSION,
    TGI_VERSION,
)

logger = get_logger(__name__)


def check_if_docker_container_is_running(docker_client, container_name: str):
    # List all running containers
    running_containers = docker_client.containers.list()

    for container in running_containers:
        # Check if the specified container name matches any running container's name
        # Container names are stored in a list
        if container_name in container.name:
            return True

    # If no running container matched the specified name
    return False


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


def get_docker_container_by_name(container_name: str):
    client = docker.from_env()
    all_containers = client.containers.list(all=True)
    for container in all_containers:
        if container_name == container.name:
            client.close()
            return container

    client.close()
    return None


def wait_for_docker_container_to_be_ready(
    docker_client, container, string_in_output: str = "", timeout=60
):
    """
    Waits for the specified Docker container to be ready.

    This function waits until the Docker container is running and the specified string appears in the container's logs.

    Args:
        docker_client: The Docker client instance.
        container: The Docker container instance.
        string_in_output (str): The string to look for in the container's output to consider it ready.
        timeout (int): Timeout in seconds

    Raises:
        RuntimeError: If the container is not ready within the timeout period.
    """
    timeout = 60
    step_time = timeout // 10
    elapsed_time = 0
    logger.info("Waiting for the container '%s' to be ready...", container)

    def is_ready():
        container_status = docker_client.containers.get(container.id).status
        container_logs = container.logs().decode("utf-8")
        return container_status == "running" and string_in_output in container_logs

    last_log_length = 0

    while not is_ready() and elapsed_time < timeout:
        sleep(step_time)
        container_logs = container.logs().decode("utf-8")
        new_logs = container_logs[last_log_length:]
        if new_logs:
            logger.info(new_logs)
        last_log_length = len(container_logs)
        elapsed_time += step_time

    if not is_ready():
        logger.error(
            "Docker container still not running or string not found in logs after %d seconds.",
            timeout,
        )
        raise RuntimeError(
            f"Docker container still not running or string '{string_in_output}' not found in logs after {timeout} seconds."
        )


@task
def start_qdrant_docker_container(
    c, workdir=DEFAULT_WORKDIR, rest_port=6333, grpc_port=6334
):
    """
    Starts a Qdrant docker container if it is not already running.

    This function checks if a Qdrant docker container named 'qdrant' is already running. If so, it logs that
    the container is already up and does nothing. If the container exists but is stopped, it simply restarts
    the container. If the container does not exist, a new Qdrant container is created and started with specified
    REST and gRPC ports, and a volume for Qdrant storage.

    Args:
        c: The context parameter for Invoke tasks, automatically passed by Invoke.
        workdir (str): The working directory where Qdrant storage will be located. Defaults to 'workdir'.
        rest_port (int): The port number for REST API of Qdrant service. Defaults to 6333.
        grpc_port (int): The port number for gRPC service of Qdrant. Defaults to 6334.

    Note:
        This function requires Docker to be installed and running on the host system. It also requires the
        Docker SDK for Python (`docker` package) to be installed.

    Raises:
        Exception: If there is any error during the creation and start of the Qdrant container, an exception
        will be logged with the failure reason.
    """
    client = docker.from_env()
    container_name = f"qdrant-v{QDRANT_VERSION}"
    is_running = check_if_docker_container_is_running(client, container_name)
    if is_running:
        logger.info("%s docker container is already running.", container_name)
        return

    container = get_docker_container_by_name(container_name)
    if container:
        # container already exists, just stopped
        container.start()
        client.close()
        return

    # Get the current working directory and construct the volume path
    current_directory = os.getcwd()

    # Convert the volume specification into a format the Docker SDK expects
    volumes = {
        os.path.join(current_directory, workdir, "qdrant_index"): {
            "bind": "/qdrant/storage",
            "mode": "rw",
        },
    }

    try:
        container = client.containers.run(
            f"qdrant/qdrant:v{QDRANT_VERSION}",
            detach=True,
            ports={"6333": rest_port, "6334": grpc_port},
            volumes=volumes,
            name=container_name,
            oom_kill_disable=True,
        )
        logger.info("%s container started, id=%s", container_name, container.id)
        wait_for_docker_container_to_be_ready(
            client, container, string_in_output="Qdrant HTTP listening on", timeout=120
        )
    except Exception as e:
        logger.error("Failed to start %s container: %s", container_name, str(e))
    client.close()


@task
def start_embedding_docker_container(
    c,
    workdir=DEFAULT_WORKDIR,
    port=DEFAULT_EMBEDDING_MODEL_PORT,
    embedding_model=DEFAULT_EMBEDDING_MODEL_NAME,
):
    """
    Starts a Docker container for HuggingFace's text embedding inference (TEI) if it is not already running.

    See https://github.com/huggingface/text-embeddings-inference for TEI documentation.
    This function checks if a text-embedding-inference Docker container is already running and starts it if not.


    Args:
        c: The context parameter for Invoke tasks, automatically passed by Invoke.
        workdir (str): The working directory where any required files are stored. This directory is mounted to the container, and is used to save the embedding model to disk.
        port (int): The port number on which the container's web server is exposed.
        embedding_model (str): The HuggingFace ID of the embedding model to be used for embedding text.

    Note:
        Requires Docker to be installed and running on the host system, and the Docker SDK for Python (`docker` package) to be installed.
        GPU support is subject to Docker's GPU access and configuration on the host system.
        If the system does not have GPU (or has an incompatible GPU), you need to modify this task to use the CPU-only docker of TEI.
    """
    client = docker.from_env()
    for gpu_id in range(DEFAULT_NUM_GPUS):
        tei_container_name = f"text-embedding-inference-v{TEI_VERSION}-gpu{gpu_id}"
        is_running = check_if_docker_container_is_running(client, tei_container_name)
        if is_running:
            logger.info(
                "text-embedding-inference-%d docker container is already running.",
                gpu_id,
            )
            continue

        container = get_docker_container_by_name(tei_container_name)
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

        try:
            container = client.containers.run(
                f"ghcr.io/huggingface/text-embeddings-inference:{TEI_VERSION}",
                detach=True,
                device_requests=[
                    docker.types.DeviceRequest(
                        capabilities=[["gpu"]], device_ids=[str(gpu_id)]
                    )
                ],
                ports={"80": port + gpu_id},
                volumes=volumes,
                name=tei_container_name,
                command=[
                    "--model-id",
                    embedding_model,
                    "--max-client-batch-size",
                    "4096",
                    "--max-batch-tokens",
                    "50000",
                    "--max-concurrent-requests",
                    "128",
                    "--hostname",
                    "0.0.0.0",
                ],
            )
            logger.info(
                "text-embeddings-inference-%d docker container started, id=%s",
                gpu_id,
                container.id,
            )

            wait_for_docker_container_to_be_ready(
                client, container, string_in_output="Ready"
            )
        except Exception as e:
            logger.error(
                "Failed to start text-embeddings-inference-%d container: %s",
                gpu_id,
                str(e),
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

    container = get_docker_container_by_name(container_name)
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
        logger.info("Distilled model docker container started, id=%s", container.id)
        wait_for_docker_container_to_be_ready(
            client, container, string_in_output="Ready"
        )
    except Exception as e:
        logger.error("Failed to start distilled model container: %s", str(e))
    finally:
        client.close()


@task
def stop_all_docker_containers(c):
    """
    Stops all Docker containers used in the project.

    This task stops the text embedding inference containers for each GPU,
    the Qdrant container, and the distilled model container.

    Args:
        c: The context parameter for Invoke tasks, automatically passed by Invoke.
    """

    # Stop text embedding inference containers
    for gpu_id in range(DEFAULT_NUM_GPUS):
        container_name = f"text-embedding-inference-v{TEI_VERSION}-gpu{gpu_id}"
        stop_docker_container(c, container_name)

    # Stop Qdrant container
    stop_docker_container(c, f"qdrant-v{QDRANT_VERSION}")

    # Stop distilled model container
    stop_docker_container(c, f"distilled-model-tgi-v{TGI_VERSION}")

    logger.info("All Docker containers have been stopped.")
