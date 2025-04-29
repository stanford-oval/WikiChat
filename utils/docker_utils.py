import re
import time

import docker
from docker.errors import NotFound, APIError
from utils.logging import logger
from typing import Optional
from docker.models.containers import Container


def check_if_docker_container_is_running(
    docker_client: docker.DockerClient, container_name: str
) -> bool:
    """
    Checks if a Docker container with the exact name is currently running.

    Args:
        docker_client: An initialized Docker client instance.
        container_name: The exact name of the container to check.

    Returns:
        True if a container with the exact name is running, False otherwise.
        Returns False on API or unexpected errors during the check.
    """
    try:
        # Use filters for efficient lookup of *running* containers by exact name.
        # The regex anchors ^ and $ ensure exact match.
        # list() returns an empty list if no match, does not raise NotFound.
        running_containers = docker_client.containers.list(
            filters={"name": f"^{container_name}$", "status": "running"}
        )

        if running_containers:
            # This case should be rare with exact name matching but log if it occurs.
            if len(running_containers) > 1:
                logger.warning(
                    f"Multiple *running* containers found matching the exact name '{container_name}'. "
                    f"This is unusual. Returning True. IDs: {[c.id for c in running_containers]}"
                )
            else:
                logger.debug(
                    f"Found running container '{container_name}' (ID: {running_containers[0].id})."
                )
            return True
        else:
            logger.debug(
                f"No *running* container found with the exact name '{container_name}'."
            )
            return False

    except APIError as e:
        logger.error(
            f"API error checking running status for container '{container_name}': {e}"
        )
        return False
    except Exception as e:
        # Log other potential errors during listing
        logger.error(
            f"Unexpected error checking running status for container '{container_name}': {e}"
        )
        return False


def get_docker_container_by_name(
    docker_client: docker.DockerClient, container_name: str
) -> Optional[Container]:
    """
    Retrieves a Docker container (running or stopped) by its exact name.

    Args:
        docker_client: An initialized Docker client instance.
        container_name: The exact name of the container to retrieve.

    Returns:
        The container object if found, otherwise None. Returns None on API or
        unexpected errors during lookup.

    Note:
        While Docker typically prevents having multiple *running* containers
        with the exact same name, it's theoretically possible (though rare,
        perhaps involving stopped containers or edge cases) for the API
        to return multiple containers matching an exact name filter. This
        function handles that possibility by logging a warning and returning
        the first container found.
    """
    try:
        # Use filters for efficient lookup by exact name across all states.
        # The regex anchors ^ and $ ensure exact match.
        containers = docker_client.containers.list(
            all=True, filters={"name": f"^{container_name}$"}
        )

        if not containers:
            logger.debug(f"No container found with the exact name '{container_name}'.")
            return None

        if len(containers) > 1:
            # This case should be rare with exact name matching but log if it occurs.
            logger.warning(
                f"Multiple containers found matching the exact name '{container_name}'. "
                f"This is unusual. Returning the first one found (ID: {containers[0].id}). "
                f"Container IDs found: {[c.id for c in containers]}"
            )

        return containers[0]

    except APIError as e:
        logger.error(f"API error while retrieving container '{container_name}': {e}")
        return None
    except Exception as e:
        # Catch other potential errors during the listing process.
        logger.error(f"Unexpected error retrieving container '{container_name}': {e}")
        return None


def strip_ansi_codes(log_text: str) -> str:
    """Removes ANSI escape codes from a string."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", log_text)


def wait_for_docker_container_to_be_ready(
    container, string_in_output: str = "", timeout: int = 600
):
    """
    Waits for the specified Docker container to be ready.

    Polls the container status and logs until the container is 'running'
    and, if `string_in_output` is provided, that string appears in the logs.

    Args:
        container: The Docker container instance.
        string_in_output (str): The string to look for in the container's output
                                to consider it ready. If empty, only status is checked.
        timeout (int): Maximum time to wait in seconds.

    Raises:
        RuntimeError: If the container does not become ready within the timeout,
                      exits unexpectedly, or is not found.
    """
    sleep_interval = 2  # Check every 2 seconds
    start_time = time.monotonic()
    logger.info(
        f"Waiting for container '{container.name}' to be ready "
        f"(timeout: {timeout}s)..."
    )

    last_log_length = 0
    container_id = container.id  # Store ID for logging in case of errors

    while time.monotonic() - start_time < timeout:
        try:
            # Reload the container's state from the server
            container.reload()
            status = container.status

            if status == "running":
                logs = ""
                try:
                    # Fetch logs only when running
                    # Use errors='replace' for robustness against decoding errors
                    logs = strip_ansi_codes(
                        container.logs().decode("utf-8", errors="replace")
                    )
                    new_logs = logs[last_log_length:]
                    if new_logs.strip():  # Log only if there's actual new content
                        logger.info(
                            f"Container '{container.name}' new logs:\n{new_logs.strip()}"
                        )
                    last_log_length = len(logs)
                except APIError as log_err:
                    logger.warning(
                        f"API error fetching logs for running container '{container.name}': {log_err}. Continuing wait."
                    )
                except Exception as log_err:
                    logger.warning(
                        f"Unexpected error fetching logs for running container '{container.name}': {log_err}. Continuing wait."
                    )

                # Check readiness condition: running and (no string needed OR string found)
                if not string_in_output or string_in_output in logs:
                    logger.info(f"Container '{container.name}' is ready.")
                    return  # Success: Container is ready

            elif status in ["exited", "dead"]:
                logger.error(
                    f"Container '{container.name}' exited unexpectedly with status '{status}'."
                )
                final_logs = ""
                try:
                    final_logs = strip_ansi_codes(
                        container.logs().decode("utf-8", errors="replace")
                    )
                    logger.error(
                        f"Final logs for '{container.name}':\n{final_logs.strip()}"
                    )
                except Exception as log_err:
                    logger.error(
                        f"Could not fetch final logs for exited container '{container.name}': {log_err}"
                    )
                raise RuntimeError(
                    f"Container '{container.name}' exited unexpectedly with status '{status}'."
                )
            # else: status is 'created', 'restarting', 'paused', 'removing' -> continue waiting loop

        except NotFound:
            logger.error(
                f"Container '{container.name}' (ID: {container_id}) not found."
            )
            raise RuntimeError(
                f"Container '{container.name}' (ID: {container_id}) not found during wait."
            )
        except APIError as api_err:
            # Catch Docker API errors during reload/status check
            logger.warning(
                f"API error checking container '{container.name}': {api_err}. Retrying..."
            )
        except Exception as e:
            # Catch other potential errors
            logger.warning(
                f"Unexpected error checking container '{container.name}': {e}. Retrying..."
            )

        # Wait before the next check
        time_left = timeout - (time.monotonic() - start_time)
        time.sleep(
            min(sleep_interval, max(0, time_left))
        )  # Avoid sleeping longer than remaining time

    # Loop finished: Timeout reached
    final_status = "unknown"
    final_logs = ""
    try:
        # Perform one final check after the loop exits
        container.reload()
        final_status = container.status
        if final_status == "running":
            final_logs = strip_ansi_codes(
                container.logs().decode("utf-8", errors="replace")
            )
            if not string_in_output or string_in_output in final_logs:
                logger.info(
                    f"Container '{container.name}' became ready just at the timeout limit."
                )
                return  # Success right at the end

    except Exception as e:
        logger.error(f"Error during final check for container '{container.name}': {e}")
        # Fall through to raise timeout error, including this information if possible

    # If we reach here, it timed out without becoming ready
    log_msg = (
        f"Timeout waiting for container '{container.name}' after {timeout} seconds. "
        f"Final Status: '{final_status}'. "
    )
    if string_in_output:
        found_string = string_in_output in final_logs
        log_msg += f"String '{string_in_output}' {'found' if found_string else 'not found'} in final logs."

    logger.error(log_msg)
    # Log any final logs obtained during the last check, if they haven't been logged yet
    if final_logs and len(final_logs) > last_log_length:
        logger.error(f"Final logs checked:\n{final_logs[last_log_length:].strip()}")

    raise RuntimeError(log_msg)
