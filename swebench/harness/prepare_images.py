import docker
import resource
import time

from argparse import ArgumentParser

from swebench.harness.constants import KEY_INSTANCE_ID
from swebench.harness.docker_build import build_instance_images
from swebench.harness.docker_utils import list_images
from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.utils import load_swebench_dataset, str2bool


def filter_dataset_to_build(
    dataset: list,
    instance_ids: list | None,
    client: docker.DockerClient,
    force_rebuild: bool,
    namespace: str = None,
    tag: str = None,
):
    """
    Filter the dataset to only include instances that need to be built.

    Args:
        dataset (list): List of instances (usually all of SWE-bench dev/test split)
        instance_ids (list): List of instance IDs to build.
        client (docker.DockerClient): Docker client.
        force_rebuild (bool): Whether to force rebuild all images.
    """
    # Get existing images
    existing_images = list_images(client)
    data_to_build = []

    if instance_ids is None:
        instance_ids = [instance[KEY_INSTANCE_ID] for instance in dataset]

    # Check if all instance IDs are in the dataset
    not_in_dataset = set(instance_ids).difference(
        set([instance[KEY_INSTANCE_ID] for instance in dataset])
    )
    if not_in_dataset:
        raise ValueError(f"Instance IDs not found in dataset: {not_in_dataset}")

    for instance in dataset:
        if instance[KEY_INSTANCE_ID] not in instance_ids:
            # Skip instances not in the list
            continue

        # Check if the instance needs to be built (based on force_rebuild flag and existing images)
        spec = make_test_spec(instance, namespace=namespace, instance_image_tag=tag)
        if force_rebuild:
            data_to_build.append(instance)
        elif spec.instance_image_key not in existing_images:
            data_to_build.append(instance)

    return data_to_build


def build_instance_images_with_retry(
    client, dataset, force_rebuild, max_workers, max_retries=3, namespace=None, tag=None
):
    """
    Build Docker images for instances with retry mechanism.
    
    Args:
        client (docker.DockerClient): Docker client.
        dataset (list): List of instances to build.
        force_rebuild (bool): Whether to force rebuild all images.
        max_workers (int): Number of workers for parallel processing.
        max_retries (int): Maximum number of retries for failed builds.
        namespace (str): Namespace to use for the images.
        tag (str): Tag to use for the images.
        
    Returns:
        tuple: Lists of successful and failed builds.
    """
    all_successful = []
    remaining_dataset = dataset
    
    for attempt in range(1, max_retries + 1):
        if not remaining_dataset:
            break
            
        print(f"Build attempt {attempt}/{max_retries} for {len(remaining_dataset)} images")
        
        # Attempt to build the remaining images
        successful, failed = build_instance_images(
            client=client,
            dataset=remaining_dataset,
            force_rebuild=force_rebuild,
            max_workers=max_workers,
            namespace=namespace,
            tag=tag,
        )
        
        # Add successful builds to the list
        all_successful.extend(successful)
        
        # Prepare for next retry if needed
        if failed and attempt < max_retries:
            # Find the instances that failed
            failed_instance_ids = [build_result[0].instance_id for build_result in failed]
            print(f"Retrying {len(failed_instance_ids)} failed builds in 5 seconds...")
            time.sleep(5)  # Add a small delay before retrying
            
            # Filter the dataset to only include failed instances
            remaining_dataset = [
                instance for instance in remaining_dataset
                if make_test_spec(
                    instance, namespace=namespace, instance_image_tag=tag
                ).instance_id in failed_instance_ids
            ]
        else:
            # No more retries or no failures
            return all_successful, failed
    
    return all_successful, failed


def main(
    dataset_name,
    split,
    instance_ids,
    max_workers,
    force_rebuild,
    open_file_limit,
    namespace,
    tag,
    max_retries=3,
):
    """
    Build Docker images for the specified instances.

    Args:
        dataset_name (str): Name of the dataset to use.
        split (str): Split to use.
        instance_ids (list): List of instance IDs to build.
        max_workers (int): Number of workers for parallel processing.
        force_rebuild (bool): Whether to force rebuild all images.
        open_file_limit (int): Open file limit.
        namespace (str): Namespace to use for the images.
        tag (str): Tag to use for the images.
        max_retries (int): Maximum number of retries for failed builds.
    """
    # Set open file limit
    resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_limit, open_file_limit))
    client = docker.from_env()

    # Filter out instances that were not specified
    dataset = load_swebench_dataset(dataset_name, split)
    dataset = filter_dataset_to_build(
        dataset, instance_ids, client, force_rebuild, namespace, tag
    )

    # Build images for remaining instances with retry logic
    successful, failed = build_instance_images_with_retry(
        client=client,
        dataset=dataset,
        force_rebuild=force_rebuild,
        max_workers=max_workers,
        max_retries=max_retries,
        namespace=namespace,
        tag=tag,
    )
    print(f"Successfully built {len(successful)} images")
    print(f"Failed to build {len(failed)} images after {max_retries} attempts")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="princeton-nlp/SWE-bench_Lite",
        help="Name of the dataset to use",
    )
    parser.add_argument("--split", type=str, default="test", help="Split to use")
    parser.add_argument(
        "--instance_ids",
        nargs="+",
        type=str,
        help="Instance IDs to run (space separated)",
    )
    parser.add_argument(
        "--max_workers", type=int, default=4, help="Max workers for parallel processing"
    )
    parser.add_argument(
        "--force_rebuild", type=str2bool, default=False, help="Force rebuild images"
    )
    parser.add_argument(
        "--open_file_limit", type=int, default=8192, help="Open file limit"
    )
    parser.add_argument(
        "--namespace", type=str, default=None, help="Namespace to use for the images"
    )
    parser.add_argument(
        "--tag", type=str, default=None, help="Tag to use for the images"
    )
    parser.add_argument(
        "--max_retries", type=int, default=3, help="Maximum number of retries for failed builds"
    )
    args = parser.parse_args()
    main(**vars(args))
