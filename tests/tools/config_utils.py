from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import yaml


def modify_stage_config(
    yaml_path: str,
    updates: dict[str, Any] = None,
    deletes: dict[str, Any] = None,
) -> str:
    """
    Modify configurations in a YAML file, supporting both top-level and stage-specific modifications,
    including addition, modification, and deletion of configurations.

    Args:
        yaml_path: Path to the YAML configuration file.
        updates: Dictionary containing both top-level and stage-specific modifications to add or update.
                Format: {
                    'async_chunk': True,
                    'stage_args': {
                        0: {'engine_args.max_model_len': 5800},
                        1: {'engine_args.max_num_seqs': 2}
                    }
                }
        deletes: Dictionary containing configurations to delete.
                Format: {
                    'old_config': None,  # Delete entire key
                    'stage_args': {
                        0: ['engine_args.old_param'],
                        1: ['runtime.unused_setting']
                    }
                }

    Returns:
        str: Path to the newly created modified YAML file with timestamp suffix.
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"yaml does not exist: {path}")

    try:
        with open(yaml_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception as e:
        raise ValueError(f"Cannot parse YAML file: {e}")

    # Helper function to apply update
    def apply_update(config_dict: dict, key_path: str, value: Any) -> None:
        """Apply update to dictionary using dot-separated path."""
        # Handle direct list assignment (e.g., engine_input_source: [1, 2])
        if "." not in key_path:
            # Simple key, set directly
            config_dict[key_path] = value
            return

        current = config_dict
        keys = key_path.split(".")

        for i in range(len(keys) - 1):
            key = keys[i]

            # Handle list indices
            if key.isdigit() and isinstance(current, list):
                index = int(key)
                if index < 0:
                    raise ValueError(f"Negative list index not allowed: {index}")
                if index >= len(current):
                    # Expand list if needed
                    while len(current) <= index:
                        # If we need to go deeper (more keys after this), create a dict
                        # Otherwise, create None placeholder
                        current.append({} if i < len(keys) - 2 else None)
                current = current[index]
            elif isinstance(current, dict):
                # Handle dictionary keys
                if key not in current:
                    # If there are more keys after this, create appropriate structure
                    if i < len(keys) - 1:
                        # Check if next key is a digit (list index) or string (dict key)
                        if keys[i + 1].isdigit():
                            current[key] = []
                        else:
                            current[key] = {}
                    else:
                        # This is the last key, create based on value type
                        current[key] = [] if isinstance(value, list) else {}
                elif not isinstance(current[key], (dict, list)) and i < len(keys) - 1:
                    # If current value is not dict/list but we need to go deeper, replace it
                    if keys[i + 1].isdigit():
                        current[key] = []
                    else:
                        current[key] = {}
                current = current[key]
            else:
                # Current is not a dict or list, cannot traverse further
                raise TypeError(
                    f"Cannot access {'.'.join(keys[: i + 1])} as a dict/list. It's a {type(current).__name__}"
                )

        # Set the final value
        last_key = keys[-1]
        if isinstance(current, list) and last_key.isdigit():
            # Setting a value in a list by index
            index = int(last_key)
            if index < 0:
                raise ValueError(f"Negative list index not allowed: {index}")
            if index >= len(current):
                # Expand list if needed
                while len(current) <= index:
                    current.append(None)
            current[index] = value
        elif isinstance(current, dict):
            # Special case: if the value is a list and we're setting a top-level key
            # Example: updating engine_input_source with [1, 2]
            current[last_key] = value
        else:
            # Current is not a dict, cannot set key
            raise TypeError(f"Cannot set value at {key_path}. Current type is {type(current).__name__}, expected dict.")

    # Helper function to delete by path
    def delete_by_path(config_dict: dict, path: str) -> None:
        """Delete configuration by dot-separated path."""
        if not path:
            return

        current = config_dict
        keys = path.split(".")

        # Traverse to the parent
        for i in range(len(keys) - 1):
            key = keys[i]

            # Handle list indices
            if key.isdigit() and isinstance(current, list):
                index = int(key)
                if index < 0 or index >= len(current):
                    raise KeyError(f"List index {index} out of bounds")
                current = current[index]
            elif isinstance(current, dict):
                if key not in current:
                    raise KeyError(f"Path {'.'.join(keys[: i + 1])} does not exist")
                current = current[key]
            else:
                raise TypeError(
                    f"Cannot access {'.'.join(keys[: i + 1])} as a dict/list. It's a {type(current).__name__}"
                )

        # Delete the item
        last_key = keys[-1]

        if isinstance(current, list) and last_key.isdigit():
            index = int(last_key)
            if index < 0 or index >= len(current):
                raise KeyError(f"List index {index} out of bounds")
            del current[index]
        elif isinstance(current, dict) and last_key in current:
            del current[last_key]
        else:
            print(f"Path {path} does not exist")

    # Apply deletions first
    if deletes:
        for key, value in deletes.items():
            if key == "stage_args":
                if value and isinstance(value, dict):
                    stage_args = config.get("stage_args", [])
                    if not stage_args:
                        raise ValueError("stage_args does not exist in config")

                    for stage_id, delete_paths in value.items():
                        if not delete_paths:
                            continue

                        # Find stage by ID
                        target_stage = None
                        for stage in stage_args:
                            if stage.get("stage_id") == int(stage_id):
                                target_stage = stage
                                break

                        if target_stage is None:
                            continue

                        # Delete specified paths in this stage
                        for path in delete_paths:
                            if path:  # Skip empty paths
                                delete_by_path(target_stage, path)
            elif "." in key:
                # Delete using dot-separated path
                delete_by_path(config, key)
            elif value is None and key in config:
                # Delete entire key
                del config[key]

    # Apply updates
    if updates:
        for key, value in updates.items():
            if key == "stage_args":
                if value and isinstance(value, dict):
                    stage_args = config.get("stage_args", [])
                    if not stage_args:
                        raise ValueError("stage_args does not exist in config")

                    for stage_id, stage_updates in value.items():
                        # Find stage by ID
                        target_stage = None
                        for stage in stage_args:
                            if stage.get("stage_id") == int(stage_id):
                                target_stage = stage
                                break

                        if target_stage is None:
                            available_ids = [s.get("stage_id") for s in stage_args if "stage_id" in s]
                            raise KeyError(f"Stage ID {stage_id} not found, available: {available_ids}")

                        # Apply updates to this stage
                        for path, val in stage_updates.items():
                            # Check if this is a simple key (not dot-separated)
                            # Example: 'engine_input_source' vs 'engine_args.max_model_len'
                            if "." not in path:
                                # Direct key assignment (e.g., updating a list value)
                                target_stage[path] = val
                            else:
                                # Dot-separated path (e.g., nested dict access)
                                apply_update(target_stage, path, val)
            elif "." in key:
                # Apply using dot-separated path
                apply_update(config, key, value)
            else:
                # Direct top-level key
                config[key] = value

    # Unique suffix: multiple modify_stage_config calls in one process often run
    # within the same second (e.g. test_qwen3_omni_expansion imports both
    # get_chunk_config and get_batch_token_config). int(time.time()) would collide
    # and the later write would overwrite the earlier YAML on disk.
    base_name = yaml_path.rsplit(".", 1)[0] if "." in yaml_path else yaml_path
    output_path = f"{base_name}_{time.time_ns()}.yaml"

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=None, sort_keys=False, allow_unicode=True, indent=2)

    return output_path
