"""Legacy entry point for resource downloads.

Prefer ``uv run python -m synwoz resources --`` instead of invoking this file directly.
"""

from synwoz.resources.setup_helpers import (
    parse_arguments,
    setup_environment,
)


if __name__ == "__main__":
    args = parse_arguments()
    setup_environment(
        model_name=args.model_name,
        model_path=args.model_path,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        persona_dataset_name=args.persona_dataset_name,
        persona_dataset_path=args.persona_dataset_path,
        force=args.force,
    )
