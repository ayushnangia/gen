import sys
import argparse
import subprocess


def _forward_to_main(callable_main, passthrough_args):
    saved_argv = list(sys.argv)
    try:
        sys.argv = [saved_argv[0]] + passthrough_args
        callable_main()
    finally:
        sys.argv = saved_argv


def main():
    parser = argparse.ArgumentParser(
        prog="python -m synwoz",
        description="SynWOZ toolkit dispatcher (forwards to existing scripts without modification)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Generation subcommands
    gen_serial = subparsers.add_parser("gen-serial", help="Run serial dialogue generation")
    gen_serial.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to serial_gen.py")

    gen_parallel = subparsers.add_parser("gen-parallel", help="Run parallel/async dialogue generation")
    gen_parallel.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to parallel_gen.py")

    gen_perf = subparsers.add_parser("gen-perf", help="Run performance-focused generation")
    gen_perf.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to performance_parallel.py")

    # Moderation
    moderate = subparsers.add_parser("moderate", help="Run moderation over JSONL dialogues")
    moderate.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to omni.py")

    # Post-processing
    post_embed = subparsers.add_parser("post-embed-dedup", help="FAISS-based embedding deduplication")
    post_embed.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to embedding_faiss.py")

    post_filter = subparsers.add_parser("post-filter", help="Filter dialogues by num_lines threshold")
    post_filter.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to filter_num_line_gt_5.py")

    post_hf = subparsers.add_parser("post-hf-upload", help="Prepare and push dataset to Hugging Face Hub")
    post_hf.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to hugging_face_upload.py")

    # Dashboard
    dash = subparsers.add_parser("dashboard", help="Launch Streamlit dashboard")
    dash.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to Streamlit run")

    # Pipeline runner
    pipeline = subparsers.add_parser("pipeline", help="Run generation, moderation, and dedup stages in sequence")
    pipeline.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for pipeline artifacts (defaults to artifacts/run-<timestamp>)",
    )
    pipeline.add_argument(
        "--gen-cmd",
        type=str,
        default="",
        help="Extra CLI args forwarded to gen-parallel (e.g. \"--total_generations 10\")",
    )
    pipeline.add_argument(
        "--moderation-cmd",
        type=str,
        default="",
        help="Extra CLI args forwarded to moderation",
    )
    pipeline.add_argument(
        "--dedupe-cmd",
        type=str,
        default="",
        help="Extra CLI args forwarded to post-embed-dedup",
    )
    pipeline.add_argument(
        "--skip-moderation",
        action="store_true",
        help="Skip the moderation stage",
    )
    pipeline.add_argument(
        "--skip-dedupe",
        action="store_true",
        help="Skip the deduplication stage",
    )

    # Resources
    resources = subparsers.add_parser("resources", help="Manage local models and datasets")
    resources.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to resources sync")

    parsed = parser.parse_args()

    if parsed.command == "gen-serial":
        from synwoz.generation.serial_gen import main as _serial_main
        _forward_to_main(_serial_main, parsed.args)
    elif parsed.command == "gen-parallel":
        from synwoz.generation.parallel_gen import main as _parallel_main
        _forward_to_main(_parallel_main, parsed.args)
    elif parsed.command == "gen-perf":
        from synwoz.generation.performance_parallel import main as _perf_main
        _forward_to_main(_perf_main, parsed.args)
    elif parsed.command == "moderate":
        from synwoz.moderation.omni import main as _omni_main
        _forward_to_main(_omni_main, parsed.args)
    elif parsed.command == "post-embed-dedup":
        from synwoz.postprocessing.embedding_faiss import main as _embed_main
        _forward_to_main(_embed_main, parsed.args)
    elif parsed.command == "post-filter":
        from synwoz.postprocessing.filter_num_line_gt_5 import main as _filter_main
        _forward_to_main(_filter_main, parsed.args)
    elif parsed.command == "post-hf-upload":
        from synwoz.postprocessing.hugging_face_upload import main as _hf_main
        _forward_to_main(_hf_main, parsed.args)
    elif parsed.command == "dashboard":
        script_path = __import__("pathlib").Path(__file__).parent / "dashboard" / "dashboard.py"
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(script_path)] + (parsed.args or []), check=True)
    elif parsed.command == "pipeline":
        from synwoz.pipeline import build_pipeline_config, run_pipeline

        config = build_pipeline_config(
            parsed.output_dir,
            parsed.gen_cmd,
            parsed.moderation_cmd,
            parsed.dedupe_cmd,
            parsed.skip_moderation,
            parsed.skip_dedupe,
        )
        run_pipeline(config)
    elif parsed.command == "resources":
        from synwoz.resources import main as _resources_main

        _forward_to_main(_resources_main, parsed.args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
