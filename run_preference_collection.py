#!/usr/bin/env python3
"""
CLI script for collecting human preference data using Prolific AI Task Builder.

This script replicates the functionality of the prolific_human_preferences_rlhf.ipynb notebook
and can be run from the command line.
"""

import sys
import argparse
import yaml
import json
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from prolific_ai_taskers import (
    ResponseGenerator,
    ProlificClient,
    load_prompts,
    create_response_pairs,
    process_preferences
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect human preference data using Prolific AI Task Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run the full pipeline
  python run_preference_collection.py

  # Use custom config and prompts
  python run_preference_collection.py --config my_config.yaml --prompts my_prompts.jsonl

  # Skip LLM generation and only run Prolific workflow
  python run_preference_collection.py --skip-generation --pairs outputs/response_pairs.csv

  # Only generate responses, don't publish to Prolific
  python run_preference_collection.py --generate-only

  # Use existing study and batch IDs to download results
  python run_preference_collection.py --skip-generation --skip-publish --study-id YOUR_STUDY_ID --batch-id YOUR_BATCH_ID
        """
    )

    # Input files
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('examples/config.yaml'),
        help='Path to config YAML file (default: examples/config.yaml)'
    )
    parser.add_argument(
        '--prompts',
        type=Path,
        default=Path('examples/prompts.jsonl'),
        help='Path to prompts JSONL file (default: examples/prompts.jsonl)'
    )
    parser.add_argument(
        '--env-file',
        type=Path,
        default=Path('.env'),
        help='Path to .env file with API tokens (default: .env)'
    )

    # Output directory
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('outputs'),
        help='Output directory for results (default: outputs)'
    )

    # Workflow control
    parser.add_argument(
        '--generate-only',
        action='store_true',
        help='Only generate LLM responses, skip Prolific workflow'
    )
    parser.add_argument(
        '--skip-generation',
        action='store_true',
        help='Skip LLM generation, use existing response pairs'
    )
    parser.add_argument(
        '--pairs',
        type=Path,
        help='Path to existing response_pairs.csv (required if --skip-generation)'
    )
    parser.add_argument(
        '--skip-publish',
        action='store_true',
        help='Skip study publishing (for testing batch creation)'
    )

    # Study/Batch IDs for resuming
    parser.add_argument(
        '--study-id',
        type=str,
        help='Existing study ID to resume from'
    )
    parser.add_argument(
        '--batch-id',
        type=str,
        help='Existing batch ID to resume from'
    )

    # Timing
    parser.add_argument(
        '--timeout',
        type=int,
        default=21600,
        help='Timeout in seconds for study completion (default: 21600 = 6 hours)'
    )

    parser.add_argument(
        '--no-demographics',
        action='store_true',
        help='Skip downloading participant demographics'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print verbose output'
    )

    return parser.parse_args()


def generate_responses(cfg, prompts_path, output_dir, verbose=False):
    """Generate LLM responses for prompts."""
    print("\n" + "="*80)
    print("PART 1: GENERATE LLM RESPONSES")
    print("="*80)

    # Load prompts
    print(f"\nLoading prompts from {prompts_path}...")
    prompts = load_prompts(prompts_path)
    print(f"Loaded {len(prompts)} prompts")

    if verbose:
        print("\nExample prompts:")
        for i, p in enumerate(prompts[:3], 1):
            print(f"{i}. {p['prompt']}")

    # Initialize generator
    print(f"\nInitializing model: {cfg['model']}...")
    generator = ResponseGenerator(model_name=cfg['model'])

    # Generate completions
    print("\nGenerating completions...")
    print(f"  Temperatures: {cfg['temperatures']}")
    print(f"  Completions per prompt: {cfg['num_completions_per_prompt']}")
    print(f"  Max new tokens: {cfg['max_new_tokens']}")

    df_completions = generator.generate_completions(
        prompts=prompts,
        temperatures=cfg['temperatures'],
        num_completions_per_prompt=cfg['num_completions_per_prompt'],
        max_new_tokens=cfg['max_new_tokens'],
        top_p=cfg['top_p'],
        verbose=verbose
    )

    print(f"\nGenerated {len(df_completions)} total completions")

    # Save completions
    csv_path, metadata_path = generator.save_completions(
        df=df_completions,
        output_dir=output_dir,
        config=cfg,
        num_prompts=len(prompts)
    )

    # Create response pairs
    print("\nCreating response pairs...")
    pairs_df = create_response_pairs(
        completions_df=df_completions,
        remove_prompt_prefix=True
    )
    print(f"Created {len(pairs_df)} response pairs")

    # Save pairs
    pairs_csv_path = output_dir / 'response_pairs.csv'
    pairs_df.to_csv(pairs_csv_path, index=False)
    print(f"Saved response pairs to {pairs_csv_path}")

    return pairs_csv_path, len(prompts)


def run_prolific_workflow(cfg, pairs_csv_path, output_dir, skip_publish=False,
                         study_id=None, batch_id=None, timeout=21600,
                         fetch_demographics=True, verbose=False):
    """Run the Prolific workflow for collecting human preferences."""
    print("\n" + "="*80)
    print("PART 2: COLLECT HUMAN PREFERENCES VIA PROLIFIC")
    print("="*80)

    # Initialize client
    print("\nInitializing Prolific client...")
    client = ProlificClient()

    # If resuming from existing study/batch
    if study_id and batch_id:
        print(f"\nResuming from existing study/batch:")
        print(f"  Study ID: {study_id}")
        print(f"  Batch ID: {batch_id}")
    else:
        # Create dataset and upload
        print("\nCreating dataset...")
        dataset_name = cfg['prolific']['batch_name']
        dataset_id = client.create_dataset(name=dataset_name)

        print("Uploading response pairs...")
        client.upload_dataset_file(dataset_id, pairs_csv_path)

        print("Waiting for dataset to be ready...")
        if not client.wait_for_dataset_ready(dataset_id):
            raise Exception("Dataset processing failed. Check your CSV format.")

        # Define task schema
        print("\nDefining task schema...")
        task_details = {
            'task_name': cfg['prolific']['task_schema']['task_name'],
            'task_introduction': cfg['prolific']['task_schema']['task_introduction'].replace('\n', ' '),
            'task_steps': cfg['prolific']['task_schema']['task_steps'].replace('\n', ' '),
            'inputs': [
                {'key': 'prompt', 'label': 'Prompt'},
                {'key': 'response_a', 'label': 'Response A'},
                {'key': 'response_b', 'label': 'Response B'},
            ],
            'judgment': {
                'type': 'single_choice',
                'options': [
                    {'value': 'A', 'label': 'Choose Response A'},
                    {'value': 'B', 'label': 'Choose Response B'},
                ]
            },
            'randomize_inputs': ['Response A', 'Response B'],
            'validation': {'require_choice': True}
        }

        # Create batch
        print("Creating batch...")
        batch_id = client.create_batch(
            name=cfg['prolific']['batch_name'],
            dataset_id=dataset_id,
            task_details=task_details
        )

        # Add instructions
        print("Adding instructions...")
        instructions = [{
            'type': 'multiple_choice',
            'created_by': client.researcher_name,
            'description': cfg['prolific']['task_schema']['task_question'],
            'options': [
                {'label': 'Response A is better', 'value': 'A'},
                {'label': 'Response B is better', 'value': 'B'},
            ]
        }]
        client.add_batch_instructions(batch_id, instructions)

        # Initialize batch
        print("Initializing batch...")
        client.initialize_batch(
            batch_id=batch_id,
            dataset_id=dataset_id,
            tasks_per_group=cfg['prolific']['task_schema']['tasks_per_group']
        )

        print("Waiting for batch to be ready...")
        if not client.wait_for_batch_ready(batch_id):
            raise Exception("Batch setup failed. Check your task configuration.")

        # Create study
        print("\nCreating study...")
        filters = [{'filter_id': 'comparative-reasoning', 'selected_values': ['0']}]

        study_id = client.create_study(
            batch_id=batch_id,
            task_name=cfg['prolific']['task_schema']['task_name'],
            internal_name=cfg['prolific']['study_setup']['internal_name'],
            description=cfg['prolific']['task_schema']['task_introduction'].replace('\n', ' '),
            estimated_completion_time=cfg['prolific']['study_setup']['estimated_completion_time'],
            max_time=cfg['prolific']['study_setup']['max_time'],
            reward=cfg['prolific']['study_setup']['reward'],
            device_compatibility=cfg['prolific']['study_setup']['device_compatibility'],
            filters=filters
        )

        print(f"\nStudy created: {study_id}")
        print(f"View in dashboard: https://app.prolific.com/researcher/workspaces/studies/{study_id}")

        # Update participants
        print("Updating study participants...")
        client.update_study_participants(
            study_id,
            cfg['prolific']['study_setup']['participants_per_task']
        )

        # Publish study
        if not skip_publish:
            print("\nPublishing study...")
            client.publish_study(study_id)
            print("Study is now live!")
        else:
            print("\nSkipping study publication (--skip-publish flag set)")
            print(f"Study ID: {study_id}")
            print(f"Batch ID: {batch_id}")
            return

    # Wait for completion
    print(f"\nWaiting for study completion (timeout: {timeout}s = {timeout/3600:.1f} hours)...")
    print("This will poll every 60 seconds. You can safely interrupt and resume later.")

    if not client.wait_for_study_completion(study_id, timeout_sec=timeout):
        print("\nStudy didn't complete in time.")
        print(f"Study ID: {study_id}")
        print(f"Batch ID: {batch_id}")
        print("\nResume later with:")
        print(f"  python run_preference_collection.py --skip-generation --study-id {study_id} --batch-id {batch_id}")
        return

    # Fetch and process results
    print("\n" + "="*80)
    print("PROCESSING RESULTS")
    print("="*80)

    print("\nFetching responses from Prolific...")
    df_responses = client.fetch_batch_responses(batch_id)

    # Save raw responses
    raw_responses_path = output_dir / 'raw_preferences.csv'
    df_responses.to_csv(raw_responses_path, index=False)
    print(f"Saved raw responses to {raw_responses_path}")

    # Process preferences
    print("\nProcessing preferences into RLHF format...")
    df_votes = process_preferences(
        responses_df=df_responses,
        participants_per_task=cfg['prolific']['study_setup']['participants_per_task'],
        output_dir=output_dir
    )

    # Fetch demographics
    if fetch_demographics:
        print("\nFetching participant demographics...")
        df_demographics = client.fetch_study_demographics(study_id)
        demo_path = output_dir / 'demographic.csv'
        df_demographics.to_csv(demo_path, index=False)
        print(f"Saved demographics to {demo_path}")

    print("\n" + "="*80)
    print("PREFERENCE COLLECTION COMPLETE!")
    print("="*80)


def print_summary(output_dir):
    """Print summary of outputs."""
    preferences_path = output_dir / 'preferences.jsonl'

    if not preferences_path.exists():
        return

    # Count preference pairs
    with open(preferences_path, 'r') as f:
        preference_pairs = sum(1 for _ in f)

    print("\n📊 Final Statistics:")
    print(f"  Preference pairs collected: {preference_pairs}")

    print("\n📁 Output Files:")
    output_files = {
        'completions.csv': 'All LLM responses',
        'response_pairs.csv': 'Pairwise combinations',
        'raw_preferences.csv': 'Raw Prolific responses',
        'votes_preferences.csv': 'Vote counts',
        'preferences.jsonl': 'Preference dataset (main output!)',
        'demographic.csv': 'Participant demographics',
        'run_metadata.json': 'Run configuration'
    }

    for filename, description in output_files.items():
        filepath = output_dir / filename
        if filepath.exists():
            print(f"  {filepath} - {description}")

    print("\n✅ Next Steps:")
    print("  1. Use preferences.jsonl to train a reward model")
    print("  2. Compatible with: Hugging Face TRL, OpenAI PPO, Anthropic Constitutional AI")
    print("\n" + "="*80)


def main():
    """Main entry point."""
    args = parse_args()

    # Load environment variables
    if args.env_file.exists():
        load_dotenv(args.env_file)
    else:
        print(f"Warning: .env file not found at {args.env_file}")
        print("API tokens will be read from environment variables.")

    # Load config
    print(f"Loading configuration from {args.config}...")
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Create output directory
    args.output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {args.output_dir}")

    # Part 1: Generate responses
    pairs_csv_path = None
    if not args.skip_generation:
        pairs_csv_path, num_prompts = generate_responses(
            cfg=cfg,
            prompts_path=args.prompts,
            output_dir=args.output_dir,
            verbose=args.verbose
        )

        if args.generate_only:
            print("\nGeneration complete (--generate-only flag set)")
            print_summary(args.output_dir)
            return
    else:
        # Use existing pairs
        if not args.pairs:
            print("Error: --skip-generation requires --pairs argument")
            sys.exit(1)
        pairs_csv_path = args.pairs
        if not pairs_csv_path.exists():
            print(f"Error: Pairs file not found: {pairs_csv_path}")
            sys.exit(1)
        print(f"\nUsing existing response pairs from {pairs_csv_path}")

    # Part 2: Prolific workflow
    run_prolific_workflow(
        cfg=cfg,
        pairs_csv_path=pairs_csv_path,
        output_dir=args.output_dir,
        skip_publish=args.skip_publish,
        study_id=args.study_id,
        batch_id=args.batch_id,
        timeout=args.timeout,
        fetch_demographics=not args.no_demographics,
        verbose=args.verbose
    )

    # Print summary
    print_summary(args.output_dir)


if __name__ == '__main__':
    main()
