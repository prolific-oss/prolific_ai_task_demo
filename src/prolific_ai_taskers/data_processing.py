"""
Data processing utilities.
"""

import json
from itertools import combinations
from pathlib import Path

import pandas as pd


def create_response_pairs(completions_df: pd.DataFrame,
                          remove_prompt_prefix: bool = True) -> pd.DataFrame:
    """
    Create pairwise combinations of responses for annotation.

    Args:
        completions_df: DataFrame with columns [prompt_id, prompt, response]
        remove_prompt_prefix: Strip prompt text from response start

    Returns:
        DataFrame with columns [Prompt, Response A, Response B]
    """
    df = completions_df[["prompt_id", "prompt", "response"]].copy()

    # Clean up responses
    if remove_prompt_prefix:
        df["response"] = df.apply(
            lambda x: (
                x["response"].replace(x["prompt"], "", 1).strip()
                if isinstance(x["response"], str) and x["response"].startswith(x["prompt"])
                else x["response"]
            ),
            axis=1,
        )

    # Create pairs
    rows = []
    for (pid, prompt), g in df.groupby(["prompt_id", "prompt"], dropna=False):
        responses = [r for r in g["response"] if isinstance(r, str) and r.strip()]
        for a, b in combinations(responses, 2):
            rows.append({"Prompt": prompt, "Response A": a, "Response B": b})

    pairs_df = pd.DataFrame(rows).reset_index(drop=True)
    print(f"✅ Created {len(pairs_df)} response pairs")
    return pairs_df


def process_preferences(responses_df: pd.DataFrame, participants_per_task: int,
                        output_dir: Path) -> pd.DataFrame:
    """
    Process Prolific responses into RLHF dataset.

    Args:
        responses_df: Raw responses from Prolific
        participants_per_task: Number of annotators per task
        output_dir: Where to save outputs

    Returns:
        DataFrame with aggregated votes
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter columns
    annotator_pattern = r"^Annotator\d+_(ID|Response)$"
    extra_cols = ["Prompt", "Response A", "Response B"]
    df = responses_df[
        responses_df.columns[
            responses_df.columns.isin(extra_cols)
            | responses_df.columns.str.match(annotator_pattern)
        ]
    ]

    # Get annotator response columns
    annotator_cols = [col for col in df.columns if "Annotator" in col and "Response" in col]

    # Count votes
    def count_votes(row):
        votes = row[annotator_cols].dropna().tolist()
        # Handle both formats: "A"/"B" and "Response A is better"/"Response B is better"
        a_wins = sum(1 for v in votes if v == "A" or "Response A" in str(v))
        b_wins = sum(1 for v in votes if v == "B" or "Response B" in str(v))
        return pd.Series({"A_wins": a_wins, "B_wins": b_wins})

    vote_counts = df.apply(count_votes, axis=1)
    df_votes = pd.concat([df[["Prompt", "Response A", "Response B"]], vote_counts], axis=1)

    # Save votes
    votes_path = output_dir / "votes_preferences.csv"
    df_votes.to_csv(votes_path, index=False)
    print(f"Saved votes → {votes_path}")

    # Create RLHF dataset with majority voting
    def majority_label(row):
        if row["A_wins"] > row["B_wins"]:
            return row["Response A"], row["Response B"]
        elif row["B_wins"] > row["A_wins"]:
            return row["Response B"], row["Response A"]
        return None, None

    rlhf_data = []
    ties = 0
    for _, row in df_votes.iterrows():
        chosen, rejected = majority_label(row)
        if chosen and rejected:
            rlhf_data.append({
                "prompt": row["Prompt"],
                "chosen_response": chosen,
                "rejected_response": rejected,
            })
        else:
            ties += 1

    # Save RLHF dataset
    preferences_path = output_dir / "preferences.jsonl"
    with open(preferences_path, "w", encoding="utf-8") as f:
        for item in rlhf_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Created RLHF dataset: {len(rlhf_data)} pairs ({ties} ties excluded)")
    print(f"Saved preferences → {preferences_path}")

    return df_votes
