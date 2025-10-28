# Prolific AI Task Builder: RLHF Data Collection Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Status: Beta](https://img.shields.io/badge/Status-Beta-orange.svg)]()
[![Purpose: Educational](https://img.shields.io/badge/Purpose-Educational-green.svg)]()

A streamlined workflow for collecting human preference data for **Reinforcement Learning from Human Feedback (RLHF)** using the **Prolific AI Task Builder**.

This repository demonstrates how to:
1. Generate multiple LLM responses for a set of prompts
2. Collect human preferences at scale using Prolific's participant pool
3. Export preference data in a format ready for reward model training

## Features

- **LLM Response Generation**: Generate multiple completions per prompt with configurable temperature and sampling parameters
- **Prolific Integration**: Seamless integration with Prolific's AI Task Builder for pairwise preference collection
- **RLHF-Ready Output**: Export data in `(prompt, chosen, rejected)` format compatible with TRL, OpenAI PPO, and Anthropic's Constitutional AI pipelines
- **Demographic Data**: Collect participant demographics for analysis and bias detection
- **Configurable Workflows**: YAML-based configuration for model parameters and study settings

## Repository Structure

```
prolific_ai_taskers/
├── src/
│   └── prolific_ai_taskers/
│       ├── __init__.py                # Package initialization
│       ├── generate_responses.py      # LLM response generation
│       ├── prolific_client.py         # Prolific API client
│       └── data_processing.py         # Data processing utilities
├── examples/
│   ├── config.yaml                    # Configuration file
│   └── prompts.jsonl                  # Sample prompts
├── notebooks/
│   └── prolific_human_preferences_rlhf.ipynb  # Jupyter notebook workflow
├── docs/
│   └── prolific_setup.md              # Detailed Prolific setup guide
├── run_preference_collection.py       # CLI script for preference collection
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment file
└── .env.example                       # Environment variables template
```

## Getting Started

### Prerequisites

- Python 3.11+
- Prolific account with API token ([sign up here](https://app.prolific.com/register))
- Hugging Face account with API token ([get token here](https://huggingface.co/settings/tokens))
- CUDA-compatible GPU, Apple Silicon (Metal), or CPU for model inference

### Installation

We recommend using a virtual environment.

**Using Conda:**

```bash
conda env create -f environment.yml
conda activate align-rlhf
```

### Configuration

1. Set up Prolific Account. Follow the [Prolific Setup Guide](docs/prolific_setup.md) for detailed instructions.

2. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

3. Edit [.env](.env.example) with your API credentials:
   ```bash
   # Prolific API Configuration
   PROLIFIC_API_TOKEN=your_prolific_api_token_here
   PROLIFIC_WORKSPACE_ID=your_workspace_id_here
   PROLIFIC_PROJECT_ID=your_project_id_here

   # Hugging Face Token
   HF_TOKEN=your_hugging_face_api_token_here
   ```

4. Customize [examples/config.yaml](examples/config.yaml) to configure:
   - Model selection and generation parameters
   - Prolific study settings (reward, completion time, participants per task)
   - Task instructions and question phrasing

5. Customize [examples/prompts.jsonl](examples/prompts.jsonl) with your own prompts:
   - Add diverse prompts relevant to your use case
   - Include varied topics and difficulty levels for robust RLHF training
   - Each line should be a JSON object with a `prompt` field: `{"prompt": "Your question here"}`

## Usage

You can run the workflow either via the Jupyter notebook or the command-line script:

- **Notebook**: [notebooks/prolific_human_preferences_rlhf.ipynb](notebooks/prolific_human_preferences_rlhf.ipynb)
- **CLI Script**: [run_preference_collection.py](run_preference_collection.py)

### Using the CLI Script

The command-line script provides the same functionality as the notebook with additional flexibility:

```bash
# Run the full pipeline
python run_preference_collection.py

# Use custom config and prompts
python run_preference_collection.py --config my_config.yaml --prompts my_prompts.jsonl

# Only generate responses (skip Prolific)
python run_preference_collection.py --generate-only

# Skip generation and use existing response pairs
python run_preference_collection.py --skip-generation --pairs outputs/response_pairs.csv

# Resume from existing study/batch IDs
python run_preference_collection.py --skip-generation --study-id YOUR_STUDY_ID --batch-id YOUR_BATCH_ID

# View all options
python run_preference_collection.py --help
```

### Workflow Steps

#### Step 1: Generate LLM Responses
- Load prompts from [examples/prompts.jsonl](examples/prompts.jsonl)
- Generate multiple responses per prompt at different temperatures
- Create pairwise combinations for preference collection
- Export to `notebooks/outputs/response_pairs.csv`

**Key outputs:**
- `completions.csv` - All individual responses with metadata
- `response_pairs.csv` - Pairwise combinations ready for annotation
- `run_metadata.json` - Generation configuration and timing

#### Step 2: Collect Human Preferences
- Upload response pairs to Prolific AI Task Builder
- Create and publish a preference collection study
- Monitor task completion
- Download annotations and demographic data

**Key outputs:**
- `preferences.jsonl` - Final preference data in `(prompt, chosen_response, rejected_response)` format
- `votes_preferences.csv` - Aggregated vote counts per response pair
- `demographic.csv` - Participant demographic information

### Example Configuration

The [examples/config.yaml](examples/config.yaml) file allows you to customize:

```yaml
# Model configuration
model: "meta-llama/Llama-3.2-3B"
num_completions_per_prompt: 2
temperatures: [0.7, 1.0]
max_new_tokens: 512

# Prolific study configuration
prolific:
  study_setup:
    estimated_completion_time: 5  # minutes
    reward: 75  # cents
    participants_per_task: 5  # annotators per comparison
```

## Output Format

The final output ([preferences.jsonl](notebooks/outputs/preferences.jsonl)) is formatted for direct use in reward model training:

```json
{
  "prompt": "How do I make homemade pasta from scratch?",
  "chosen_response": "Making your own pasta dough is easier than you think...",
  "rejected_response": "How to make homemade pasta from scratch 1. Mix..."
}
```

This format is compatible with popular RLHF libraries like [Hugging Face TRL](https://github.com/huggingface/trl).


## Use Cases

- **AI Alignment Research**: Collect human preferences for training safer AI systems
- **Model Evaluation**: Compare multiple models or prompt strategies
- **Instruction Tuning**: Gather feedback on instruction-following quality
- **Constitutional AI**: Implement preference-based training pipelines

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Important Notice

> This project is provided **as-is** for **educational and research purposes only**.
> - 🔬 **Beta Status**: This is experimental code and may contain bugs or incomplete features
> - 📚 **Not Maintained**: No active development or support is provided
> - 🎓 **Educational Use**: Intended as a learning resource for RLHF workflows
> - ⚖️ **Use at Your Own Risk**: Test thoroughly before using in production environments
