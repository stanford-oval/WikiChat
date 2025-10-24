# PersuaBot

PersuaBot is a zero-shot persuasive chatbot with LLM-generated strategies and information retrieval, based on the paper:

**"Zero-shot Persuasive Chatbots with LLM-Generated Strategies and Information Retrieval"**
Kazuaki Furumai, Roberto Legaspi, Julio Vizcarra, Yudai Yamazaki, Yasutaka Nishimura, Sina J. Semnani, Kazushi Ikeda, Weiyan Shi, and Monica S. Lam
arXiv:2407.03585 (2024)

## Overview

PersuaBot addresses a major limitation of prior persuasive chatbot approaches: existing methods rely on fine-tuning with task-specific training data which is costly to collect, and they employ only a handful of pre-defined persuasion strategies.

PersuaBot is a **multi-step pipeline of LLM calls with in-context learning** that:
- Generates persuasive responses using diverse, dynamically-generated strategies
- Ensures factual correctness through systematic fact-checking
- Retrieves factual information to substantiate claims
- Works in a zero-shot manner without requiring domain-specific training data

## Architecture

PersuaBot consists of two parallel modules that work together:

### 1. Question Handling Module (QHM)
The QHM handles direct user questions by:
- Identifying when the user asks for factual information
- Generating appropriate search queries
- Retrieving relevant information from the corpus

### 2. Strategy Maintenance Module (SMM)
The SMM is the core of PersuaBot's persuasive capabilities:

1. **Response Generation**: Generates an initial persuasive response using various strategies
2. **Strategy Decomposition**: Breaks down the response into sections, each with a distinct persuasion strategy
3. **Fact-Checking**: Verifies each section for factual accuracy
4. **Information Retrieval**: For unverified sections, retrieves factual information from the corpus
5. **Result Merging**: Combines persuasion strategies with factual information to create the final response

### Key Concept

The key innovation is to **separate persuasion strategies from factual content**:
- Keep the persuasive intent and strategies
- Replace LLM-generated claims with verified facts when needed
- Ensure both persuasiveness and factual accuracy

## Installation

PersuaBot is built on top of the WikiChat framework and uses the same dependencies. Follow the WikiChat installation instructions in the main [README.md](README.md).

## Usage

### Basic Usage

Run PersuaBot with default settings:

```bash
python -m command_line_persuabot --engine gpt-4o
```

### Specify Persuasion Domain and Goal

PersuaBot can be customized for different persuasion scenarios:

```bash
# Donation solicitation
python -m command_line_persuabot \
  --engine gpt-4o \
  --persuasion_domain "donation" \
  --target_goal "encourage donation to education charities"

# Health intervention
python -m command_line_persuabot \
  --engine gpt-4o \
  --persuasion_domain "health" \
  --target_goal "encourage healthier lifestyle choices"

# Recommendation
python -m command_line_persuabot \
  --engine gpt-4o \
  --persuasion_domain "recommendation" \
  --target_goal "recommend books similar to user preferences"
```

### Advanced Options

```bash
# Disable fact-checking (faster but less accurate)
python -m command_line_persuabot \
  --engine gpt-4o \
  --no_fact_checking

# Disable strategy decomposition (treats response as single unit)
python -m command_line_persuabot \
  --engine gpt-4o \
  --no_strategy_decomposition

# Enable debug mode to see strategy breakdown
python -m command_line_persuabot \
  --engine gpt-4o \
  --debug
```

### Retrieval Configuration

PersuaBot uses the same retrieval system as WikiChat:

```bash
# Use default Wikipedia API (rate-limited)
python -m command_line_persuabot --engine gpt-4o

# Use custom corpus
python -m command_line_persuabot \
  --engine gpt-4o \
  --corpus_id "my_corpus" \
  --retriever_endpoint "http://localhost:5100/my_collection"
```

### All Available Options

```bash
python -m command_line_persuabot --help
```

Key parameters:
- `--engine`: LLM to use (default: gpt-4o)
- `--persuasion_domain`: Domain for persuasion (e.g., 'donation', 'health', 'recommendation')
- `--target_goal`: Specific persuasion goal
- `--do_fact_checking` / `--no_fact_checking`: Enable/disable fact-checking (default: enabled)
- `--do_strategy_decomposition` / `--no_strategy_decomposition`: Enable/disable strategy decomposition (default: enabled)
- `--debug`: Show strategy breakdown for each response
- `--corpus_id`: ID of the corpus to use
- `--retriever_endpoint`: Endpoint for information retrieval
- `--do_reranking`: Enable reranking of search results
- `--query_post_reranking_num`: Number of documents to retrieve for user queries
- `--claim_post_reranking_num`: Number of evidences per claim

## Example Conversations

### Donation Solicitation

```
User: I'm thinking about donating to charity.
PersuaBot: That's wonderful! Donating to charity can make a real difference...
[Uses social proof, statistics, and authority to encourage donation while ensuring all claims are fact-checked]
```

### Health Intervention

```
User: I've been feeling really tired lately.
PersuaBot: I'm sorry to hear that. According to the CDC, about 35% of adults...
[Provides fact-checked health information using persuasive strategies]
```

### Recommendation

```
User: I just finished The Hunger Games. Any recommendations?
PersuaBot: If you enjoyed The Hunger Games, you might like The Maze Runner series...
[Uses ratings, sales figures, and social proof to recommend books]
```

## Implementation Details

### File Structure

```
WikiChat/
├── pipelines/
│   ├── persuabot.py                  # Main PersuaBot pipeline
│   ├── persuabot_dialogue_state.py   # State management for PersuaBot
│   └── prompts/
│       └── persuabot/
│           ├── qhm_query.prompt          # QHM query generation
│           ├── smm_generate.prompt       # SMM response generation
│           ├── smm_decompose.prompt      # Strategy decomposition
│           ├── smm_fact_check.prompt     # Fact-checking
│           ├── smm_generate_query.prompt # Query generation for retrieval
│           └── smm_merge.prompt          # Merge strategies with facts
├── command_line_persuabot.py         # Command-line interface
└── PERSUABOT.md                      # This file
```

### Pipeline Stages

1. **QHM Query Stage**: Generates search queries from user questions
2. **QHM Search Stage**: Retrieves information for user queries
3. **SMM Generate Response**: Creates initial persuasive response
4. **SMM Decompose Strategies**: Breaks response into strategy sections
5. **SMM Fact Check**: Verifies factual accuracy of each section
6. **SMM Retrieve Facts**: Gets factual information for unverified sections
7. **SMM Merge Results**: Combines strategies with facts to create final response

## Persuasion Strategies

PersuaBot can dynamically employ various persuasion strategies, including:

- **Statistics**: Using numbers and data to support claims
- **Social Proof**: Referencing what others do or believe
- **Authority**: Citing expert opinions or authoritative sources
- **Emotional Appeal**: Connecting with emotions
- **Scarcity**: Highlighting limited availability or urgency
- **Reciprocity**: Creating a sense of mutual benefit
- **Comparison**: Comparing options or alternatives
- **Storytelling**: Using narratives and examples
- **Expert Testimony**: Quoting or referencing experts
- **Logical Reasoning**: Using rational arguments

The chatbot automatically selects and combines appropriate strategies based on the conversation context.

## Performance

According to the paper, PersuaBot:
- Achieves up to **26.6% higher factuality** than GPT-3.5
- Has an advantage of **0.6 points on a 5-point persuasiveness scale** compared to manually designed rule-oriented methods
- Works across multiple domains: donation solicitation, recommendation systems, and healthcare intervention

## Citation

If you use PersuaBot in your research, please cite:

```bibtex
@article{furumai2024persuabot,
  title={Zero-shot Persuasive Chatbots with LLM-Generated Strategies and Information Retrieval},
  author={Furumai, Kazuaki and Legaspi, Roberto and Vizcarra, Julio and
          Yamazaki, Yudai and Nishimura, Yasutaka and Semnani, Sina J. and
          Ikeda, Kazushi and Shi, Weiyan and Lam, Monica S.},
  journal={arXiv preprint arXiv:2407.03585},
  year={2024}
}
```

## Differences from WikiChat

While PersuaBot is built on the WikiChat framework, it has several key differences:

| Feature | WikiChat | PersuaBot |
|---------|----------|-----------|
| **Purpose** | Factual, informative conversations | Persuasive conversations with factual grounding |
| **Response Style** | Neutral, informative | Persuasive, engaging |
| **Strategy** | Fact-checking claims | Generating + fact-checking persuasive strategies |
| **Modules** | Single pipeline | Dual modules (QHM + SMM) |
| **Strategy Decomposition** | N/A | Decomposes response by persuasion strategies |
| **Use Cases** | Information seeking | Donation, health, recommendations, persuasion |

## License

PersuaBot code is released under Apache-2.0 license, consistent with the WikiChat repository.

## Acknowledgments

PersuaBot is implemented based on the paper by Furumai et al. (2024) and uses the WikiChat framework developed by Stanford OVAL Lab.
