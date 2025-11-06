# Communication Health Analysis Agent via LangGraph

A LangGraph workflow that analyzes communication health from emails and meeting transcripts, producing structured scoring across 6 key dimensions and provides key insights for communication health improvement, via LLM-powered reasoning.

## Overview

This implementation demonstrates **structured reasoning through a graph-based workflow** using LangGraph. The workflow uses LLM calls at multiple nodes to analyze communication quality, aggregate results, and generate explanations.

## Graph Structure

The workflow is structured as a **LangGraph StateGraph** with the following architecture:

![Communication Health Analysis Workflow](graph.png)

```
Input (Emails/Transcript)
    ↓
[preprocess] → Prepare content for analysis
    ↓
    ├─→ [analyze_clarity] (LLM) ──┐
    ├─→ [analyze_completeness] (LLM) ──┤
    ├─→ [analyze_correctness] (LLM) ──┤
    ├─→ [analyze_courtesy] (LLM) ─────┤
    ├─→ [analyze_audience] (LLM) ─────┤
    └─→ [analyze_timeliness] (LLM) ───┤
                                      ↓
                            [aggregate_health] → Consolidate scores (deterministic)
                                      ↓
                            [explain_health] (LLM) → Generate explanation
                                      ↓
                              Output (JSON + Explanation)
```

### Node Types

1. **Preprocessing Node** (Deterministic)

   - Handles email threads or meeting transcripts
   - Extracts metadata (participants, timestamps, speaker info)
   - Prepares unified content format
2. **Analysis Nodes** (6 parallel LLM nodes)

   - Each node makes an independent LLM call
   - Evaluates one dimension of communication health
   - Returns structured JSON: `{"score": float, "reasoning": str}`
3. **Aggregation Node** (Deterministic)

   - Waits for all 6 dimensions to complete
   - Calculates overall health score (unweighted average)
   - Consolidates dimension scores
4. **Explanation Node** (LLM)

   - Synthesizes all results into natural language
   - Provides actionable recommendations
   - Demonstrates LLM reasoning for interpretation

## Modeling Approach: Communication Health

### Why 6 Dimensions?

Communication health is modeled as a **multi-dimensional construct** because effective communication requires excellence across multiple aspects:

1. **Clarity** - Can the message be understood?
2. **Completeness** - Is all necessary information present?
3. **Correctness** - Is the information accurate and coherent?
4. **Courtesy** - Is the tone appropriate and respectful?
5. **Audience-Centricity** - Is it tailored to the recipient?
6. **Timeliness** - Is it sent at the right time?

This framework captures both **content quality** (clarity, completeness, correctness) and **interpersonal effectiveness** (courtesy, audience, timeliness).

### Why Parallel Analysis?

The 6 analysis nodes run **in parallel** because:

- **Independence**: Each dimension can be evaluated independently
- **Performance**: Parallel execution reduces total analysis time
- **Scalability**: Easy to add more dimensions without restructuring
- **LangGraph Feature**: Demonstrates graph-based parallel execution

### LangGraph Benefits

LangGraph provides:

- **State Management**: Automatic handling of complex state transitions
- **Parallel Execution**: Built-in support for concurrent node execution
- **Visualization**: Can be visualized in LangGraph Studio
- **Error Handling**: Built-in mechanisms for workflow failures
- **Extensibility**: Easy to add conditional logic or feedback loops

### LLM at Multiple Nodes

LLMs are used at **7 nodes** (6 analysis + 1 explanation) because:

- **Specialized Reasoning**: Each dimension requires nuanced evaluation
- **Structured Output**: JSON parsing ensures consistent format
- **Interpretability**: Natural language reasoning for each score
- **Synthesis**: Final LLM call combines all insights into coherent explanation

This demonstrates **multi-step LLM reasoning** where each step builds on previous results.

## Communication Health Dimensions

### 1. Clarity (0.0-1.0)

**Factors:** Readability, directness, jargon usage, brevity

- **Excellent (0.8-1.0)**: Clear, direct, no jargon issues
- **Good (0.6-0.8)**: Mostly clear, minor improvements needed
- **Fair (0.4-0.6)**: Some clarity issues, needs revision
- **Poor (0.0-0.4)**: Unclear, confusing, requires significant revision

### 2. Completeness (0.0-1.0)

**Factors:** Sufficient information, actionable details, clear next steps

- **Excellent (0.8-1.0)**: All information present, no follow-up needed
- **Good (0.6-0.8)**: Most information present, minor gaps
- **Fair (0.4-0.6)**: Missing some important information
- **Poor (0.0-0.4)**: Critical information missing

### 3. Correctness (0.0-1.0)

**Factors:** Factual accuracy, grammar/spelling, logical flow, consistent tone

- **Excellent (0.8-1.0)**: No errors, perfect coherence
- **Good (0.6-0.8)**: Minor errors, mostly coherent
- **Fair (0.4-0.6)**: Some errors or coherence issues
- **Poor (0.0-0.4)**: Multiple errors, confusing flow

### 4. Courtesy (0.0-1.0)

**Factors:** Politeness, respect, empathy, professionalism

- **Excellent (0.8-1.0)**: Highly courteous, professional, empathetic
- **Good (0.6-0.8)**: Polite and professional
- **Fair (0.4-0.6)**: Adequate but could be more courteous
- **Poor (0.0-0.4)**: Discourteous, unprofessional

### 5. Audience-Centricity (0.0-1.0)

**Factors:** Relevance, appropriate knowledge level, personalization, context awareness

- **Excellent (0.8-1.0)**: Highly personalized, perfectly relevant
- **Good (0.6-0.8)**: Relevant and appropriately tailored
- **Fair (0.4-0.6)**: Somewhat relevant but generic
- **Poor (0.0-0.4)**: Not relevant or completely generic

### 6. Timeliness (0.0-1.0)

**Factors:** Appropriate timing, response delays, urgency handling, follow-up frequency

- **Excellent (0.8-1.0)**: Perfect timing, appropriate urgency
- **Good (0.6-0.8)**: Generally timely, minor timing issues
- **Fair (0.4-0.6)**: Some timing problems or urgency mismatch
- **Poor (0.0-0.4)**: Untimely, inappropriate urgency


## To Run the Workflow in Langraph Studio:

## Install Requirements

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file or set environment variables (I will include a fully working .env file to be utilised for this demo):

```bash
# Required: NVIDIA API for DeepSeek
NVIDIA_API_KEY=your_nvidia_api_key
NVIDIA_MODEL=deepseek-ai/deepseek-v3.1-terminus
NVIDIA_API_URL=https://integrate.api.nvidia.com/v1

```
Place the .env in the root path of this repo.

### Start the Langgraph Studio Server:

From the root of this repo, in the terminal, run:

```
langgraph dev
```

### Included Email and Meeting Transcription Samples

For the demonstration of this workflow, I've included sample emails and meetings in the expected JSON format.
They are in the folder: example_communication_files, e.g., meeting_transcript_sample2.json, email_sample_3.json.
To test the workflow, once in the Langgraph studio page, paste the entire contents of any of the JSON files
into the input bar, under the graph. The system will automatically determine if it is a meeting or an email. Press Submit.
Watch the analysis happen in real-time and check on any of the stages.
I will also include short video demonstrating the workflow.

## Design Decisions

### This Graph Structure Includes

1. **Preprocessing First**: Ensures consistent input format before analysis
2. **Parallel Analysis**: 6 dimensions analyzed simultaneously for efficiency
3. **Aggregation Before Explanation**: Explanation node needs complete scores
4. **Sequential Final Steps**: Aggregation → Explanation → End (logical flow)



## File Structure

```
var_comm_health/
├── __init__.py          # Module exports
├── config.py            # Configuration management
├── workflow.py          # LangGraph workflow (THIS IS THE CORE)
├── analyzer.py          # Helper functions (optional, for standalone use)
├── llm_factory.py       # LLM instance creation
├── langgraph_studio_server.py  # LangGraph Studio server entry point
├── langgraph.json       # LangGraph Studio configuration
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## Features

 **LangGraph Workflow**: Proper StateGraph with nodes and edges
 **LLM at Multiple Nodes**: 7 efficient LLM calls for specialized reasoning
 **Parallel Execution**: 6 dimensions analyzed simultaneously
 **Thread/Transcript Support**: Handles both input types natively
 **Structured Output**: JSON format with scores and reasoning
 **Natural Language Explanation**: LLM-generated summary
 **DeepSeek via NVIDIA**: Uses DeepSeek v3.1-terminus model

## Evaluation Criteria Alignment

-  **Product Thinking**: 6-dimension framework models communication health comprehensively
-  **LangGraph Structure**: Proper StateGraph with clear node/edge relationships
-  **Readability**: Clean code with type hints and documentation
-  **Smart LLM Use**: Purposeful LLM calls at analysis and explanation nodes

