"""LangGraph workflow for Communication Health Analysis.

This module implements a LangGraph workflow that analyzes communication health
using LLM-powered analysis, across six dimensions: clarity, completeness, correctness, courtesy, audience-centricity, and timeliness."""

from __future__ import annotations

import json
import logging
from typing import List, Optional, Dict, Any, Literal, Annotated
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

# Maximum context length to prevent prompt bloat
MAX_CONTEXT_LENGTH = 2000


class CommunicationHealthState(BaseModel):
    """
    State for communication health analysis workflow using LangGraph 1.x patterns.
    
    Supports two input types:
    1. Email threads: List of email messages with timestamps
    2. Meeting transcripts: Single transcript text with speaker labels
    
    Uses separate fields for each dimension to avoid concurrent write conflicts.
    """
    # Input data
    input_type: Literal["emails", "transcript"]
    emails: Optional[List[Dict[str, Any]]] = None  # For email threads: [{"from": str, "subject": str, "body": str, "date": str, ...}]
    transcript: Optional[str] = None  # For meeting transcripts: "Speaker 1: ...\nSpeaker 2: ..."
    
    # Preprocessing
    preprocessed_content: Optional[str] = None  # Unified content for analysis
    context: Optional[Dict[str, Any]] = None  # Additional context (metadata, timing, etc.)
    
    # Individual dimension scores (to avoid concurrent write conflicts)
    clarity_score: Optional[Dict[str, Any]] = None
    completeness_score: Optional[Dict[str, Any]] = None
    correctness_score: Optional[Dict[str, Any]] = None
    courtesy_score: Optional[Dict[str, Any]] = None
    audience_score: Optional[Dict[str, Any]] = None
    timeliness_score: Optional[Dict[str, Any]] = None
    
    # Aggregation
    communication_health_scores: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    aggregated_health: Optional[Dict[str, Any]] = None
    
    # Explanation
    health_explanation: Optional[str] = None
    
    # Status
    status: str = "ready"
    error: Optional[str] = None


def _format_context(context: Optional[Dict[str, Any]], max_length: int = MAX_CONTEXT_LENGTH) -> str:
    """Format context for prompts with truncation if needed."""
    if not context:
        return ""
    
    context_str = str(context)
    
    if len(context_str) > max_length:
        context_str = context_str[:max_length] + "... [truncated]"
        logger.warning(f"Context truncated to {max_length} characters")
    
    return f"\nContext: {context_str}"


def _parse_analysis_result(result: str, dimension_name: str) -> Dict[str, Any]:
    """Parse LLM analysis result with standardized format."""
    try:
        # Clean the result string - remove markdown code blocks and whitespace
        cleaned_result = result.strip()
        if cleaned_result.startswith("```json"):
            cleaned_result = cleaned_result[7:]
        if cleaned_result.startswith("```"):
            cleaned_result = cleaned_result[3:]
        if cleaned_result.endswith("```"):
            cleaned_result = cleaned_result[:-3]
        cleaned_result = cleaned_result.strip()
        
        analysis = json.loads(cleaned_result)
        
        # Try multiple possible score keys
        score = (
            analysis.get(f"{dimension_name}_score") or
            analysis.get("score") or
            0.5
        )
        
        return {
            "score": float(score),
            "reasoning": analysis.get("reasoning", "")
        }
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse {dimension_name} analysis JSON: {e}. Raw result: {result[:200]}")
        return {
            "score": 0.5,
            "reasoning": f"Analysis failed: JSON parse error"
        }
    except Exception as e:
        logger.error(f"Failed to parse {dimension_name} analysis: {e}")
        return {
            "score": 0.5,
            "reasoning": "Analysis failed"
        }
def preprocess_node(state: CommunicationHealthState) -> dict:
    """
    LangGraph Node: Preprocess input data into unified format for analysis.
    
    Handles both email threads and meeting transcripts, extracting relevant
    information and preparing context for analysis.
    
    Returns:
        Dictionary with updates to apply to state
    """
    logger.info("Node: Preprocessing communication content")
    
    updates = {
        "status": "preprocessing"
    }
    
    try:
        if state.input_type == "emails":
            # Process email thread
            emails = state.emails or []
            if not emails:
                return {"error": "No emails provided", "status": "error"}
            
            # Combine emails into a single text with thread structure
            email_texts = []
            participants = set()
            timestamps = []
            
            for email in emails:
                sender = email.get("from", "Unknown")
                subject = email.get("subject", "")
                body = email.get("body", "")
                date = email.get("date") or email.get("sent_at", "")
                
                participants.add(sender)
                if date:
                    timestamps.append(date)
                
                email_text = f"From: {sender}\nDate: {date}\nSubject: {subject}\n\n{body}\n"
                email_texts.append(email_text)
            
            updates["preprocessed_content"] = "\n---\n".join(email_texts)
            
            # Extract context from emails
            updates["context"] = {
                "type": "email_thread",
                "email_count": len(emails),
                "participants": list(participants),
                "timestamps": timestamps,
                "has_thread_structure": len(emails) > 1
            }
            
        elif state.input_type == "transcript":
            # Process meeting transcript
            transcript = state.transcript or ""
            if not transcript:
                return {"error": "No transcript provided", "status": "error"}
            
            updates["preprocessed_content"] = transcript
            
            # Extract speaker information if available
            speakers = set()
            for line in transcript.split("\n"):
                if ":" in line:
                    speaker = line.split(":")[0].strip()
                    if speaker:
                        speakers.add(speaker)
            
            updates["context"] = {
                "type": "meeting_transcript",
                "speaker_count": len(speakers),
                "speakers": list(speakers) if speakers else []
            }
        else:
            return {"error": f"Unknown input_type: {state.input_type}", "status": "error"}
        
        updates["status"] = "preprocessed"
        updates["communication_health_scores"] = {}
        
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return {"error": str(e), "status": "error"}
    
    return updates


def analyze_clarity_node(state: CommunicationHealthState, llm: BaseChatModel) -> dict:
    """
    LangGraph Node: Analyze clarity dimension using LLM.
    
    This node demonstrates LLM-powered reasoning for evaluating communication clarity.
    
    Returns:
        Dictionary with updates to apply to state
    """
    logger.info("Node: Analyzing clarity")
    
    content = state.preprocessed_content or ""
    if not content or not content.strip():
        return {
            "clarity_score": {
                "score": 0.0,
                "reasoning": "Empty content cannot be analyzed"
            }
        }
    
    # LLM call directly in the graph node
    prompt = ChatPromptTemplate.from_template(
        """
        Evaluate the clarity and conciseness of this communication on a scale of 0.0 to 1.0.
        Consider: readability, directness, absence of jargon, brevity.
        
        Examples:
        - "Buy milk." → {{"score": 0.8, "reasoning": "Direct but lacks context"}}
        - "I wanted to reach out to you regarding the possibility of potentially scheduling a meeting at some point in the near future to discuss various topics." → {{"score": 0.3, "reasoning": "Wordy, indirect, unclear purpose"}}
        - "Let's meet tomorrow at 3pm to discuss the Q4 budget." → {{"score": 0.95, "reasoning": "Clear, direct, specific"}}
        
        Content:
        {content}
        
        {context}
        
        Respond with JSON: {{"score": 0.0-1.0, "reasoning": "brief explanation"}}
        """
    )
    
    context_str = _format_context(state.context)
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "content": content,
        "context": context_str
    })
    
    return {
        "clarity_score": _parse_analysis_result(result, "clarity")
    }


def analyze_completeness_node(state: CommunicationHealthState, llm: BaseChatModel) -> dict:
    """LangGraph Node: Analyze completeness dimension using LLM."""
    logger.info("Node: Analyzing completeness")
    
    content = state.preprocessed_content or ""
    if not content or not content.strip():
        return {
            "completeness_score": {
                "score": 0.0,
                "reasoning": "Empty content cannot be analyzed"
            }
        }
    
    prompt = ChatPromptTemplate.from_template(
        """
        Evaluate the completeness of this communication on a scale of 0.0 to 1.0.
        Consider: sufficient information, actionable details, clear next steps, answers potential questions.
        
        Examples:
        - "The report is delayed." → {{"score": 0.2, "reasoning": "Missing why, when, what's next"}}
        - "The Q3 report is delayed due to data issues. I'll fix it by Friday EOD. Let me know if you need anything else." → {{"score": 0.85, "reasoning": "Includes cause, timeline, and next steps"}}
        
        Content:
        {content}
        
        {context}
        
        Respond with JSON: {{"score": 0.0-1.0, "reasoning": "brief explanation"}}
        """
    )
    
    context_str = _format_context(state.context)
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "content": content,
        "context": context_str
    })
    
    return {
        "completeness_score": _parse_analysis_result(result, "completeness")
    }


def analyze_correctness_node(state: CommunicationHealthState, llm: BaseChatModel) -> dict:
    """LangGraph Node: Analyze correctness dimension using LLM."""
    logger.info("Node: Analyzing correctness")
    
    content = state.preprocessed_content or ""
    if not content or not content.strip():
        return {
            "correctness_score": {
                "score": 0.0,
                "reasoning": "Empty content cannot be analyzed"
            }
        }
    
    prompt = ChatPromptTemplate.from_template(
        """
        Evaluate the correctness and coherence of this communication on a scale of 0.0 to 1.0.
        Consider: factual accuracy, grammar/spelling, logical flow, consistent tone.
        
        Examples:
        - "The meeting is tomorrow, but it was yesterday." → {{"score": 0.2, "reasoning": "Logical contradiction"}}
        - "Hi team, the Q3 report shows 15% growth. We exceeded targets." → {{"score": 0.9, "reasoning": "Clear, grammatically correct, coherent"}}
        
        Content:
        {content}
        
        {context}
        
        Respond with JSON: {{"score": 0.0-1.0, "reasoning": "brief explanation"}}
        """
    )
    
    context_str = _format_context(state.context)
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "content": content,
        "context": context_str
    })
    
    return {
        "correctness_score": _parse_analysis_result(result, "correctness")
    }


def analyze_courtesy_node(state: CommunicationHealthState, llm: BaseChatModel) -> dict:
    """LangGraph Node: Analyze courtesy dimension using LLM."""
    logger.info("Node: Analyzing courtesy")
    
    content = state.preprocessed_content or ""
    if not content or not content.strip():
        return {
            "courtesy_score": {
                "score": 0.0,
                "reasoning": "Empty content cannot be analyzed"
            }
        }
    
    prompt = ChatPromptTemplate.from_template(
        """
        Evaluate the courtesy and tone of this communication on a scale of 0.0 to 1.0.
        Consider: politeness, respect, empathy, professionalism.
        
        Examples:
        - "Do this now." → {{"score": 0.3, "reasoning": "Demanding, lacks politeness"}}
        - "Hi, could you please review this when you have a moment? Thank you!" → {{"score": 0.95, "reasoning": "Polite, respectful, professional"}}
        
        Content:
        {content}
        
        {context}
        
        Respond with JSON: {{"score": 0.0-1.0, "reasoning": "brief explanation"}}
        """
    )
    
    context_str = _format_context(state.context)
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "content": content,
        "context": context_str
    })
    
    return {
        "courtesy_score": _parse_analysis_result(result, "courtesy")
    }


def analyze_audience_node(state: CommunicationHealthState, llm: BaseChatModel) -> dict:
    """LangGraph Node: Analyze audience-centricity dimension using LLM."""
    logger.info("Node: Analyzing audience")
    
    content = state.preprocessed_content or ""
    if not content or not content.strip():
        return {
            "audience_score": {
                "score": 0.0,
                "reasoning": "Empty content cannot be analyzed"
            }
        }
    
    prompt = ChatPromptTemplate.from_template(
        """
        Evaluate how well this communication is tailored to its audience on a scale of 0.0 to 1.0.
        Consider: relevance to recipient, appropriate knowledge level, personalized content.
        
        Examples:
        - Generic mass email with no personalization → {{"score": 0.3, "reasoning": "Not tailored to specific audience"}}
        - "Hi Sarah, based on our discussion about the API integration, here's the updated documentation..." → {{"score": 0.9, "reasoning": "Personalized, relevant to recipient's context"}}
        
        Content:
        {content}
        
        {context}
        
        Respond with JSON: {{"score": 0.0-1.0, "reasoning": "brief explanation"}}
        """
    )
    
    context_str = _format_context(state.context)
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "content": content,
        "context": context_str
    })
    
    return {
        "audience_score": _parse_analysis_result(result, "audience")
    }


def analyze_timeliness_node(state: CommunicationHealthState, llm: BaseChatModel) -> dict:
    """LangGraph Node: Analyze timeliness dimension using LLM."""
    logger.info("Node: Analyzing timeliness")
    
    content = state.preprocessed_content or ""
    if not content or not content.strip():
        return {
            "timeliness_score": {
                "score": 0.0,
                "reasoning": "Empty content cannot be analyzed"
            }
        }
    
    # Extract timing metadata from context if available
    context = state.context.copy() if state.context else {}
    
    # Calculate response times if timestamps available
    if state.input_type == "emails" and state.emails:
        emails = state.emails
        if len(emails) > 1:
            # Calculate response times between emails
            response_times = []
            for i in range(1, len(emails)):
                prev_date = emails[i-1].get("date") or emails[i-1].get("sent_at")
                curr_date = emails[i].get("date") or emails[i].get("sent_at")
                if prev_date and curr_date:
                    try:
                        prev_dt = datetime.fromisoformat(str(prev_date).replace('Z', '+00:00'))
                        curr_dt = datetime.fromisoformat(str(curr_date).replace('Z', '+00:00'))
                        response_time = (curr_dt - prev_dt).total_seconds() / 3600  # hours
                        response_times.append(response_time)
                    except:
                        pass
            
            if response_times:
                context["average_response_time_hours"] = sum(response_times) / len(response_times)
                context["response_times"] = response_times
    
    prompt = ChatPromptTemplate.from_template(
        """
        Evaluate the timeliness and responsiveness of this communication on a scale of 0.0 to 1.0.
        Consider: appropriate timing, response delays, frequency of follow-ups, urgency handling.
        
        Examples:
        - Urgent request sent at 11pm on Friday → {{"score": 0.3, "reasoning": "Poor timing for urgent request"}}
        - Response sent within 2 hours of request during business hours → {{"score": 0.9, "reasoning": "Timely and appropriate"}}
        
        Content:
        {content}
        
        {context}
        
        Respond with JSON: {{"score": 0.0-1.0, "reasoning": "brief explanation"}}
        """
    )
    
    context_str = _format_context(context)
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "content": content,
        "context": context_str
    })
    
    return {
        "timeliness_score": _parse_analysis_result(result, "timeliness")
    }


def aggregate_health_node(state: CommunicationHealthState) -> dict:
    """
    LangGraph Node: Aggregate all health scores.
    
    This node consolidates the parallel analysis results into a single
    overall health score and dimension averages.
    
    Returns:
        Dictionary with updates to apply to state
    """
    logger.info("Node: Aggregating health scores")
    
    # Collect individual dimension scores
    dimension_scores = {}
    health_scores = {}
    
    # Check if all 6 dimensions are available
    expected_dims = {
        'clarity': state.clarity_score,
        'completeness': state.completeness_score,
        'correctness': state.correctness_score,
        'courtesy': state.courtesy_score,
        'audience': state.audience_score,
        'timeliness': state.timeliness_score
    }
    
    # Check if all dimensions have been computed
    missing_dims = [dim for dim, score in expected_dims.items() if score is None]
    if missing_dims:
        logger.info(f"Waiting for all dimensions. Missing: {missing_dims}")
        return {}
    
    # Build the health_scores dictionary and calculate dimension scores
    for dim_name, dim_data in expected_dims.items():
        if dim_data is not None:
            health_scores[dim_name] = dim_data
            dimension_scores[dim_name] = dim_data.get("score", 0.5)
        else:
            dimension_scores[dim_name] = 0.5
            health_scores[dim_name] = {"score": 0.5, "reasoning": "No data available"}
    
    # Calculate overall health score (unweighted average)
    overall_health = sum(dimension_scores.values()) / len(dimension_scores) if dimension_scores else 0.5
    
    return {
        "communication_health_scores": health_scores,
        "aggregated_health": {
            "overall": overall_health,
            "dimensions": dimension_scores,
            "total_dimensions": len(expected_dims)
        },
        "status": "aggregated"
    }


def explain_health_node(state: CommunicationHealthState, llm: BaseChatModel) -> dict:
    """
    LangGraph Node: Generate natural language explanation using LLM.
    
    This node synthesizes all analysis results into a coherent explanation,
    demonstrating LLM-powered reasoning for interpretation.
    
    Returns:
        Dictionary with updates to apply to state
    """
    logger.info("Node: Generating health explanation")
    
    aggregated = state.aggregated_health
    health_scores = state.communication_health_scores or {}
    
    if not aggregated:
        return {
            "health_explanation": "Unable to generate explanation: no aggregated scores",
            "status": "completed"
        }
    
    # LLM call directly in the graph node
    prompt = ChatPromptTemplate.from_template(
        """
        Based on the communication health analysis results, provide a concise natural language explanation
        summarizing the main findings and actionable recommendations.
        
        Overall Health Score: {overall_score}
        Dimension Scores: {dimensions}
        
        Key findings from each dimension:
        - Clarity: {clarity_findings}
        - Completeness: {completeness_findings}
        - Correctness: {correctness_findings}
        - Courtesy: {courtesy_findings}
        - Audience-Centricity: {audience_findings}
        - Timeliness: {timeliness_findings}
        
        Provide a 2-3 paragraph summary explaining:
        1. Overall communication health assessment
        2. Key strengths and weaknesses
        3. Actionable recommendations for improvement
        """
    )
    
    # Extract reasoning from each dimension
    dimension_findings = {}
    for dim in ['clarity', 'completeness', 'correctness', 'courtesy', 'audience', 'timeliness']:
        dim_data = health_scores.get(dim, {})
        if isinstance(dim_data, dict):
            dimension_findings[dim] = dim_data.get('reasoning', 'No findings')
        else:
            dimension_findings[dim] = 'No data available'
    
    chain = prompt | llm | StrOutputParser()
    explanation = chain.invoke({
        "overall_score": aggregated.get('overall', 0.5),
        "dimensions": aggregated.get('dimensions', {}),
        "clarity_findings": dimension_findings.get('clarity', ''),
        "completeness_findings": dimension_findings.get('completeness', ''),
        "correctness_findings": dimension_findings.get('correctness', ''),
        "courtesy_findings": dimension_findings.get('courtesy', ''),
        "audience_findings": dimension_findings.get('audience', ''),
        "timeliness_findings": dimension_findings.get('timeliness', '')
    })
    
    return {
        "health_explanation": explanation,
        "status": "completed"
    }


def create_communication_health_workflow(llm: BaseChatModel) -> StateGraph:
    """
    Create a LangGraph workflow for communication health analysis using LangGraph 1.x patterns.
    
    This function demonstrates structured reasoning through a graph-based approach:
    
    Workflow Structure:
    1. preprocess -> Prepare content for analysis (handles threads/transcripts)
    2. Parallel analysis (6 nodes with LLM calls):
       - analyze_clarity (LLM)
       - analyze_completeness (LLM)
       - analyze_correctness (LLM)
       - analyze_courtesy (LLM)
       - analyze_audience (LLM)
       - analyze_timeliness (LLM)
    3. aggregate_health -> Consolidate scores (deterministic)
    4. explain_health -> Generate explanation (LLM)
    
    The graph uses LangGraph's StateGraph to manage state transitions and
    parallel execution of analysis nodes with separate fields to avoid conflicts.
    
    Args:
        llm: Language model instance (e.g., DeepSeek via NVIDIA)
        
    Returns:
        Compiled LangGraph workflow
    """
    # Create workflow with Pydantic state model
    workflow = StateGraph(CommunicationHealthState)
    
    # Create node functions that bind the LLM
    def clarity_node(state): return analyze_clarity_node(state, llm)
    def completeness_node(state): return analyze_completeness_node(state, llm)
    def correctness_node(state): return analyze_correctness_node(state, llm)
    def courtesy_node(state): return analyze_courtesy_node(state, llm)
    def audience_node(state): return analyze_audience_node(state, llm)
    def timeliness_node(state): return analyze_timeliness_node(state, llm)
    def explain_node(state): return explain_health_node(state, llm)
    
    # Add nodes to the graph
    workflow.add_node("preprocess", preprocess_node)
    workflow.add_node("analyze_clarity", clarity_node)
    workflow.add_node("analyze_completeness", completeness_node)
    workflow.add_node("analyze_correctness", correctness_node)
    workflow.add_node("analyze_courtesy", courtesy_node)
    workflow.add_node("analyze_audience", audience_node)
    workflow.add_node("analyze_timeliness", timeliness_node)
    workflow.add_node("aggregate_health", aggregate_health_node)
    workflow.add_node("explain_health", explain_node)
    
    # Set entry point
    workflow.set_entry_point("preprocess")
    
    # Preprocessing feeds into all 6 parallel analysis nodes
    workflow.add_edge("preprocess", "analyze_clarity")
    workflow.add_edge("preprocess", "analyze_completeness")
    workflow.add_edge("preprocess", "analyze_correctness")
    workflow.add_edge("preprocess", "analyze_courtesy")
    workflow.add_edge("preprocess", "analyze_audience")
    workflow.add_edge("preprocess", "analyze_timeliness")
    
    # All parallel nodes feed into aggregation
    workflow.add_edge("analyze_clarity", "aggregate_health")
    workflow.add_edge("analyze_completeness", "aggregate_health")
    workflow.add_edge("analyze_correctness", "aggregate_health")
    workflow.add_edge("analyze_courtesy", "aggregate_health")
    workflow.add_edge("analyze_audience", "aggregate_health")
    workflow.add_edge("analyze_timeliness", "aggregate_health")
    
    # Aggregation -> Explanation -> End
    workflow.add_edge("aggregate_health", "explain_health")
    workflow.add_edge("explain_health", END)
    
    return workflow.compile()
