"""Core communication health analysis logic.

This module provides a class to analyze text-based communication across six
key dimensions: clarity, completeness, correctness, courtesy, audience-centricity,
and timeliness. It leverages a large language model (LLM) with explicit
guardrails to ensure objective, evidence-based analysis and reduce hallucinations.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# Maximum context length to prevent prompt bloat
MAX_CONTEXT_LENGTH = 2000

# --- Prompt Configuration with Guardrails ---
# Each prompt includes explicit rules to ensure objective, impartial, and evidence-based analysis

PROMPT_CONFIG = {
    "clarity": {
        "score_key": "clarity_score",
        "template": """
        You are an impartial and objective communication analyst. Your task is to evaluate the clarity and conciseness of the following content on a scale of 0.0 to 1.0.

        **Analysis Rules & Guardrails:**
        - **Objective Analysis:** Base your evaluation **solely** on the text provided. Do not invent, assume, or infer any information not explicitly stated.
        - **No External Context:** Do not assume this communication is part of a larger, unseen conversation. Analyze only what is given.
        - **Impartial Judgment:** Do not be optimistic or biased. Judge the communication based on its effectiveness, not the presumed intent or individuals involved.
        - **Evidence-Based Reasoning:** Your 'reasoning' must reference specific phrases or aspects of the provided text to justify your score.

        **Evaluation Criteria:**
        - **Readability:** Is the language simple and easy to understand?
        - **Directness:** Is the main point stated clearly and early?
        - **Jargon:** Is technical language used appropriately for the context?
        - **Brevity:** Is the message concise or overly wordy?

        **Examples:**
        - **Content:** "I wanted to reach out to you regarding the possibility of potentially scheduling a meeting at some point in the near future to discuss various topics."
          **Analysis:** {{"score": 0.2, "reasoning": "The phrasing 'wanted to reach out regarding the possibility of potentially' is extremely wordy and indirect, obscuring the purpose."}}
        - **Content:** "Let's sync on the project deliverables."
          **Analysis:** {{"score": 0.6, "reasoning": "Direct but uses jargon ('sync', 'deliverables') and lacks specific details. It's unclear what needs to be discussed or when."}}
        - **Content:** "We need to discuss the Q4 budget."
          **Analysis:** {{"score": 0.7, "reasoning": "Direct and clear about the topic, but lacks a specific call to action or proposed time, making it incomplete."}}
        - **Content:** "Can we meet for 30 minutes tomorrow to finalize the Q4 budget before the 5 PM deadline? Please check the attached draft."
          **Analysis:** {{"score": 1.0, "reasoning": "Perfectly clear, direct, and concise. It states the purpose, duration, timing, and necessary action, leaving no room for ambiguity."}}

        **Content to Analyze:**
        {content}
        {context}

        **Respond ONLY with a valid JSON object with "score" and "reasoning" keys.**
        """
    },
    "completeness": {
        "score_key": "completeness_score",
        "template": """
        You are an impartial and objective communication analyst. Evaluate the completeness of the following communication on a scale of 0.0 to 1.0.

        **Analysis Rules & Guardrails:**
        - **Objective Analysis:** Base your evaluation **solely** on the text provided. Do not invent, assume, or infer any information not explicitly stated.
        - **No External Context:** Do not assume this communication is part of a larger, unseen conversation. Analyze only what is given.
        - **Impartial Judgment:** Do not be optimistic or biased. Judge the communication based on its effectiveness.
        - **Evidence-Based Reasoning:** Your 'reasoning' must reference specific phrases or aspects of the provided text to justify your score.

        **Evaluation Criteria:**
        - **Sufficient Information:** Does it provide all necessary details (who, what, when, where, why) for the recipient to understand and act?
        - **Actionable Details:** Are the next steps, if any, clearly defined?
        - **Anticipates Questions:** Does it proactively answer likely follow-up questions?

        **Examples:**
        - **Content:** "The report is delayed."
          **Analysis:** {{"score": 0.1, "reasoning": "Critically incomplete. The statement provides no reason, no new timeline, and no information on the impact."}}
        - **Content:** "I've attached the presentation for the meeting."
          **Analysis:** {{"score": 0.5, "reasoning": "Partially complete. It provides the attachment but omits key details like when the meeting is or what feedback is expected."}}
        - **Content:** "The Q3 report is delayed due to unexpected data validation issues. I'm working on it now and will send it by Friday EOD. The main dashboard will not be affected."
          **Analysis:** {{"score": 0.9, "reasoning": "Very complete. It explains the 'what' (delay), 'why' (data issues), 'when' (Friday EOD), and manages expectations by mentioning impact."}}
        - **Content:** "Team, the project launch is moved to next Tuesday, Oct 17th. This is to allow for final QA on the new payment module. All other deadlines remain the same. The updated project plan is attached. No action is needed from you at this time."
          **Analysis:** {{"score": 1.0, "reasoning": "Fully complete. It provides all necessary information: the change, the new date, the reason, the impact on other tasks, and explicitly states the required action (none)."}}

        **Content to Analyze:**
        {content}
        {context}

        **Respond ONLY with a valid JSON object with "score" and "reasoning" keys.**
        """
    },
    "correctness": {
        "score_key": "correctness_score",
        "template": """
        You are an impartial and objective communication analyst. Evaluate the correctness and coherence of this communication on a scale of 0.0 to 1.0.

        **Analysis Rules & Guardrails:**
        - **Objective Analysis:** Base your evaluation **solely** on the text provided. Do not invent, assume, or infer any information not explicitly stated.
        - **No External Context:** Do not assume this communication is part of a larger, unseen conversation. Analyze only what is given.
        - **Impartial Judgment:** Do not be optimistic or biased.
        - **Evidence-Based Reasoning:** Your 'reasoning' must reference specific phrases or aspects of the provided text to justify your score.

        **Evaluation Criteria:**
        - **Factual Accuracy:** Does the text contain self-contradictions? (You cannot verify external facts).
        - **Grammar & Spelling:** Is the text free of grammatical errors and typos?
        - **Logical Flow:** Do the ideas connect logically and make sense in sequence?

        **Examples:**
        - **Content:** "Their is three issue's with the deploy, i will fix it latter."
          **Analysis:** {{"score": 0.2, "reasoning": "Contains multiple grammatical and spelling errors, specifically 'Their' instead of 'There', 'issue's' instead of 'issues', and 'latter' instead of 'later'."}}
        - **Content:** "The meeting is at 2 PM. The new time is 4 PM. See you at 2 PM."
          **Analysis:** {{"score": 0.3, "reasoning": "The text contains a logical contradiction, stating the time is both 2 PM and 4 PM, creating confusion."}}
        - **Content:** "Hi team, the Q3 report shows 15% growth. We exceeded our targets."
          **Analysis:** {{"score": 0.9, "reasoning": "Clear, grammatically correct, and logically coherent message."}}

        **Content to Analyze:**
        {content}
        {context}

        **Respond ONLY with a valid JSON object with "score" and "reasoning" keys.**
        """
    },
    "courtesy": {
        "score_key": "courtesy_score",
        "template": """
        You are an impartial and objective communication analyst. Evaluate the courtesy and professionalism of the tone on a scale of 0.0 to 1.0.

        **Analysis Rules & Guardrails:**
        - **Objective Analysis:** Base your evaluation **solely** on the text provided. Do not invent, assume, or infer any information not explicitly stated.
        - **No External Context:** Do not assume this communication is part of a larger, unseen conversation.
        - **Impartial Judgment:** Do not be optimistic or biased. Analyze the tone of the words used, not the presumed intent.
        - **Evidence-Based Reasoning:** Your 'reasoning' must reference specific phrases or aspects of the provided text to justify your score.

        **Evaluation Criteria:**
        - **Politeness:** Does it use polite language (e.g., "please", "thank you")?
        - **Respect:** Is the tone respectful or demanding and dismissive?
        - **Professionalism:** Does the language maintain a professional standard?

        **Examples:**
        - **Content:** "Why wasn't this done yesterday? This is unacceptable."
          **Analysis:** {{"score": 0.1, "reasoning": "The tone is accusatory ('Why wasn't this done') and unprofessional ('unacceptable'), which is discourteous and likely to create conflict."}}
        - **Content:** "Do this now."
          **Analysis:** {{"score": 0.3, "reasoning": "Overly demanding and lacks any form of politeness. Comes across as a command rather than a request."}}
        - **Content:** "Just wanted to follow up on the report."
          **Analysis:** {{"score": 0.7, "reasoning": "Neutral and professional, but could be softened. It's direct but not explicitly impolite."}}
        - **Content:** "Hi Jane, I hope you're having a good week. Could you please review the attached document when you have a moment? No rush, EOD Friday is fine. Thank you for your help!"
          **Analysis:** {{"score": 1.0, "reasoning": "Extremely courteous. It uses polite phrases like 'could you please' and 'Thank you', and is respectful of the recipient's time by saying 'when you have a moment'."}}

        **Content to Analyze:**
        {content}
        {context}

        **Respond ONLY with a valid JSON object with "score" and "reasoning" keys.**
        """
    },
    "audience": {
        "score_key": "audience_score",
        "template": """
        You are an impartial and objective communication analyst. Evaluate how well this communication is tailored to its intended audience on a scale of 0.0 to 1.0.

        **Analysis Rules & Guardrails:**
        - **Objective Analysis:** Base your evaluation **solely** on the text provided. Use the 'Additional Context' if it specifies the audience.
        - **No External Context:** Do not assume this communication is part of a larger, unseen conversation.
        - **Impartial Judgment:** Do not be optimistic or biased.
        - **Evidence-Based Reasoning:** Your 'reasoning' must reference specific phrases or aspects of the provided text to justify your score.

        **Evaluation Criteria:**
        - **Relevance:** Is the content relevant to the recipient?
        - **Appropriate Language:** Is the level of technicality and jargon appropriate for the likely audience?
        - **Personalization:** Is the message personalized or generic?

        **Examples:**
        - **Content (Context: Audience is a non-technical client):** "We need to refactor the React components and optimize the API middleware to reduce latency."
          **Analysis:** {{"score": 0.2, "reasoning": "Poorly tailored. The text uses technical jargon like 'refactor React components' and 'API middleware' which is inappropriate for a non-technical client."}}
        - **Content:** "Dear employee, please see attached."
          **Analysis:** {{"score": 0.4, "reasoning": "Lacks personalization and context. It is generic and feels impersonal, reducing its impact and engagement."}}
        - **Content:** "Hi Sarah, based on our conversation about streamlining invoicing, here is the draft of the new workflow."
          **Analysis:** {{"score": 0.95, "reasoning": "Highly audience-centric. It is personalized ('Hi Sarah') and directly references a previous, specific conversation ('our conversation about streamlining invoicing')."}}

        **Content to Analyze:**
        {content}
        {context}

        **Respond ONLY with a valid JSON object with "score" and "reasoning" keys.**
        """
    },
    "timeliness": {
        "score_key": "timeliness_score",
        "template": """
        You are an impartial and objective communication analyst. Evaluate the timeliness of this communication on a scale of 0.0 to 1.0 using the provided context.

        **Analysis Rules & Guardrails:**
        - **Objective Analysis:** Base your evaluation **solely** on the metadata provided in the 'Additional Context' (e.g., sent_at, response_time_hours).
        - **No External Context:** Do not invent reasons for delays or make assumptions beyond the data given.
        - **Impartial Judgment:** Do not be optimistic or biased. Evaluate based on the timing data.
        - **Evidence-Based Reasoning:** Your 'reasoning' must reference specific data points from the context to justify your score.

        **Evaluation Criteria:**
        - **Appropriate Timing:** Was the message sent at a reasonable time?
        - **Response Speed:** Was the response time within expected limits?
        - **Urgency Handling:** Does the timing match the expressed urgency of the content?

        **Examples:**
        - **Context:** {"sent_at": "Friday 11:00 PM"}; **Content:** "URGENT: Need the sales numbers now!"
          **Analysis:** {{"score": 0.2, "reasoning": "Poor timing. The context shows an 'URGENT' request was sent at 'Friday 11:00 PM', which is outside of standard business hours and disrespectful of personal time."}}
        - **Context:** {"expected_response_time_hours": 8, "response_time_hours": 48}
          **Analysis:** {{"score": 0.4, "reasoning": "Untimely response. The 'response_time_hours' of 48 significantly exceeded the 'expected_response_time_hours' of 8."}}
        - **Context:** {"expected_response_time_hours": 24, "response_time_hours": 4}; **Content:** "Got it, thanks! I'll review this afternoon."
          **Analysis:** {{"score": 0.95, "reasoning": "Excellent timeliness. The response was sent well within the expected timeframe, demonstrating efficiency and respect for the sender's time."}}

        **Content to Analyze:**
        {content}
        {context}

        **Respond ONLY with a valid JSON object with "score" and "reasoning" keys.**
        """
    }
}


class CommunicationHealthAnalyzer:
    """Analyzes communication health across six dimensions using a Language Model.

    This class provides methods to evaluate a piece of text (like an email or
    a meeting transcript) on clarity, completeness, correctness, courtesy,
    audience-centricity, and timeliness.
    """

    def __init__(self, llm: BaseChatModel, context_max_length: int = MAX_CONTEXT_LENGTH):
        """
        Initializes the analyzer with an LLM.

        Args:
            llm: An instance of a LangChain-compatible language model.
            context_max_length: The maximum character length for context strings.
        """
        self.llm = llm
        self.context_max_length = context_max_length

    def _validate_content(self, content: str) -> Optional[Dict[str, Any]]:
        """Validates that the input content is a non-empty string."""
        if not isinstance(content, str) or not content.strip():
            logger.warning("Validation failed: Content was empty or not a string.")
            return {"score": 0.0, "reasoning": "Input content cannot be empty."}
        return None

    def _format_context(self, context: Optional[Dict[str, Any]]) -> str:
        """Formats the context dictionary into a string for the prompt, with truncation."""
        if not context:
            return ""
        
        # Pretty print the JSON for better readability by the LLM
        context_str = json.dumps(context, indent=2)

        if len(context_str) > self.context_max_length:
            context_str = context_str[:self.context_max_length] + "\n... [truncated]"
            logger.warning(f"Context truncated to {self.context_max_length} characters")

        return f"\n**Additional Context:**\n```json\n{context_str}\n```"

    def _parse_analysis_result(self, result: str, dimension_name: str) -> Dict[str, Any]:
        """
        Parses and validates the JSON output from the LLM.

        Args:
            result: The raw string output from the LLM.
            dimension_name: The name of the dimension being analyzed for logging.

        Returns:
            A dictionary with "score" and "reasoning" keys.
        """
        try:
            # Clean up potential markdown code fences
            if result.strip().startswith("```json"):
                result = result.strip()[7:-3]

            data = json.loads(result)
            
            if "score" not in data or "reasoning" not in data:
                raise KeyError("The JSON response is missing 'score' or 'reasoning'.")

            # Ensure score is a float
            score = float(data["score"])
            reasoning = str(data["reasoning"])

            return {"score": score, "reasoning": reasoning}
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse JSON for '{dimension_name}'. Raw result: '{result[:200]}...'"
            logger.error(f"{error_msg} | Error: {e}")
            return {"score": 0.0, "reasoning": "Analysis failed: Invalid JSON response from model."}
        except (KeyError, ValueError, TypeError) as e:
            error_msg = f"Invalid data format for '{dimension_name}'. Raw result: '{result[:200]}...'"
            logger.error(f"{error_msg} | Error: {e}")
            return {"score": 0.0, "reasoning": f"Analysis failed: {e}."}

    def _run_analysis(
        self,
        dimension: str,
        content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generic method to run analysis for a specific dimension.

        Args:
            dimension: The name of the dimension to analyze (e.g., 'clarity').
            content: The text content to analyze.
            context: An optional dictionary providing additional context.

        Returns:
            A dictionary containing the analysis score and reasoning.
        """
        if (validation_error := self._validate_content(content)):
            return validation_error

        config = PROMPT_CONFIG.get(dimension)
        if not config:
            raise ValueError(f"Invalid analysis dimension: {dimension}")

        prompt = ChatPromptTemplate.from_template(config["template"])
        context_str = self._format_context(context)

        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"content": content, "context": context_str})

        return self._parse_analysis_result(result, dimension)
    
    # --- Public Analysis Methods ---

    def analyze_clarity(self, content: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyzes clarity and conciseness."""
        return self._run_analysis("clarity", content, context)

    def analyze_completeness(self, content: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyzes the completeness of information."""
        return self._run_analysis("completeness", content, context)

    def analyze_correctness(self, content: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyzes correctness and logical coherence."""
        return self._run_analysis("correctness", content, context)

    def analyze_courtesy(self, content: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyzes the courtesy and tone."""
        return self._run_analysis("courtesy", content, context)

    def analyze_audience(self, content: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyzes how well the content is tailored to its audience."""
        return self._run_analysis("audience", content, context)

    def analyze_timeliness(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        sent_at: Optional[datetime] = None,
        received_at: Optional[datetime] = None,
        expected_response_time: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Analyzes timeliness and responsiveness based on timing metadata.
        """
        enhanced_context = context.copy() if context else {}
        if sent_at:
            enhanced_context["sent_at"] = sent_at.isoformat()
        if received_at:
            enhanced_context["received_at"] = received_at.isoformat()
        if sent_at and received_at:
            response_time = received_at - sent_at
            enhanced_context["response_time_hours"] = round(response_time.total_seconds() / 3600, 2)
        if expected_response_time:
            enhanced_context["expected_response_time_hours"] = round(expected_response_time.total_seconds() / 3600, 2)
            if "response_time_hours" in enhanced_context:
                enhanced_context["was_response_timely"] = enhanced_context["response_time_hours"] <= enhanced_context["expected_response_time_hours"]
        
        return self._run_analysis("timeliness", content, enhanced_context)

    def aggregate_scores(
        self, 
        health_scores: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate communication health scores across all dimensions.
        
        Uses standardized format: {"score": float, "reasoning": str}
        
        Args:
            health_scores: Dictionary with dimension names as keys and analysis results as values
            
        Returns:
            Aggregated health scores with overall score and per-dimension averages
        """
        dimensions = ['clarity', 'completeness', 'correctness', 'courtesy', 'audience', 'timeliness']
        
        dimension_scores = {}
        
        for dimension in dimensions:
            dim_data = health_scores.get(dimension, {})
            if isinstance(dim_data, dict):
                # Use standardized "score" key
                score = dim_data.get("score", 0.5)
                dimension_scores[dimension] = float(score)
            else:
                dimension_scores[dimension] = 0.5
                logger.warning(f"Invalid data format for dimension {dimension}: {type(dim_data)}")
        
        # Calculate overall health score (unweighted average)
        overall_health = sum(dimension_scores.values()) / len(dimension_scores) if dimension_scores else 0.5
        
        aggregated = {
            "overall": overall_health,
            "dimensions": dimension_scores,
            "total_dimensions": len(dimensions)
        }
        
        return aggregated
    
    def explain_results(
        self, 
        aggregated_health: Dict[str, Any],
        health_scores: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Generate natural language explanation of communication health results.
        
        Args:
            aggregated_health: Aggregated health scores
            health_scores: Individual dimension scores with reasoning
            
        Returns:
            Natural language explanation
        """
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
        
        # Extract reasoning from each dimension (using standardized format)
        dimension_findings = {}
        for dim in ['clarity', 'completeness', 'correctness', 'courtesy', 'audience', 'timeliness']:
            dim_data = health_scores.get(dim, {})
            if isinstance(dim_data, dict):
                dimension_findings[dim] = dim_data.get('reasoning', 'No findings')
            else:
                dimension_findings[dim] = 'No data available'
        
        chain = prompt | self.llm | StrOutputParser()
        explanation = chain.invoke({
            "overall_score": aggregated_health.get('overall', 0.5),
            "dimensions": aggregated_health.get('dimensions', {}),
            "clarity_findings": dimension_findings.get('clarity', ''),
            "completeness_findings": dimension_findings.get('completeness', ''),
            "correctness_findings": dimension_findings.get('correctness', ''),
            "courtesy_findings": dimension_findings.get('courtesy', ''),
            "audience_findings": dimension_findings.get('audience', ''),
            "timeliness_findings": dimension_findings.get('timeliness', '')
        })
        
        return explanation
