import requests
import json
import re
from typing import Dict, List, Tuple

# Import the RCA pipeline
from integrated import process_incident


class RCAEvaluator:
    """Evaluator for RCA system outputs."""

    def __init__(self, judge_model: str = "mistral:7b"):
        self.judge_model = judge_model
        self.base_url = "http://localhost:11434/api/generate"
        print(f"Evaluator initialized with judge model: {judge_model}")

    def _call_ollama(self, prompt: str, temperature: float = 0.1) -> str:
        """Make a call to Ollama API."""
        payload = {
            "model": self.judge_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 1024
            }
        }
        try:
            response = requests.post(self.base_url, json=payload, timeout=180)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            return f"Error: {str(e)}"

    # =========================================================================
    # METRIC 1: SPECIFICITY SCORE (Rule-based)
    # =========================================================================

    def calculate_specificity_score(self, text: str) -> Dict:
        """
        Calculate specificity score based on concrete details in the response.

        Higher score = more specific, actionable content
        Lower score = vague, generic content
        """

        # Indicators of SPECIFIC content (good)
        specific_indicators = {
            "numbers": len(re.findall(r'\b\d+\.?\d*\b', text)),

            "technical_terms": len(re.findall(
                r'\b(API|SQL|HTTP|TCP|UDP|DNS|CPU|RAM|GPU|disk|server|database|'
                r'cache|timeout|connection|port|endpoint|config|log|error|exception|'
                r'stack|trace|memory|thread|process|query|index|table|schema|'
                r'latency|throughput|bandwidth|SSL|TLS|SSH|FTP|REST|SOAP|JSON|XML|'
                r'timeout|retry|rollback|commit|transaction|deadlock|mutex|semaphore|'
                r'heap|stack|buffer|overflow|null|pointer|segfault|core dump)\b',
                text, re.IGNORECASE
            )),

            "action_verbs": len(re.findall(
                r'\b(run|execute|check|verify|restart|update|configure|set|increase|'
                r'decrease|add|remove|modify|edit|create|delete|install|deploy|'
                r'monitor|debug|trace|profile|analyze|inspect|validate|test|'
                r'rollback|revert|patch|upgrade|downgrade|backup|restore)\b',
                text, re.IGNORECASE
            )),

            "commands_code": len(re.findall(r'`[^`]+`|```[\s\S]*?```', text)),

            "file_paths": len(re.findall(
                r'[/\\][\w./\\-]+\.\w+|[A-Za-z]:\\[\w\\.-]+|/etc/|/var/|/opt/',
                text
            )),

            "time_references": len(re.findall(
                r'\b(\d+\s*(seconds?|minutes?|hours?|days?|ms|milliseconds?)|'
                r'\d+:\d+|\d+\s*(s|m|h|d))\b',
                text, re.IGNORECASE
            )),

            "specific_values": len(re.findall(
                r'\b(true|false|null|none|enabled?|disabled?|on|off|yes|no|'
                r'localhost|127\.0\.0\.1|0\.0\.0\.0|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b',
                text, re.IGNORECASE
            )),
        }

        # Indicators of VAGUE content (bad)
        vague_indicators = {
            "vague_words": len(re.findall(
                r'\b(maybe|perhaps|sometimes|might|could|possibly|generally|'
                r'usually|often|some|various|several|many|few|etc|things|stuff|'
                r'somehow|somewhere|something|anything|everything)\b',
                text, re.IGNORECASE
            )),

            "hedging_phrases": len(re.findall(
                r'\b(I think|I believe|it seems|it appears|probably|likely|'
                r'unlikely|may or may not|not sure|hard to say|depends on|'
                r'in some cases|it varies)\b',
                text, re.IGNORECASE
            )),

            "filler_phrases": len(re.findall(
                r'\b(as you know|of course|obviously|basically|essentially|'
                r'in general|for the most part|more or less)\b',
                text, re.IGNORECASE
            )),
        }

        # Calculate scores
        total_words = len(text.split())
        if total_words == 0:
            return {"score": 0, "interpretation": "Empty response"}

        specific_points = sum(specific_indicators.values())
        vague_points = sum(vague_indicators.values())

        # Normalized score (0-100)
        raw_score = (specific_points - vague_points * 0.5) / (total_words / 10)
        normalized_score = max(0, min(100, raw_score * 10))

        return {
            "score": round(normalized_score, 2),
            "specific_indicators": specific_indicators,
            "specific_total": specific_points,
            "vague_indicators": vague_indicators,
            "vague_total": vague_points,
            "total_words": total_words,
            "interpretation": self._interpret_specificity(normalized_score)
        }

    def _interpret_specificity(self, score: float) -> str:
        """Interpret specificity score."""
        if score >= 80:
            return "Excellent - Highly specific and actionable"
        elif score >= 60:
            return "Good - Mostly specific with some general statements"
        elif score >= 40:
            return "Fair - Mix of specific and vague content"
        elif score >= 20:
            return "Poor - Mostly vague, needs more concrete details"
        else:
            return "Very Poor - Too generic to be useful"

    # =========================================================================
    # METRIC 2: LLM-AS-JUDGE (Model-based)
    # =========================================================================

    def llm_judge_evaluation(self, incident_query: str, llm_response: str) -> Dict:
        """
        Use another LLM to evaluate the response quality across multiple dimensions.
        """

        judge_prompt = f"""You are an expert evaluator assessing the quality of an AI-generated incident analysis response.

## Original Incident
{incident_query}

## Response to Evaluate
{llm_response}

## Evaluation Task
Rate the response on each criterion from 1-10 (1=poor, 10=excellent).

Respond ONLY with this exact JSON format (no other text before or after):
{{
    "relevance": {{
        "score": <1-10>,
        "reason": "<one sentence explanation>"
    }},
    "accuracy": {{
        "score": <1-10>,
        "reason": "<one sentence explanation>"
    }},
    "completeness": {{
        "score": <1-10>,
        "reason": "<one sentence explanation>"
    }},
    "actionability": {{
        "score": <1-10>,
        "reason": "<one sentence explanation>"
    }},
    "clarity": {{
        "score": <1-10>,
        "reason": "<one sentence explanation>"
    }},
    "technical_depth": {{
        "score": <1-10>,
        "reason": "<one sentence explanation>"
    }},
    "overall": {{
        "score": <1-10>,
        "reason": "<one sentence overall assessment>"
    }},
    "improvements": "<2-3 specific suggestions to improve the response>"
}}
"""

        result = self._call_ollama(judge_prompt)

        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', result)
            if json_match:
                json_str = json_match.group()

                json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
                parsed = json.loads(json_str)
                parsed["_raw_response"] = result
                return parsed
        except json.JSONDecodeError as e:
            # Return partial result with error details
            return {
                "error": f"Failed to parse judge response: {str(e)}",
                "raw": result[:500]  # Truncate for readability
            }

        return {"error": "No JSON found in judge response", "raw": result[:500]}

    # =========================================================================
    # METRIC 3: STRUCTURE VALIDATION
    # =========================================================================

    def validate_structure(
            self,
            response: str,
            required_sections: List[str] = None
    ) -> Dict:
        """
        Check if response contains all required sections.
        """

        if required_sections is None:
            required_sections = [
                "Root Cause",
                "Immediate Actions",
                "Solution",
                "Prevention",
                "Escalation"
            ]

        found_sections = []
        missing_sections = []

        for section in required_sections:
            # Simple substring search (case-insensitive)
            if section.lower() in response.lower():
                found_sections.append(section)
            else:
                missing_sections.append(section)

        completeness = len(found_sections) / len(required_sections) * 100

        return {
            "completeness_score": round(completeness, 2),
            "found_sections": found_sections,
            "missing_sections": missing_sections,
            "total_required": len(required_sections),
            "total_found": len(found_sections),
            "interpretation": self._interpret_structure(completeness)
        }

    def _interpret_structure(self, score: float) -> str:
        """Interpret structure completeness score."""
        if score >= 100:
            return "Complete - All required sections present"
        elif score >= 80:
            return "Mostly Complete - Minor sections missing"
        elif score >= 60:
            return "Partial - Some important sections missing"
        else:
            return "Incomplete - Major sections missing"

    # =========================================================================
    # METRIC 4: PAIRWISE COMPARISON
    # =========================================================================

    def pairwise_comparison(
            self,
            incident_query: str,
            response_a: str,
            response_b: str,
            model_a_name: str = "Model A",
            model_b_name: str = "Model B"
    ) -> Dict:
        """
        Compare two responses head-to-head using LLM judge.
        """

        compare_prompt = f"""You are comparing two AI-generated incident analysis responses.

## Incident Query
{incident_query}

## Response A ({model_a_name})
{response_a[:2000]}

## Response B ({model_b_name})
{response_b[:2000]}

## Comparison Task
Determine which response is better for resolving the incident.

Respond ONLY with this exact JSON format:
{{
    "winner": "<A or B or TIE>",
    "confidence": "<HIGH or MEDIUM or LOW>",
    "comparison": {{
        "relevance": "<A or B or TIE> - <brief reason>",
        "specificity": "<A or B or TIE> - <brief reason>",
        "actionability": "<A or B or TIE> - <brief reason>",
        "completeness": "<A or B or TIE> - <brief reason>",
        "clarity": "<A or B or TIE> - <brief reason>"
    }},
    "summary": "<2-3 sentence overall comparison>"
}}
"""

        result = self._call_ollama(compare_prompt)

        try:
            json_match = re.search(r'\{[\s\S]*\}', result)
            if json_match:
                parsed = json.loads(json_match.group())
                # Add model names to result
                if parsed.get("winner") == "A":
                    parsed["winner_model"] = model_a_name
                elif parsed.get("winner") == "B":
                    parsed["winner_model"] = model_b_name
                else:
                    parsed["winner_model"] = "TIE"
                return parsed
        except json.JSONDecodeError:
            pass

        return {"error": "Failed to parse comparison", "raw": result}

    # =========================================================================
    # COMPREHENSIVE EVALUATION
    # =========================================================================

    def full_evaluation(
            self,
            incident_query: str,
            llm_response: str,
            required_sections: List[str] = None
    ) -> Dict:
        """
        Run all evaluation metrics and return comprehensive report.
        """

        print("\n" + "=" * 60)
        print("RUNNING FULL EVALUATION")
        print("=" * 60)

        # 1. Specificity Score
        print("\nCalculating Specificity Score...")
        specificity = self.calculate_specificity_score(llm_response)

        # 2. Structure Validation
        print("Validating Structure...")
        structure = self.validate_structure(llm_response, required_sections)

        # 3. LLM Judge Evaluation
        print("Running LLM Judge Evaluation...")
        llm_judge = self.llm_judge_evaluation(incident_query, llm_response)

        # Calculate overall score
        scores = [specificity["score"], structure["completeness_score"]]

        if "error" not in llm_judge and "overall" in llm_judge:
            scores.append(llm_judge["overall"]["score"] * 10)

        overall_score = sum(scores) / len(scores)

        return {
            "overall_score": round(overall_score, 2),
            "overall_interpretation": self._interpret_overall(overall_score),
            "specificity": specificity,
            "structure": structure,
            "llm_judge": llm_judge
        }

    def _interpret_overall(self, score: float) -> str:
        """Interpret overall evaluation score."""
        if score >= 80:
            return "Excellent - High quality, production-ready response"
        elif score >= 60:
            return "Good - Useful response with minor improvements needed"
        elif score >= 40:
            return "Fair - Acceptable but needs more detail"
        elif score >= 20:
            return "Poor - Significant improvements required"
        else:
            return "Very Poor - Response does not meet requirements"

    # =========================================================================
    # HELPER: PRINT EVALUATION REPORT
    # =========================================================================

    def print_evaluation_report(self, evaluation: Dict):
        """Pretty print the evaluation results."""

        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)

        # Overall Score
        print(f"\nOVERALL SCORE: {evaluation['overall_score']}/100")
        print(f"   {evaluation['overall_interpretation']}")

        # Specificity
        spec = evaluation["specificity"]
        print(f"\nSPECIFICITY SCORE: {spec['score']}/100")
        print(f"   {spec['interpretation']}")
        print(f"   Specific indicators: {spec['specific_total']} | Vague indicators: {spec['vague_total']}")

        # Structure
        struct = evaluation["structure"]
        print(f"\nSTRUCTURE SCORE: {struct['completeness_score']}/100")
        print(f"   {struct['interpretation']}")
        print(f"   Found: {struct['found_sections']}")
        if struct['missing_sections']:
            print(f"   Missing: {struct['missing_sections']}")

        # LLM Judge
        judge = evaluation["llm_judge"]
        if "error" not in judge:
            print("\nLLM JUDGE SCORES:")
            for metric in ["relevance", "accuracy", "completeness", "actionability", "clarity", "technical_depth",
                           "overall"]:
                if metric in judge and isinstance(judge[metric], dict):
                    score = judge[metric].get("score", "N/A")
                    reason = judge[metric].get("reason", "")
                    print(f"   {metric.capitalize()}: {score}/10 - {reason}")

            if "improvements" in judge:
                print(f"\nSUGGESTED IMPROVEMENTS:")
                print(f"   {judge['improvements']}")
        else:
            print(f"\nLLM JUDGE: {judge.get('error', 'Unknown error')}")

        print("\n" + "=" * 60)


# =============================================================================
# MAIN: RUN EVALUATION
# =============================================================================

def evaluate_incident(
        issue_name: str,
        issue_description: str,
        priority: str,
        judge_model: str = "mistral:7b"
) -> Dict:
    """
    Run the full pipeline and evaluate the results.

    Args:
        issue_name: Name of the incident
        issue_description: Description of the incident
        priority: Priority level
        judge_model: Model to use for LLM-as-Judge

    Returns:
        Dictionary with pipeline results and evaluation
    """

    # Run RCA pipeline
    print("Running RCA Pipeline...")
    pipeline_result = process_incident(
        issue_name=issue_name,
        issue_description=issue_description,
        priority=priority
    )

    # Initialize evaluator
    evaluator = RCAEvaluator(judge_model=judge_model)

    # Build incident query string for evaluation
    incident_query = f"""Issue: {issue_name}
Description: {issue_description}
Category: {pipeline_result['predicted_category']}
Priority: {priority}"""

    # Run full evaluation
    evaluation = evaluator.full_evaluation(
        incident_query=incident_query,
        llm_response=pipeline_result['llm_analysis']
    )

    # Print report
    evaluator.print_evaluation_report(evaluation)

    # Combine results
    return {
        "pipeline_result": pipeline_result,
        "evaluation": evaluation
    }


if __name__ == "__main__":
    # Sample incident
    incident = {
        "issue_name": "Database Connection Pool Exhausted",
        "issue_description": "Application throwing 'Cannot acquire connection from pool' errors. Started after deployment of v2.3.1. Affecting 30% of API requests. Connection count at max limit (100). No recent DB config changes.",
        "priority": "P1 - Critical"
    }

    # Run evaluation
    result = evaluate_incident(**incident)

    # Save results
    with open("evaluation_result.json", "w") as f:
        # Remove raw response before saving
        if "_raw_response" in result["evaluation"]["llm_judge"]:
            del result["evaluation"]["llm_judge"]["_raw_response"]
        json.dump(result, f, indent=2)

    print("\nResults saved to evaluation_result.json")