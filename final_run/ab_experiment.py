"""
A/B Experiment: Compare LLM-only vs ML+LLM approaches
======================================================
Tests the hypothesis: ML-predicted categories improve LLM resolution quality
"""

import requests
import json
from test import main as classify_incident
from LLM_evaluate import RCAEvaluator

# Test incidents for experiment
TEST_INCIDENTS = [
    {
        "issue_name": "Database Connection Pool Exhausted",
        "issue_description": "Application throwing 'Cannot acquire connection from pool' errors. Started after deployment of v2.3.1. Affecting 30% of API requests. Connection count at max limit (100).",
        "priority": "P1 - Critical"
    },
    {
        "issue_name": "Memory Leak in Production",
        "issue_description": "Application crashed with OutOfMemoryError. Heap space exhausted after processing large batch job. JVM heap set to 4GB.",
        "priority": "P1 - Critical"
    },
    {
        "issue_name": "API Response Time Degradation",
        "issue_description": "API response time degraded from 200ms to 5 seconds. CPU usage at 95%. Slow database queries detected.",
        "priority": "P2 - High"
    }
]


def call_llm(prompt: str) -> str:
    """Call Ollama LLM API."""
    payload = {
        "model": "llama3.2:3b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "num_predict": 1024
        }
    }
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=300
        )
        response.raise_for_status()
        return response.json().get("response", "No response generated.")
    except Exception as e:
        return f"Error: {str(e)}"


def approach_a_llm_only(incident: dict) -> str:
    """Approach A: LLM without ML category prediction."""

    prompt = f"""You are an expert IT incident analyst specializing in root cause analysis and resolution.

## Incident Details
- **Issue Name**: {incident['issue_name']}
- **Description**: {incident['issue_description']}
- **Priority**: {incident['priority']}

## Your Task
Analyze this incident and provide a structured response with:

1. **Root Cause Analysis**: Identify the most likely root cause(s) of this incident.

2. **Immediate Actions**: List 2-3 immediate steps to mitigate or resolve the issue quickly.

3. **Detailed Solutions**: Provide 3-5 possible solutions ranked by likelihood of success.

4. **Prevention Recommendations**: Suggest measures to prevent this incident from recurring.

5. **Escalation Guidance**: If the suggested solutions don't work, specify when and to whom this should be escalated.

Be specific, actionable, and prioritize solutions based on the incident's priority level.
"""

    print("  Calling LLM (Approach A - No Category)...")
    return call_llm(prompt)


def approach_b_ml_plus_llm(incident: dict) -> tuple:
    """Approach B: LLM with ML category prediction."""

    # Get ML prediction
    _, predicted_category, confidence = classify_incident(incident['issue_description'])

    # Build confidence context
    confidence_pct = confidence * 100
    if confidence >= 0.85:
        confidence_text = f"HIGH ({confidence_pct:.1f}%)"
        guidance = f"The classifier is confident. Focus solutions on {predicted_category} issues."
    elif confidence >= 0.60:
        confidence_text = f"MEDIUM ({confidence_pct:.1f}%)"
        guidance = "Verify if this category fits. Consider related categories if needed."
    else:
        confidence_text = f"LOW ({confidence_pct:.1f}%)"
        guidance = "The classifier is uncertain. Analyze the description carefully."

    prompt = f"""You are an expert IT incident analyst specializing in root cause analysis and resolution.

## Incident Details
- **Issue Name**: {incident['issue_name']}
- **Description**: {incident['issue_description']}
- **Priority**: {incident['priority']}

## ML Classification Result
- **Predicted Category**: {predicted_category}
- **Confidence**: {confidence_text}
- {guidance}

## Your Task
Analyze this incident and provide a structured response with:

1. **Root Cause Analysis**: Identify the most likely root cause(s) of this incident. Consider common patterns for {predicted_category}.

2. **Immediate Actions**: List 2-3 immediate steps to mitigate or resolve the issue quickly.

3. **Detailed Solutions**: Provide 3-5 possible solutions ranked by likelihood of success.

4. **Prevention Recommendations**: Suggest measures to prevent this incident from recurring.

5. **Escalation Guidance**: If the suggested solutions don't work, specify when and to whom this should be escalated.

Be specific, actionable, and prioritize solutions based on the incident's priority level.
"""

    print(f"  ML Prediction: {predicted_category} ({confidence_pct:.1f}%)")
    print("  Calling LLM (Approach B - With Category)...")

    response = call_llm(prompt)
    return response, predicted_category, confidence


def run_experiment():
    """Run A/B experiment on all test incidents."""

    print("=" * 70)
    print("A/B EXPERIMENT: LLM-Only vs ML+LLM")
    print("=" * 70)

    evaluator = RCAEvaluator(judge_model="mistral:7b")
    results = []

    for i, incident in enumerate(TEST_INCIDENTS):
        print(f"\n{'=' * 70}")
        print(f"TEST CASE {i + 1}: {incident['issue_name']}")
        print(f"{'=' * 70}")

        # Build query string for evaluation
        incident_query = f"Issue: {incident['issue_name']}\nDescription: {incident['issue_description']}\nPriority: {incident['priority']}"

        # Approach A: LLM Only
        print("\n[APPROACH A: LLM Only]")
        response_a = approach_a_llm_only(incident)

        # Approach B: ML + LLM
        print("\n[APPROACH B: ML + LLM]")
        response_b, category, confidence = approach_b_ml_plus_llm(incident)

        # Evaluate both approaches
        print("\n[EVALUATING RESPONSES]")

        print("  Evaluating Approach A...")
        eval_a = evaluator.full_evaluation(incident_query, response_a)

        print("  Evaluating Approach B...")
        eval_b = evaluator.full_evaluation(incident_query, response_b)

        # Pairwise comparison
        print("  Running pairwise comparison...")
        comparison = evaluator.pairwise_comparison(
            incident_query=incident_query,
            response_a=response_a,
            response_b=response_b,
            model_a_name="LLM-Only",
            model_b_name="ML+LLM"
        )

        # Store results
        result = {
            "incident": incident,
            "ml_category": category,
            "ml_confidence": confidence,
            "approach_a": {
                "response": response_a[:500] + "..." if len(response_a) > 500 else response_a,
                "overall_score": eval_a["overall_score"],
                "specificity": eval_a["specificity"]["score"],
                "structure": eval_a["structure"]["completeness_score"]
            },
            "approach_b": {
                "response": response_b[:500] + "..." if len(response_b) > 500 else response_b,
                "overall_score": eval_b["overall_score"],
                "specificity": eval_b["specificity"]["score"],
                "structure": eval_b["structure"]["completeness_score"]
            },
            "comparison": comparison
        }
        results.append(result)

        # Print summary for this test case
        print(f"\n--- RESULTS FOR TEST CASE {i + 1} ---")
        print(
            f"Approach A (LLM-Only):  Overall={eval_a['overall_score']:.1f}, Structure={eval_a['structure']['completeness_score']:.0f}")
        print(
            f"Approach B (ML+LLM):    Overall={eval_b['overall_score']:.1f}, Structure={eval_b['structure']['completeness_score']:.0f}")
        if "winner" in comparison:
            print(f"Winner: {comparison.get('winner_model', comparison.get('winner', 'N/A'))}")
        print("-" * 40)

    # Final summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    a_wins = 0
    b_wins = 0
    ties = 0

    for i, r in enumerate(results):
        winner = r["comparison"].get("winner", "")
        if winner == "A":
            a_wins += 1
            winner_text = "LLM-Only"
        elif winner == "B":
            b_wins += 1
            winner_text = "ML+LLM"
        else:
            ties += 1
            winner_text = "TIE"

        print(f"Test {i + 1}: {r['incident']['issue_name'][:40]}")
        print(f"  A (LLM-Only): {r['approach_a']['overall_score']:.1f}/100")
        print(f"  B (ML+LLM):   {r['approach_b']['overall_score']:.1f}/100")
        print(f"  Winner: {winner_text}")
        print()

    print("-" * 40)
    print(f"FINAL TALLY:")
    print(f"  LLM-Only wins:  {a_wins}")
    print(f"  ML+LLM wins:    {b_wins}")
    print(f"  Ties:           {ties}")
    print("-" * 40)

    if b_wins > a_wins:
        print("CONCLUSION: ML+LLM approach produces better resolutions!")
    elif a_wins > b_wins:
        print("CONCLUSION: LLM-Only approach performs better (unexpected)")
    else:
        print("CONCLUSION: Both approaches perform similarly")

    # Save results
    with open("ab_experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to ab_experiment_results.json")

    return results


if __name__ == "__main__":
    run_experiment()