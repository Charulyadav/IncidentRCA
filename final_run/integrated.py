"""
Incident RCA Module - Integrated with ML Classifier
====================================================
Uses test.py for category prediction, then generates LLM-based solutions.
"""

import requests
import json

# Import the ML classifier from the other team's file
from test import main as classify_incident


def analyze_incident(
        issue_name: str,
        issue_description: str,
        issue_category: str,
        priority: str,
        ml_confidence: float = None
) -> str:
    """
    Analyze an IT incident and generate possible solutions using llama3.2:3b.

    Args:
        issue_name: Name/title of the incident
        issue_description: Detailed description
        issue_category: Category (from ML classifier)
        priority: Priority level
        ml_confidence: Confidence score from ML classifier (optional)
    """

    # Build confidence context if ML confidence is provided
    confidence_section = ""
    if ml_confidence is not None:
        confidence_pct = ml_confidence * 100
        if ml_confidence >= 0.85:
            confidence_section = f"""
## ML Classification Info
- **Predicted Category**: {issue_category}
- **Confidence**: {confidence_pct:.1f}% (HIGH)
- The classifier is confident about this category. Focus solutions on {issue_category} issues.
"""
        elif ml_confidence >= 0.60:
            confidence_section = f"""
## ML Classification Info
- **Predicted Category**: {issue_category}
- **Confidence**: {confidence_pct:.1f}% (MEDIUM)
- Verify if this category fits the description. Consider related categories if needed.
"""
        else:
            confidence_section = f"""
## ML Classification Info
- **Predicted Category**: {issue_category}
- **Confidence**: {confidence_pct:.1f}% (LOW)
- The classifier is uncertain. Analyze the description carefully to confirm the correct category.
"""

    prompt = f"""You are an expert IT incident analyst specializing in root cause analysis and resolution.

## Incident Details
- **Issue Name**: {issue_name}
- **Description**: {issue_description}
- **Category**: {issue_category}
- **Priority**: {priority}
{confidence_section}

## Your Task
Analyze this incident and provide a structured response with:

1. **Root Cause Analysis**: Identify the most likely root cause(s) of this incident. Consider common patterns and underlying issues.

2. **Immediate Actions**: List 2-3 immediate steps to mitigate or resolve the issue quickly.

3. **Detailed Solutions**: Provide 3-5 possible solutions ranked by likelihood of success. For each solution:
   - Describe the fix
   - Explain why it addresses the root cause
   - Note any prerequisites or dependencies

4. **Prevention Recommendations**: Suggest measures to prevent this incident from recurring.

5. **Escalation Guidance**: If the suggested solutions don't work, specify when and to whom this should be escalated.

Be specific, actionable, and prioritize solutions based on the incident's priority level ({priority}).
"""

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

    print("Calling LLM API...")

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=1200
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", "No response generated.")

    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Ollama. Make sure it's running (ollama serve)."
    except requests.exceptions.Timeout:
        return "Error: Request timed out. The model may be loading or overloaded."
    except Exception as e:
        return f"Error: {str(e)}"


def process_incident(issue_name: str, issue_description: str, priority: str) -> dict:
    """
    Full pipeline: ML Classification → LLM Analysis

    Args:
        issue_name: Name/title of the incident
        issue_description: Detailed description
        priority: Priority level (P1-P4)

    Returns:
        Dictionary with classification and analysis results
    """
    print("=" * 60)
    print("INCIDENT RCA PIPELINE")
    print("=" * 60)
    print(f"Issue: {issue_name}")
    print(f"Priority: {priority}")
    print("-" * 60)

    # STAGE 1: ML Classification (using other team's code)
    print("\nSTAGE 1: ML Classification")
    print("-" * 40)

    _, predicted_category, confidence = classify_incident(issue_description)

    print(f"Predicted Category: {predicted_category}")
    print(f"Confidence: {confidence * 100:.2f}%")

    if confidence >= 0.85:
        print("Confidence Level: HIGH")
    elif confidence >= 0.60:
        print("Confidence Level: MEDIUM")
    else:
        print("Confidence Level: LOW")

    # STAGE 2: LLM Analysis
    print("\nSTAGE 2: LLM Analysis")
    print("-" * 40)

    analysis = analyze_incident(
        issue_name=issue_name,
        issue_description=issue_description,
        issue_category=predicted_category,
        priority=priority,
        ml_confidence=confidence
    )

    return {
        "issue_name": issue_name,
        "issue_description": issue_description,
        "priority": priority,
        "predicted_category": predicted_category,
        "ml_confidence": confidence,
        "llm_analysis": analysis
    }


# Example usage
if __name__ == "__main__":
    # Sample incident
    incident = {
        "issue_name": "Database Connection Pool Exhausted",
        "issue_description": "Application throwing 'Cannot acquire connection from pool' errors. Started after deployment of v2.3.1. Affecting 30% of API requests. Connection count at max limit (100). No recent DB config changes.",
        "priority": "P1 - Critical"
    }

    # Run full pipeline
    result = process_incident(
        issue_name=incident["issue_name"],
        issue_description=incident["issue_description"],
        priority=incident["priority"]
    )

    # Print results
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"\nIssue: {result['issue_name']}")
    print(f"Category: {result['predicted_category']} ({result['ml_confidence'] * 100:.1f}% confidence)")
    print(f"Priority: {result['priority']}")
    print("\nLLM Analysis:")
    print("-" * 60)
    print(result['llm_analysis'])

    # Optionally save to JSON
    with open("rca_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nResult saved to rca_result.json")