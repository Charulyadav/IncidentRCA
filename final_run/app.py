"""
Streamlit UI for Incident RCA System
=====================================
Two-stage pipeline: ML Classification + LLM Resolution
"""

import streamlit as st
import json
from integrated import process_incident
from LLM_evaluate import RCAEvaluator

# Page configuration
st.set_page_config(
    page_title="Incident RCA System",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if "result" not in st.session_state:
    st.session_state.result = None
if "evaluation" not in st.session_state:
    st.session_state.evaluation = None

# Title and description
st.title("Incident Root Cause Analysis System")
st.markdown("Automated incident diagnosis using **ML classification** + **LLM reasoning**")
st.markdown("---")

# Sidebar for inputs
st.sidebar.header("Incident Details")

issue_name = st.sidebar.text_input(
    "Issue Name",
    placeholder="e.g., Database Connection Pool Exhausted",
    help="A short title describing the incident"
)

issue_description = st.sidebar.text_area(
    "Issue Description",
    placeholder="Describe the incident in detail...",
    height=200,
    help="Provide as much detail as possible for better analysis"
)

priority = st.sidebar.selectbox(
    "Priority Level",
    ["P1 - Critical", "P2 - High", "P3 - Medium", "P4 - Low"],
    help="Select the severity of the incident"
)

# Analyze button
analyze_button = st.sidebar.button("Analyze Incident", type="primary", use_container_width=True)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### How it works")
st.sidebar.markdown("""
1. **ML Classifier** predicts the incident category
2. **LLM** generates detailed resolution steps
3. **Evaluator** scores the response quality
""")

# Main content area
if analyze_button:
    if not issue_name or not issue_description:
        st.error("Please fill in both **Issue Name** and **Issue Description**")
    else:
        # Show loading spinner
        with st.spinner("Analyzing incident... This may take 30-60 seconds."):
            try:
                # Run the pipeline
                result = process_incident(
                    issue_name=issue_name,
                    issue_description=issue_description,
                    priority=priority
                )
                st.session_state.result = result

                # Run evaluation
                evaluator = RCAEvaluator(judge_model="llama3.2:3b")
                incident_query = f"Issue: {issue_name}\nDescription: {issue_description}\nPriority: {priority}"
                evaluation = evaluator.full_evaluation(incident_query, result["llm_analysis"])
                st.session_state.evaluation = evaluation

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Make sure Ollama is running: `ollama serve`")

# Display results if available
if st.session_state.result:
    result = st.session_state.result
    evaluation = st.session_state.evaluation

    # Create tabs
    tab1, tab2 = st.tabs(["Analysis Results", "Evaluation Metrics"])

    # TAB 1: Analysis Results
    with tab1:
        st.success("Analysis complete!")

        # Display metrics in columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Predicted Category",
                value=result["predicted_category"]
            )

        with col2:
            confidence = result["ml_confidence"] * 100
            st.metric(
                label="Confidence",
                value=f"{confidence:.1f}%"
            )

        with col3:
            if confidence >= 85:
                level = "HIGH"
            elif confidence >= 60:
                level = "MEDIUM"
            else:
                level = "LOW"
            st.metric(
                label="Confidence Level",
                value=level
            )

        # Warning for low confidence
        if confidence < 60:
            st.warning("Low confidence prediction. Please verify the category manually.")

        st.markdown("---")

        # Incident summary
        st.subheader("Incident Summary")
        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            st.markdown(f"**Issue:** {result['issue_name']}")
            st.markdown(f"**Priority:** {result['priority']}")
        with summary_col2:
            st.markdown(f"**Category:** {result['predicted_category']}")
            st.markdown(f"**Confidence:** {confidence:.1f}%")

        st.markdown("---")

        # LLM Analysis
        st.subheader("AI-Generated Resolution")

        if result["llm_analysis"].startswith("Error:"):
            st.error(result["llm_analysis"])
            st.info("Make sure Ollama is running: `ollama serve`")
        else:
            st.markdown(result["llm_analysis"])

        # Download button
        st.markdown("---")
        st.download_button(
            label="Download Results as JSON",
            data=json.dumps(result, indent=2),
            file_name=f"rca_result_{result['issue_name'].replace(' ', '_')[:30]}.json",
            mime="application/json"
        )

    # TAB 2: Evaluation Metrics
    with tab2:
        if evaluation:
            st.subheader("Response Quality Evaluation")

            # Overall score with color coding
            overall_score = evaluation["overall_score"]
            if overall_score >= 80:
                score_color = "green"
                score_label = "Excellent"
            elif overall_score >= 60:
                score_color = "orange"
                score_label = "Good"
            elif overall_score >= 40:
                score_color = "orange"
                score_label = "Fair"
            else:
                score_color = "red"
                score_label = "Needs Improvement"

            st.markdown(
                f"### Overall Score: <span style='color:{score_color};font-size:2em;'>{overall_score:.1f}/100</span> ({score_label})",
                unsafe_allow_html=True)
            st.markdown(f"*{evaluation['overall_interpretation']}*")

            st.markdown("---")

            # Three metrics in columns
            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                st.markdown("#### Specificity Score")
                spec = evaluation["specificity"]
                st.metric(label="Score", value=f"{spec['score']:.1f}/100")
                st.caption(spec["interpretation"])

                with st.expander("Details"):
                    st.markdown(f"**Specific indicators:** {spec['specific_total']}")
                    st.markdown(f"**Vague indicators:** {spec['vague_total']}")
                    st.markdown(f"**Total words:** {spec['total_words']}")

            with metric_col2:
                st.markdown("#### Structure Score")
                struct = evaluation["structure"]
                st.metric(label="Score", value=f"{struct['completeness_score']:.0f}/100")
                st.caption(struct["interpretation"])

                with st.expander("Details"):
                    st.markdown(f"**Found sections:** {', '.join(struct['found_sections'])}")
                    if struct['missing_sections']:
                        st.markdown(f"**Missing:** {', '.join(struct['missing_sections'])}")

            with metric_col3:
                st.markdown("#### LLM Judge Score")
                judge = evaluation.get("llm_judge", {})

                if "error" not in judge and "overall" in judge:
                    judge_score = judge["overall"].get("score", "N/A")
                    st.metric(label="Score", value=f"{judge_score}/10")
                    st.caption(judge["overall"].get("reason", ""))
                else:
                    st.metric(label="Score", value="N/A")
                    st.caption("Judge evaluation failed")

            st.markdown("---")

            # Detailed LLM Judge scores
            if "error" not in judge and "relevance" in judge:
                st.subheader("Detailed LLM Judge Scores")

                judge_metrics = ["relevance", "accuracy", "completeness", "actionability", "clarity", "technical_depth"]

                cols = st.columns(3)
                for i, metric in enumerate(judge_metrics):
                    if metric in judge and isinstance(judge[metric], dict):
                        with cols[i % 3]:
                            score = judge[metric].get("score", "N/A")
                            reason = judge[metric].get("reason", "")
                            st.markdown(f"**{metric.capitalize()}:** {score}/10")
                            st.caption(reason)

                # Improvement suggestions
                if "improvements" in judge:
                    st.markdown("---")
                    st.subheader("Suggested Improvements")
                    st.info(judge["improvements"])
        else:
            st.warning("Evaluation not available. Run analysis first.")

else:
    # Show placeholder when no analysis is running
    st.info("Enter incident details in the sidebar and click **Analyze Incident** to begin.")

    # Example incidents
    st.subheader("Example Incidents")

    examples = [
        {
            "name": "Database Connection Pool Exhausted",
            "description": "Application throwing 'Cannot acquire connection from pool' errors. Started after deployment of v2.3.1. Affecting 30% of API requests.",
            "priority": "P1 - Critical"
        },
        {
            "name": "Memory Leak in Production",
            "description": "Application crashed with OutOfMemoryError. Heap space exhausted after processing large batch job. JVM heap set to 4GB.",
            "priority": "P2 - High"
        },
        {
            "name": "API Performance Degradation",
            "description": "API response time degraded from 200ms to 5 seconds. CPU usage at 95%. Slow database queries detected.",
            "priority": "P2 - High"
        }
    ]

    for i, example in enumerate(examples):
        with st.expander(f"Example {i + 1}: {example['name']}"):
            st.markdown(f"**Description:** {example['description']}")
            st.markdown(f"**Priority:** {example['priority']}")
            st.markdown("*Copy these values to the sidebar to try them out.*")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Incident RCA System | ML + LLM Pipeline | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
