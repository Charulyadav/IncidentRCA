"""
Streamlit UI for Incident RCA System
=====================================
Two-stage pipeline: ML Classification + LLM Resolution
"""

import streamlit as st
import json
from integrated import process_incident

# Page configuration
st.set_page_config(
    page_title="Incident RCA System",
    page_icon="🔍",
    layout="wide"
)

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
    placeholder="Describe the incident in detail...\n\nExample: Application throwing 'Cannot acquire connection from pool' errors. Started after deployment of v2.3.1. Affecting 30% of API requests.",
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
3. Results displayed with confidence scores
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### Categories")
categories = [
    "Configuration Issue", "Database Issue", "Deployment Issue",
    "Infrastructure Issue", "Integration Issue", "Memory Issue",
    "Network Hardware", "Network Issue", "Other",
    "Performance Issue", "Security Issue", "Software Error"
]
st.sidebar.markdown(", ".join(categories))

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

                # Success message
                st.success("Analysis complete!")

                # Display results in columns
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
                    # Confidence level indicator
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

                # Check for errors
                if result["llm_analysis"].startswith("Error:"):
                    st.error(result["llm_analysis"])
                    st.info("Make sure Ollama is running: `ollama serve`")
                else:
                    st.markdown(result["llm_analysis"])

                # Download results as JSON
                st.markdown("---")
                st.download_button(
                    label="Download Results as JSON",
                    data=json.dumps(result, indent=2),
                    file_name=f"rca_result_{issue_name.replace(' ', '_')[:30]}.json",
                    mime="application/json"
                )

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Make sure Ollama is running: `ollama serve`")

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