import streamlit as st
import os
from dotenv import load_dotenv
from agent.graph import valuation_agent

load_dotenv()

# Page Config
st.set_page_config(
    page_title="Valuation AI Agent",
    page_icon="🚀",
    layout="wide"
)

# Sidebar - Settings & About
with st.sidebar:
    st.title("Settings")
    st.info("This agent uses Gemini 2.5 Flash Lite to perform financial analysis.")
    if not os.getenv("GOOGLE_API_KEY"):
        api_key = st.text_input("Enter Google API Key:", type="password")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
    st.divider()
    st.markdown("### Common Queries")
    st.caption("- Valuate Apple (AAPL)")
    st.caption("- What is Tesla's P/E ratio?")
    st.caption("- Analyze MSFT based on assets.")

# Main UI
st.title("🚀 Business Valuation AI Agent")
st.markdown("---")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask about a stock (e.g., 'Valuate AAPL')"):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run the Agent
    with st.chat_message("assistant"):
        with st.spinner("Analyzing financials and building models..."):
            try:
                # Prepare Initial State
                # Note: 'messages' in state needs to be LangChain compatible format
                # For the graph, we'll pass a simple dict structure
                initial_state = {
                    "messages": [{"role": "user", "content": prompt}],
                    "ticker": "UNKNOWN",
                    "errors": [],
                    "current_step": "start"
                }
                
                # Invoke Graph
                # valuation_agent is the compiled LangGraph object
                final_state = valuation_agent.invoke(initial_state)
                
                # Check for errors
                if final_state.get("errors"):
                    response = f"❌ **Error:** {final_state['errors'][-1]}"
                else:
                    response = final_state.get("analysis_report", "I couldn't generate a report. Please try again.")

                # If there are valuation results, show them in a neat expando
                if "valuation_results" in final_state:
                    res = final_state["valuation_results"]
                    assump = final_state.get("assumptions", {})
                    
                    with st.expander("📊 View Detailed Calculation Data"):
                        st.subheader("Implied Valuation Prices")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("DCF Implied Price", f"${res['dcf'].get('implied_share_price', 0):.2f}")
                        with col2:
                            pe_price = res['multiples'].get('pe_implied_price', 0)
                            st.metric("PE Implied Price", f"${pe_price:.2f}")
                        with col3:
                            st.metric("NAV Per Share", f"${res['nav'].get('nav_per_share', 0):.2f}")
                        with col4:
                            liq_price = res['liquidation'].get('liquidation_per_share', 0)
                            st.metric("Liquidation (Floor)", f"${liq_price:.2f}")
                        
                        st.divider()
                        
                        st.subheader("Key Model Assumptions")
                        acol1, acol2, acol3, acol4 = st.columns(4)
                        with acol1:
                            st.metric("WACC (Calculated)", f"{res.get('calculated_wacc', 0):.2%}")
                        with acol2:
                            st.metric("Equity Risk Premium", f"{assump.get('equity_risk_premium', 0):.2%}")
                        with acol3:
                            st.metric("Risk-Free Rate", f"{assump.get('risk_free_rate', 0):.2%}")
                        with acol4:
                            st.metric("Forward Growth", f"{assump.get('forward_growth', 0):.2%}")
                        
                        with st.container():
                            st.caption(f"**Peers used for relative valuation:** {', '.join(assump.get('peers', []))}")
                            st.caption(f"**Inventory/Receivables Haircuts:** {assump.get('haircuts', [0,0])}")

                        st.divider()
                        with st.expander("Raw State Data (JSON)"):
                            st.json(final_state)

                # Output final analysis
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                error_msg = f"An unexpected error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

st.divider()
st.caption("Disclaimer: This tool is for research purposes only. Not financial advice.")
