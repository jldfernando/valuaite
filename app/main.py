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

def display_valuation_results(state):
    """Refactored helper to show the metrics dashboard."""
    if "valuation_results" in state:
        res = state["valuation_results"]
        assump = state.get("assumptions", {})
        
        with st.expander("📊 View Detailed Calculation Data", expanded=True):
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
                st.json(state)

# Sidebar - Settings & About
with st.sidebar:
    st.title("Settings")
    
    # Provider Selection
    provider = st.selectbox(
        "Select LLM Provider:",
        ["Gemini", "Groq", "Mistral", "OpenAI"],
        index=0
    )
    
    # Provider-specific settings
    if provider == "Gemini":
        st.info("Using Gemini 2.5 Flash Lite")
        api_key_field = "GOOGLE_API_KEY"
    elif provider == "Groq":
        st.info("Using Groq (Llama 3.3 70B)")
        api_key_field = "GROQ_API_KEY"
    elif provider == "Mistral":
        st.info("Using Mistral (Mistral Large)")
        api_key_field = "MISTRAL_API_KEY"
    elif provider == "OpenAI":
        st.info("Using OpenAI (GPT-4o-mini)")
        api_key_field = "OPENAI_API_KEY"
    
    # API Key Input
    if not os.getenv(api_key_field):
        api_key = st.text_input(f"Enter {provider} API Key:", type="password")
        if api_key:
            os.environ[api_key_field] = api_key
    
    # Store settings in session state
    st.session_state.llm_config = {
        "provider": provider,
        "api_key": os.getenv(api_key_field)
    }
    
    # Session Persistence (Thread ID)
    if "thread_id" not in st.session_state:
        import uuid
        st.session_state.thread_id = str(uuid.uuid4())
    
    st.caption(f"Thread ID: {st.session_state.thread_id}")
    
    if st.button("Clear Session"):
        import uuid
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

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

# --- AGENT PERSISTENCE LOGIC ---
config = {"configurable": {"thread_id": st.session_state.thread_id}}
current_snapshot = valuation_agent.get_state(config)

# 1. Handle Human-in-the-Loop Interrupts
if current_snapshot.next:
    state_values = current_snapshot.values
    assump = state_values.get("assumptions", {})
    ticker = state_values.get("ticker", "Stock")

    with st.chat_message("assistant"):
        st.warning(f"🎯 **Strategy Checkpoint: {ticker} Blueprint**")
        st.info(f"**Analyst Rationale:** {assump.get('reasoning', 'No reasoning provided.')}")
        
        with st.form("approval_form"):
            st.write("### 🛠️ Modeling Levers")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**Growth & Premium**")
                new_growth = st.number_input("Forward Growth (%)", value=float(assump.get("forward_growth", 0.05)*100), step=0.5) / 100
                new_term_growth = st.number_input("Terminal Growth (%)", value=float(assump.get("terminal_growth", 0.025)*100), step=0.1) / 100
                new_erp = st.number_input("Equity Risk Premium (%)", value=float(assump.get("equity_risk_premium", 0.055)*100), step=0.1) / 100
            with col2:
                st.write("**WACC Inputs**")
                new_rf = st.number_input("Risk-Free Rate (%)", value=float(assump.get("risk_free_rate", 0.045)*100), step=0.1) / 100
                new_beta = st.number_input("Beta", value=float(assump.get("beta", 1.0)), step=0.05)
                new_tax = st.number_input("Tax Rate (%)", value=float(assump.get("tax_rate", 0.25)*100), step=1.0) / 100
            with col3:
                st.write("**Global Settings**")
                new_peers = st.text_input("Peer Tickers", value=", ".join(assump.get("peers", [])))
                intent = st.selectbox("Depth", ["FULL_VALUATION", "QUICK_INQUIRY"], index=0 if assump.get("intent")=="FULL_VALUATION" else 1)
                
                h_inv, h_rec = assump.get("haircuts", [0.1, 0.15])
                new_h_inv = st.slider("Inventory Haircut", 0.0, 1.0, float(h_inv))
                new_h_rec = st.slider("Receivables Haircut", 0.0, 1.0, float(h_rec))
            
            feedback = st.text_area("💬 Feedback / Argument (Optional)", placeholder="e.g., 'I think 40% growth is too aggressive, let's use 20% instead.'")
            
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                submitted = st.form_submit_button("🚀 Approve & Execute Calculations")
            with btn_col2:
                negotiate = st.form_submit_button("🔄 Negotiate / Update Plan")
            
            if submitted:
                # Update State with user overrides and move forward
                new_assumptions = {
                    **assump,
                    "forward_growth": new_growth,
                    "terminal_growth": new_term_growth,
                    "equity_risk_premium": new_erp,
                    "risk_free_rate": new_rf,
                    "beta": new_beta,
                    "tax_rate": new_tax,
                    "peers": [p.strip().upper() for p in new_peers.split(",") if p.strip()],
                    "haircuts": [new_h_inv, new_h_rec],
                    "intent": intent
                }
                valuation_agent.update_state(config, {"assumptions": new_assumptions})
                
                # Resume execution
                with st.spinner("Processing approved strategy..."):
                    final_state = valuation_agent.invoke(None, config=config)
                    if final_state.get("analysis_report"):
                        report = final_state["analysis_report"]
                        st.session_state.messages.append({"role": "assistant", "content": report})
                        st.rerun()

            elif negotiate:
                # Add user feedback to chat history and REWIND to planner
                if feedback:
                    user_msg = {"role": "user", "content": f"Feedback on your plan: {feedback}"}
                    st.session_state.messages.append(user_msg)
                    # We need to manually add the message to the state and then invoke starting from planner
                    valuation_agent.update_state(config, {"messages": [user_msg]})
                    
                    with st.spinner("Renegotiating strategy..."):
                        # By invoking with the config and NO initial state, it resumes. 
                        # But since we're stuck at the interrupt BEFORE financial_engine, 
                        # we want to go BACK to analyst_planner.
                        # In this simple graph, we can just invoke again from START with the updated messages.
                        current_ticker = state_values.get("ticker", "UNKNOWN")
                        valuation_agent.update_state(config, {"current_step": "analyst_planner"})
                        final_state = valuation_agent.invoke(None, config=config)
                        st.rerun()

# 2. Handle New Queries
if prompt := st.chat_input("Ask about a stock (e.g., 'Valuate AAPL')"):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Researching and planning..."):
            initial_state = {
                "messages": [{"role": "user", "content": prompt}],
                "ticker": "UNKNOWN",
                "errors": [],
                "current_step": "start",
                "config": st.session_state.get("llm_config", {"provider": "Gemini"})
            }
            
            # This will run until it hits an interrupt or END
            final_state = valuation_agent.invoke(initial_state, config=config)
            
            # If we hit an interrupt, we just rerun the page to show the form
            if valuation_agent.get_state(config).next:
                st.rerun()
            
            # Otherwise, handle completion or errors
            if final_state.get("errors"):
                st.error(f"❌ Error: {final_state['errors'][-1]}")
            elif final_state.get("analysis_report"):
                st.markdown(final_state["analysis_report"])
                st.session_state.messages.append({"role": "assistant", "content": final_state["analysis_report"]})
                # Show results dashboard
                display_valuation_results(final_state)

# 3. Always show results if they exist in the current snapshot (for persistence)
if "valuation_results" in current_snapshot.values:
    display_valuation_results(current_snapshot.values)

st.divider()
st.caption("Disclaimer: This tool is for research purposes only. Not financial advice.")
