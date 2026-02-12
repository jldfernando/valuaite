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

import pandas as pd
import plotly.graph_objects as go

def display_valuation_results(state):
    """Refactored helper to show the metrics dashboard with tabs and charts."""
    if "valuation_results" in state:
        res = state["valuation_results"]
        assump = state.get("assumptions", {})
        ticker = state.get("ticker", "Stock")
        
        tabs = st.tabs(["📊 Executive Summary", "🧮 Model Details", "📈 Visual Insights"])

        # --- TAB 1: EXECUTIVE SUMMARY ---
        with tabs[0]:
            st.subheader(f"Valuation Summary: {ticker}")
            mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
            curr_price = state.get("company_data", {}).get("market_info", {}).get("current_price", 0)
            with mcol1:
                st.metric("Current Price", f"${curr_price:.2f}")
            with mcol2:
                st.metric("DCF Fair Value", f"${res['dcf'].get('implied_share_price', 0):.2f}")
            with mcol3:
                st.metric("Relative (PE)", f"${res['multiples'].get('pe_implied_price', 0):.2f}")
            with mcol4:
                st.metric("NAV Per Share", f"${res['nav'].get('nav_per_share', 0):.2f}")
            with mcol5:
                st.metric("Liquidation", f"${res['liquidation'].get('liquidation_per_share', 0):.2f}")
            
            st.divider()
            
            # Display AI Report if it exists in state
            if "analysis_report" in state:
                st.markdown(state["analysis_report"])
            
            # Reset Button
            if st.button("✨ Start New Analysis", key="dash_reset", width='stretch'):
                import uuid
                st.session_state.thread_id = str(uuid.uuid4())
                st.session_state.messages = []
                st.rerun()

        # --- TAB 2: MODEL DETAILS (The "Excel" View) ---
        with tabs[1]:
            st.subheader("Model Inputs & Projections")
            
            # 1. DCF Table (Historical + Projections)
            if "dcf" in res and "projections" in res["dcf"]:
                st.write("**Full FCF Audit (Historical -> Projected)**")
                proj = res["dcf"]["projections"]
                hist_fcf = res.get("historical_fcf", [])[::-1] # Reverse to chronological [Oldest -> Newest]
                
                # Create combined timeline
                hist_labels = [f"Y-{len(hist_fcf)-i}" for i in range(len(hist_fcf))]
                proj_labels = proj["years"]
                
                combined_fcf = hist_fcf + proj["fcf"]
                combined_labels = hist_labels + proj_labels
                
                # Types for coloring
                fcf_types = ["Historical"] * len(hist_fcf) + ["Projected"] * len(proj["fcf"])
                
                df_dcf = pd.DataFrame({
                    "Period": combined_labels,
                    "Type": fcf_types,
                    "Free Cash Flow": combined_fcf,
                })
                
                st.dataframe(df_dcf.style.format({
                    "Free Cash Flow": "${:,.2f}"
                }), width='stretch', hide_index=True)
                
                # Terminal Value Box
                tcol1, tcol2 = st.columns(2)
                tcol1.metric("Terminal Value", f"${proj['terminal_value']:,.2f}")
                tcol2.metric("PV of Terminal Value", f"${proj['pv_terminal']:,.2f}")

            st.divider()
            
            # 2. Peer Multiples Table
            if "peer_data" in res and "raw" in res["peer_data"]:
                st.write("**Peer Benchmark Analysis**")
                peers_raw = res["peer_data"]["raw"]
                df_peers = pd.DataFrame(peers_raw)
                # Filter to only show key valuation columns if they exist
                cols = [c for c in ["ticker", "price", "pe", "ps", "ev_ebitda", "peg"] if c in df_peers.columns]
                if not df_peers.empty:
                    st.dataframe(df_peers[cols].set_index("ticker").T, width='stretch')
                
            st.divider()
            
            # 3. Asset-Based Breakdown
            st.write("**Asset-Based & WACC Parameters**")
            wcol1, wcol2 = st.columns(2)
            with wcol1:
                df_wacc = pd.DataFrame({
                    "Parameter": ["Risk-Free Rate", "Beta", "Equity Risk Premium", "Cost of Equity", "Tax Rate"],
                    "Value": [
                        f"{assump.get('risk_free_rate', 0):.2%}",
                        f"{assump.get('beta', 1.0):.2f}",
                        f"{assump.get('equity_risk_premium', 0):.2%}",
                        f"{res.get('calculated_wacc', 0):.2%}",
                        f"{assump.get('tax_rate', 0.25):.1%}"
                    ]
                })
                st.table(df_wacc)
            with wcol2:
                df_haircuts = pd.DataFrame({
                    "Asset Class": ["Inventory", "Receivables"],
                    "Discount (Haircut)": [f"{h:.1%}" for h in assump.get("haircuts", [0.1, 0.15])]
                })
                st.table(df_haircuts)

        # --- TAB 3: VISUAL INSIGHTS ---
        with tabs[2]:
            st.subheader("Valuation Visualizations")
            
            # 1. Football Field Chart (Models + Peers)
            models = ["Current Price", "DCF", "PE Multiples", "NAV (Asset)", "Liquidation"]
            prices = [
                curr_price,
                res['dcf'].get('implied_share_price', 0),
                res['multiples'].get('pe_implied_price', 0),
                res['nav'].get('nav_per_share', 0),
                res['liquidation'].get('liquidation_per_share', 0)
            ]
            colors = ['#FFFFFF', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            
            # Add peers to the chart
            if "peer_data" in res and "raw" in res["peer_data"]:
                for p in res["peer_data"]["raw"]:
                    models.append(f"Peer: {p['ticker']}")
                    prices.append(p.get("price", 0))
                    colors.append('#9467bd') # Purple for peers
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=prices,
                y=models,
                orientation='h',
                marker_color=colors,
                text=[f"${p:.2f}" for p in prices],
                textposition='auto'
            ))
            
            # Add current price line
            if curr_price > 0:
                fig.add_vline(x=curr_price, line_dash="dash", line_color="white", 
                             annotation_text=f"Market: ${curr_price:.2f}", 
                             annotation_position="top",
                             annotation_font_color="white")

            fig.update_layout(title="Valuation Comparison vs. Market & Peers", xaxis_title="Price ($)", 
                             yaxis_title="Methodology / Benchmarks", template="plotly_white", height=400 + (len(models)*20))
            st.plotly_chart(fig, width='stretch')

            # 2. Scenario Comparison Chart (Football Field format)
            if "scenarios" in state and state["scenarios"]:
                st.divider()
                st.write("**Scenario Sensitivity Analysis**")
                s_names = list(state["scenarios"].keys())
                s_prices = [v["implied_price"] for v in state["scenarios"].values()]
                s_colors = ['#d62728', '#1f77b4', '#2ca02c'] # Bear (Red), Base (Blue), Bull (Green)
                
                # Add Current Price for perspective
                s_names = ["Current Price"] + s_names
                s_prices = [curr_price] + s_prices
                s_colors = ['#FFFFFF'] + s_colors
                
                s_fig = go.Figure()
                s_fig.add_trace(go.Bar(
                    x=s_prices,
                    y=s_names,
                    orientation='h',
                    marker_color=s_colors,
                    text=[f"${p:.2f}" for p in s_prices],
                    textposition='auto'
                ))
                
                # Add market line
                if curr_price > 0:
                    s_fig.add_vline(x=curr_price, line_dash="dash", line_color="white")
                
                s_fig.update_layout(title="Scenario Sensitivity Comparison", xaxis_title="Implied Price ($)", template="plotly_white")
                st.plotly_chart(s_fig, width='stretch')
                
                # Explicit Assumption Table
                st.write("**Scenario Assumptions Used:**")
                scen_data = []
                for name, vals in state["scenarios"].items():
                    scen_data.append({
                        "Scenario": name,
                        "Assumptions": vals.get("assumptions_desc", "N/A"),
                        "Implied Price": f"${vals['implied_price']:.2f}"
                    })
                st.table(pd.DataFrame(scen_data))
            else:
                if st.button("📊 Generate Scenario Sensitivity Charts", width='stretch'):
                    with st.spinner("Simulating Bull/Bear market conditions..."):
                        from agent.nodes import scenario_analysis_node
                        scenario_state = scenario_analysis_node(state)
                        from agent.graph import valuation_agent
                        config = {"configurable": {"thread_id": st.session_state.thread_id}}
                        valuation_agent.update_state(config, {"scenarios": scenario_state["scenarios"]})
                        st.rerun()

# Sidebar - Settings & About
with st.sidebar:
    st.title("Settings")
    
    # Provider Selection
    provider = st.selectbox(
        "Select LLM Provider:",
        ["Gemini", "Groq", "Mistral", "OpenAI"],
        index=1
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
                valuation_agent.update_state(config, {
                    "assumptions": new_assumptions,
                    "config": st.session_state.llm_config
                })
                
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
                        valuation_agent.update_state(config, {
                            "current_step": "analyst_planner",
                            "config": st.session_state.llm_config
                        })
                        final_state = valuation_agent.invoke(None, config=config)
                        st.rerun()

# 2. Handle New Queries
if prompt := st.chat_input("Ask about a stock (e.g., 'Valuate AAPL')"):
    # Clear session logic removed to prevent message loss. 
    # Use the manual 'Clear Session' button in the sidebar or results panel.
    
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            # Prepare state update: we only send the new message
            # LangGraph handles the merging/appending via the checkpointer
            state_update = {
                "messages": [{"role": "user", "content": prompt}],
                "errors": [], # Clear old errors
                "current_step": "start", # Restart flow
                "config": st.session_state.llm_config # Pass provider settings
            }
            
            # This will run from the start node, but will have access to existing state (like 'ticker')
            final_state = valuation_agent.invoke(state_update, config=config)
            
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
