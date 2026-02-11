# Business Valuation AI Agent 🚀

An AI-powered financial co-pilot built with LangGraph and Gemini 2.0 Flash. This agent performs intrinsic (DCF), relative (Multiples), and asset-based valuations for investors and students. It features a "Chat-First" interface with Human-in-the-Loop (HITL) checkpoints to ensure mathematical and data accuracy.

## 🌟 Key Features

* Conversational Valuation: Interactively build financial models through a chat interface.

* Proactive Logic: The agent suggests WACC, growth rates, and peer groups based on real-time data.

* Multi-Methodology: Supports DCF, Peer Multiples (P/E, PEG, EV/EBITDA, etc.), and Asset-Based (NAV) valuations.

* Advanced Logic Loop: Specialized mode for professional investors to apply custom technical methodologies (e.g., Moving Averages).

* Expansion Ready: Architected to support Philippine (PH) market data via PDF RAG and MCP integration.

## 🛠 Tech Stack

* Orchestration: LangGraph

* Brains: Multi-LLM support (Gemini 1.5/2.0, Groq Llama 3.3, Mistral Large, OpenAI GPT-4o)

* UI: Streamlit

* Data: yfinance (US Stocks), PyPDF2 (PH SEC Filings)

* Engine: Python (NumPy/Pandas)

* Evaluation: RAGAS

## 📂 Project Structure
```
valuation-ai-agent/
├── app/
│   ├── main.py              # Streamlit Entry Point
│   ├── agent/               
│   │   ├── graph.py         # LangGraph State Machine
│   │   ├── nodes.py         # Agent Branching & Logic
│   │   ├── state.py         # Typed State Schema
│   │   └── llm_factory.py   # Multi-Provider LLM Wrapper (NEW)
│   ├── tools/               # Data & Math Engines
│   └── utils/               # Guardrails & Visuals
├── Dockerfile               # Containerization
└── requirements.txt         # Dependencies
```

## 🚀 Quick Start (Prototype)

Clone the repository:
```
git clone <your-repo-url>
cd valuation-ai-agent
```

Install Dependencies:
```
pip install -r requirements.txt
```

Set Environment Variables:
Create a .env file in the root directory:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```

Run the App:
```
streamlit run app/main.py
```

## 📈 Roadmap

[ ] Phase 1: US MVP (yfinance + Basic DCF/Multiples)

[ ] Phase 2: HITL Implementation (Interrupt nodes for verification)

[ ] Phase 3: Advanced Logic (Custom assumptions & Moving Averages)

[ ] Phase 4: International Expansion (PH Market PDF RAG)

## ⚖️ Disclaimer

This tool is for educational and research purposes only. It does not constitute financial advice. Always verify AI-generated numbers with official SEC filings.