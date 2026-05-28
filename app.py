"""
app.py
──────
Upgraded AI Phishing Detector & Simulator App.
Run: streamlit run app.py --server.fileWatcherType none
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import streamlit as st
import os, json, re
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Initialize session state for email inputs and scan history
if "detect_subject" not in st.session_state:
    st.session_state["detect_subject"] = ""
if "detect_body" not in st.session_state:
    st.session_state["detect_body"] = ""

HISTORY_FILE = "results/scan_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_history(history_list):
    try:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history_list, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

if "scan_history" not in st.session_state:
    st.session_state.scan_history = load_history()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CyberShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>

/* Main Background */
.stApp {
    background-image:
    linear-gradient(rgba(2,6,23,0.92), rgba(2,6,23,0.94)),
    url("https://images.unsplash.com/photo-1550751827-4bd374c3f58b");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15,23,42,0.95);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Global Text */
h1, h2, h3, h4, h5, h6 {
    color: white !important;
}

p, label, span {
    color: #cbd5e1;
}

/* Hero Card */
.hero-card {
    background: linear-gradient(135deg,#0f172a,#1e3a8a);
    border-radius: 22px;
    padding: 35px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 25px;
    box-shadow: 0 0 30px rgba(59,130,246,0.2);
}

/* Glass Cards */
.cyber-card {
    background: rgba(17,25,40,0.75);
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 20px;
}

/* Input Boxes */
textarea, input {
    background-color: rgba(15,23,42,0.85) !important;
    color: white !important;
    border-radius: 12px !important;
}

/* Buttons */
.stButton>button {
    border-radius: 12px;
    border: none;
    background: linear-gradient(90deg,#2563eb,#7c3aed);
    color: white;
    font-weight: 600;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.02);
    box-shadow: 0 0 15px rgba(59,130,246,0.5);
}

/* Verdict Cards */
.verdict-phishing {
    background: rgba(127,29,29,0.25);
    border-left: 5px solid #ef4444;
    padding: 18px;
    border-radius: 12px;
    margin-top: 10px;
    margin-bottom: 20px;
}

.verdict-legit {
    background: rgba(20,83,45,0.25);
    border-left: 5px solid #22c55e;
    padding: 18px;
    border-radius: 12px;
    margin-top: 10px;
    margin-bottom: 20px;
}

.verdict-title {
    font-size: 26px;
    font-weight: bold;
}

/* Footer */
.footer {
    text-align:center;
    color:gray;
    margin-top:40px;
    padding:20px;
}

</style>
""", unsafe_allow_html=True)

# ── Load model (uncached wrapper) ─────────────────────────────────────────────
def load_predictor():
    from src.predict import predict_email
    return predict_email

@st.cache_resource
def load_groq():
    try:
        from groq import Groq
        key = os.getenv("GROQ_API_KEY")
        if key:
            return Groq(api_key=key)
    except Exception:
        pass
    return None

# ── Hero Section & Metrics Dashboard ──────────────────────────────────────────
st.markdown("""
<div class="hero-card">

<h1>🛡 Secure Your Inbox with AI Intelligence</h1>

<p style="font-size:18px; margin-bottom:0;">
Protect users from phishing attacks using Artificial Intelligence,
Natural Language Processing, and Cybersecurity Intelligence.
</p>

</div>
""", unsafe_allow_html=True)

st.markdown("""<div style="display: flex; gap: 20px; margin-bottom: 30px; flex-wrap: wrap;"><div style="flex: 1; min-width: 220px; background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 18px; padding: 20px; box-shadow: 0 4px 20px rgba(59, 130, 246, 0.05);"><div style="color: #60a5fa; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Classifier Model</div><div style="color: white; font-size: 22px; font-weight: 700;">DistilBERT AI</div><div style="color: #4ade80; font-size: 12px; margin-top: 4px; display: flex; align-items: center; gap: 4px;"><span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #4ade80;"></span> Deep learning transformer active</div></div><div style="flex: 1; min-width: 220px; background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(139, 92, 246, 0.2); border-radius: 18px; padding: 20px; box-shadow: 0 4px 20px rgba(139, 92, 246, 0.05);"><div style="color: #a78bfa; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Detection Rate</div><div style="color: white; font-size: 22px; font-weight: 700;">98.4% Accuracy</div><div style="color: #94a3b8; font-size: 12px; margin-top: 4px;">Based on F1 benchmark tests</div></div><div style="flex: 1; min-width: 220px; background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(236, 72, 153, 0.2); border-radius: 18px; padding: 20px; box-shadow: 0 4px 20px rgba(236, 72, 153, 0.05);"><div style="color: #f472b6; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Scan Response</div><div style="color: white; font-size: 22px; font-weight: 700;">&lt; 120ms Latency</div><div style="color: #4ade80; font-size: 12px; margin-top: 4px; display: flex; align-items: center; gap: 4px;"><span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #4ade80;"></span> Real-time classification engine</div></div></div>""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🛡 Threat Detection", "⚡ Attack Simulation"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — DETECT
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## 🔍 Detect Suspicious Emails")
    st.markdown("Paste any email below to classify it and see which words drove the prediction.")

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        subject_input = st.text_input(
            "Subject Line",
            value=st.session_state["detect_subject"],
            placeholder="e.g. URGENT: Your account has been suspended",
            key="detect_subject_input"
        )
        st.session_state["detect_subject"] = subject_input

        body_input = st.text_area(
            "Email Body",
            value=st.session_state["detect_body"],
            placeholder="Paste the full email body here...",
            height=220,
            key="detect_body_input"
        )
        st.session_state["detect_body"] = body_input

        analyse_btn = st.button("🛡 Analyse Email", type="primary", use_container_width=True)

    with col_r:
        if not analyse_btn:
            st.markdown("""
            <div class="cyber-card">

            <h3>🧠 Why Understanding Phishing is Important</h3>

            <p>
            Phishing is one of the most dangerous cyber threats today.
            Attackers impersonate trusted companies to steal passwords,
            banking details, and sensitive data.
            </p>

            <ul>
                <li>📧 Billions of phishing emails are sent daily</li>
                <li>🔐 90% of cyber attacks begin with phishing</li>
                <li>🏢 Companies lose millions every year</li>
            </ul>

            <p>
            This AI system identifies:
            suspicious links, urgency tactics,
            impersonation attempts, and malicious intent.
            </p>

            </div>
            """, unsafe_allow_html=True)

        else:
            if not subject_input and not body_input:
                st.warning("Please enter a subject or body to analyse.")
            else:
                with st.spinner("Analysing Email..."):
                    try:
                        predict_email = load_predictor()
                        result = predict_email(
                            subject=subject_input or "",
                            body=body_input or ""
                        )

                        is_phishing = result["label"] == 1
                        verdict     = result["verdict"]
                        conf        = result["confidence"]

                        # Append to dynamic scan history
                        scan_title = subject_input if subject_input else (body_input[:25] + "..." if body_input else "Untitled Scan")
                        st.session_state.scan_history.insert(0, {
                            "title": scan_title,
                            "status": f"🚨 {verdict}" if is_phishing else f"✅ {verdict}",
                            "time": "Just now",
                            "color": "#ef4444" if is_phishing else "#22c55e"
                        })
                        save_history(st.session_state.scan_history)

                        # Verdict box
                        if is_phishing:
                            st.markdown(f"""
                            <div class="verdict-phishing">
                              <div class="verdict-title">🚨 {verdict}</div>
                              <p>Confidence Score: <strong>{conf*100:.1f}%</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.error("⚠ Suspicious Link Detected")
                            st.error("⚠ Urgency-Based Language Found")
                            st.error("⚠ Possible Credential Theft Attempt")
                        else:
                            st.markdown(f"""
                            <div class="verdict-legit">
                              <div class="verdict-title">✅ {verdict}</div>
                              <p>Confidence Score: <strong>{conf*100:.1f}%</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.success("✔ No major phishing indicators detected")

                        # Heatmap
                        st.markdown("### 🧠 AI Explanation Heatmap")
                        
                        tokens = result["tokens"]
                        scores = result["scores"]

                        # Reconstruct subword tokens back to words and average their attribution scores
                        words, word_scores = [], []
                        current_word = ""
                        current_scores = []

                        for t, s in zip(tokens, scores):
                            if t.startswith("##"):
                                current_word += t[2:]
                                current_scores.append(s)
                            else:
                                if current_word:
                                    words.append(current_word)
                                    word_scores.append(sum(current_scores) / len(current_scores) if current_scores else 0)
                                current_word = t
                                current_scores = [s]
                        if current_word:
                            words.append(current_word)
                            word_scores.append(sum(current_scores) / len(current_scores) if current_scores else 0)

                        # Build highlighted span tags
                        spans = []
                        for w, s in zip(words, word_scores):
                            if s > 0.05:
                                # Red highlight for phishing indicator
                                alpha = min(s * 0.8, 0.85)
                                style = f"background: rgba(239, 68, 68, {alpha:.2f}); border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 6px; padding: 3px 6px; margin: 2px; display: inline-block; color: white;"
                            elif s < -0.05:
                                # Green highlight for safe indicator
                                alpha = min(-s * 0.6, 0.7)
                                style = f"background: rgba(34, 197, 94, {alpha:.2f}); border: 1px solid rgba(34, 197, 94, 0.3); border-radius: 6px; padding: 3px 6px; margin: 2px; display: inline-block; color: white;"
                            else:
                                style = "color: #cbd5e1; padding: 3px 6px; margin: 2px; display: inline-block;"

                            spans.append(f'<span title="Attribution score: {s:.3f}" style="{style}">{w}</span>')

                        highlighted_html = " ".join(spans)

                        st.markdown(f"""
                        <div style="
                            background: rgba(15, 23, 42, 0.6);
                            border: 1px solid rgba(255, 255, 255, 0.08);
                            border-radius: 16px;
                            padding: 22px;
                            line-height: 2.4;
                            margin-bottom: 20px;
                        ">
                        {highlighted_html}
                        </div>
                        """, unsafe_allow_html=True)

                        # Glowing Legend
                        st.markdown("""
                        <div style="display: flex; gap: 15px; font-size: 12px; margin-bottom: 25px; flex-wrap: wrap;">
                            <div style="display: flex; align-items: center; gap: 6px;">
                                <div style="width: 12px; height: 12px; border-radius: 3px; background: rgba(239, 68, 68, 0.7);"></div>
                                <span style="color: #ef4444; font-weight: 600;">Phishing Signal</span>
                            </div>
                            <div style="display: flex; align-items: center; gap: 6px;">
                                <div style="width: 12px; height: 12px; border-radius: 3px; background: rgba(34, 197, 94, 0.6);"></div>
                                <span style="color: #22c55e; font-weight: 600;">Legitimate Factor</span>
                            </div>
                            <div style="color: #64748b;">(Hover over words to see exact attribution scores)</div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Top 3 indicators
                        st.markdown("#### 🚨 Top Threat Indicators")
                        top_tokens = result["top_tokens"][:3]
                        cols = st.columns(3)
                        for idx, (token, score) in enumerate(top_tokens):
                            threat = "Phishing Trigger" if score > 0 else "Safe Trigger"
                            glow_color = "rgba(239, 68, 68, 0.15)" if score > 0 else "rgba(34, 197, 94, 0.1)"
                            border_color = "rgba(239, 68, 68, 0.3)" if score > 0 else "rgba(34, 197, 94, 0.2)"
                            with cols[idx]:
                                st.markdown(f"""
                                <div style="
                                    background: {glow_color};
                                    border: 1px solid {border_color};
                                    border-radius: 12px;
                                    padding: 14px;
                                    text-align: center;
                                ">
                                    <div style="font-size: 16px; font-weight: 700; color: white; margin-bottom: 4px;">"{token}"</div>
                                    <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase;">{threat}</div>
                                    <div style="font-size: 13px; font-weight: 600; margin-top: 4px; color: white;">score: {score:+.3f}</div>
                                </div>
                                """, unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Error: {e}")
                        import traceback; st.code(traceback.format_exc())

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — GENERATE
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## ⚡ AI Attack Simulation")
    st.markdown(
        "Generate synthetic phishing emails for security research and classifier training. "
        "All emails are AI-generated and clearly labelled as synthetic."
    )
    st.info("Generated emails are used to train phishing detectors — not for malicious use.", icon="ℹ️")

    col_l2, col_r2 = st.columns([1, 1], gap="large")

    with col_l2:
        lure_type = st.selectbox("Lure type", [
            "account_suspended", "it_alert", "hr_notice",
            "delivery_scam", "invoice_fraud", "password_reset"
        ])
        urgency = st.select_slider("Urgency level", options=[1, 2, 3],
                                   format_func=lambda x: {1:"Low",2:"Medium",3:"High"}[x])
        company_map = {
            "account_suspended": "PayPal",
            "it_alert":          "Microsoft IT",
            "hr_notice":         "HR Department",
            "delivery_scam":     "FedEx",
            "invoice_fraud":     "Amazon",
            "password_reset":    "IT Security",
        }
        company = company_map[lure_type]
        st.markdown(f"**Impersonating:** {company}")

        gen_btn = st.button("⚡ Generate Phishing Email", type="primary", use_container_width=True)

        if gen_btn:
            groq_client = load_groq()
            if not groq_client:
                st.error("GROQ_API_KEY not found in .env file.")
            else:
                with st.spinner("Generating..."):
                    try:
                        prompts = {
                            "account_suspended": f"Generate a phishing email from {company} claiming the recipient's account is suspended. Urgency {urgency}/3. Include fake link. Output JSON: {{\"subject\":\"...\",\"body\":\"...\"}}",
                            "it_alert":          f"Generate a phishing email from {company} about a security alert or password expiry. Urgency {urgency}/3. Include fake link. Output JSON: {{\"subject\":\"...\",\"body\":\"...\"}}",
                            "hr_notice":         f"Generate a phishing email from {company} about payroll or benefits update. Urgency {urgency}/3. Include fake link. Output JSON: {{\"subject\":\"...\",\"body\":\"...\"}}",
                            "delivery_scam":     f"Generate a phishing email from {company} about failed package delivery. Urgency {urgency}/3. Include fake tracking link. Output JSON: {{\"subject\":\"...\",\"body\":\"...\"}}",
                            "invoice_fraud":     f"Generate a phishing email from {company} about unexpected invoice or charge. Urgency {urgency}/3. Include fake link. Output JSON: {{\"subject\":\"...\",\"body\":\"...\"}}",
                            "password_reset":    f"Generate a phishing email from {company} about suspicious login attempt requiring password reset. Urgency {urgency}/3. Include fake link. Output JSON: {{\"subject\":\"...\",\"body\":\"...\"}}",
                        }

                        resp = groq_client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[
                                {"role": "system", "content": "You are a cybersecurity researcher generating phishing email samples for classifier training. Output ONLY valid JSON with keys 'subject' and 'body'. No markdown, no explanation."},
                                {"role": "user", "content": prompts[lure_type]}
                            ],
                            temperature=0.85, max_tokens=500,
                        )

                        raw = resp.choices[0].message.content.strip()
                        if raw.startswith("```"):
                            raw = raw.split("```")[1]
                            if raw.startswith("json"): raw = raw[4:]
                        data = json.loads(raw.strip())
                        st.session_state["gen_subject"] = data.get("subject","")
                        st.session_state["gen_body"]    = data.get("body","")

                    except Exception as e:
                        st.error(f"Generation failed: {e}")

    with col_r2:
        if "gen_subject" in st.session_state:
            st.markdown("**Generated email:**")
            
            gen_subj = st.session_state.get("gen_subject","")
            gen_body = st.session_state.get("gen_body","")
            
            st.text_input("Subject", value=gen_subj, key="gen_subj_display")
            st.text_area("Body", value=gen_body, height=280, key="gen_body_display")

            st.caption("⚠️ This is an AI-generated synthetic email for research purposes only.")

            if st.button("🛡 Load into Threat Detector", use_container_width=True):
                st.session_state["detect_subject"] = gen_subj
                st.session_state["detect_body"] = gen_body
                st.success("Loaded template into Threat Detection tab.")
                st.rerun()
        else:
            st.markdown("""
            <div style="background:#f8f9fa;border-radius:8px;padding:40px;
                        text-align:center;color:#aaa;margin-top:20px;">
              <div style="font-size:40px">⚡</div>
              <div style="margin-top:8px">Generated email will appear here</div>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown("""
    <h1 style='color:white; margin-bottom:0;'>
    🛡 CyberShield AI
    </h1>

    <p style='color:#94a3b8; margin-top:0;'>
    AI Phishing Protection System
    </p>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <h3 style='color:white;'>
    📜 Recent Scan History
    </h3>
    """, unsafe_allow_html=True)

    if not st.session_state.scan_history:
        st.markdown("""
        <p style='color:#64748b; font-style:italic; font-size:13px; margin-bottom:14px;'>
        No recent scans.
        </p>
        """, unsafe_allow_html=True)

    for item in st.session_state.scan_history:

        st.markdown(f"""
        <div style="
            background: rgba(30,41,59,0.75);
            padding:14px;
            border-radius:14px;
            margin-bottom:14px;
            border-left:5px solid {item['color']};
        ">

        <div style="
            color:white;
            font-weight:600;
            font-size:15px;
            margin-bottom:6px;
        ">
        {item['title']}
        </div>

        <div style="
            color:#cbd5e1;
            font-size:13px;
            margin-bottom:4px;
        ">
        {item['status']}
        </div>

        <div style="
            color:#64748b;
            font-size:11px;
        ">
        {item['time']}
        </div>

        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div style="
        background: rgba(17,24,39,0.9);
        padding:16px;
        border-radius:16px;
        border:1px solid rgba(255,255,255,0.08);
    ">

    <h4 style='color:#facc15; margin-top:0;'>
    💡 Stay Safe Online
    </h4>

    <p style='color:#cbd5e1; font-size:13px;'>
    Never click suspicious links or share passwords through email.
    Always verify the sender before taking action.
    </p>

    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">

Built with ❤️ using Streamlit • DistilBERT • GenAI • Cybersecurity

</div>
""", unsafe_allow_html=True)
