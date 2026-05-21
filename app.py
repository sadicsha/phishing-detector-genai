"""
app.py
──────
Day 8 — Streamlit demo app.
Run: streamlit run app.py
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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Phishing Detector",
    page_icon="🎣",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.verdict-phishing {
    background: #fdecea; border-left: 4px solid #c0392b;
    padding: 16px; border-radius: 6px; margin: 10px 0;
}
.verdict-legit {
    background: #eafaf1; border-left: 4px solid #27ae60;
    padding: 16px; border-radius: 6px; margin: 10px 0;
}
.verdict-title { font-size: 24px; font-weight: bold; margin-bottom: 4px; }
.metric-row { display: flex; gap: 20px; margin: 10px 0; flex-wrap: wrap; }
.metric-box {
    background: white; border: 1px solid #e0e0e0;
    border-radius: 8px; padding: 14px 20px; min-width: 130px; text-align: center;
}
.metric-val { font-size: 22px; font-weight: bold; color: #2c3e50; }
.metric-lbl { font-size: 12px; color: #888; margin-top: 2px; }
.token-heatmap { line-height: 2.6; font-size: 15px; }
.section-title { font-size: 16px; font-weight: 600; color: #2c3e50;
                 margin: 18px 0 8px; border-bottom: 1px solid #eee; padding-bottom: 6px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar — metrics ─────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/fishing-pole.png", width=60)
    st.title("AI Phishing Detector")
    st.caption("Final Year Project · Cybersecurity + GenAI")
    st.divider()

    st.markdown("### Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("F1 Score",    "0.9928")
        st.metric("Precision",   "0.9885")
    with col2:
        st.metric("Recall",      "0.9971")
        st.metric("ROC-AUC",     "0.9998")

    st.divider()
    st.markdown("### Adversarial Eval")
    st.metric("Detection Rate", "100%", "798/798 caught")

    st.divider()
    st.markdown("### Dataset")
    st.markdown("- 1,500 real phishing emails\n- 1,500 real legitimate emails\n- 798 AI-generated phishing")
    st.divider()
    st.caption("Model: DistilBERT fine-tuned\nGenerator: Llama-3.1 via Groq")

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource
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

# ── Token heatmap renderer ────────────────────────────────────────────────────
def render_token_heatmap(tokens, scores):
    """Render inline token heatmap using HTML spans."""
    words, word_scores = [], []
    cw, cs, cnt = "", 0.0, 0
    for t, s in zip(tokens, scores):
        if t.startswith("##"):
            cw += t[2:]; cs += s; cnt += 1
        else:
            if cw: words.append(cw); word_scores.append(cs / max(cnt,1))
            cw, cs, cnt = t, s, 1
    if cw: words.append(cw); word_scores.append(cs / max(cnt,1))

    def color(s):
        if s > 0.15:  return f"rgba(220,60,60,{min(s*0.85,0.85):.2f})"
        if s < -0.15: return f"rgba(60,130,220,{min(-s*0.7,0.7):.2f})"
        return "transparent"

    spans = ""
    for w, s in zip(words, word_scores):
        bg    = color(s)
        title = f"Score: {s:.3f}"
        spans += (f'<span title="{title}" style="background:{bg};padding:2px 5px;'
                  f'border-radius:3px;margin:1px 2px;display:inline-block;">{w}</span> ')

    html = f"""
    <div class="token-heatmap">{spans}</div>
    <div style="display:flex;gap:20px;margin-top:12px;font-size:12px;color:#666;">
      <span>🔴 Strong phishing signal</span>
      <span>🔵 Legit signal</span>
      <span>⬜ Neutral</span>
    </div>
    """
    return html

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Detect", "⚡ Generate", "📊 Examples"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — DETECT
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## Phishing Email Detector")
    st.markdown("Paste any email below to classify it and see which words drove the prediction.")

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        subject_input = st.text_input(
            "Subject line",
            placeholder="e.g. URGENT: Your account has been suspended",
            key="detect_subject"
        )
        body_input = st.text_area(
            "Email body",
            placeholder="Paste the full email body here...",
            height=220,
            key="detect_body"
        )

        # Pre-loaded examples
        st.markdown('<div class="section-title">Quick examples</div>', unsafe_allow_html=True)
        ex_col1, ex_col2 = st.columns(2)
        with ex_col1:
            if st.button("Load phishing example", use_container_width=True):
                st.session_state["detect_subject"] = "URGENT: Verify your PayPal account"
                st.session_state["detect_body"] = (
                    "Dear Customer,\n\nWe have detected unusual activity on your PayPal account. "
                    "Your account has been temporarily suspended for security reasons.\n\n"
                    "Click here IMMEDIATELY to verify your identity and restore access:\n"
                    "http://paypa1-secure-verify.com/restore\n\n"
                    "Failure to verify within 24 hours will result in permanent account closure.\n\n"
                    "PayPal Security Team"
                )
                st.rerun()
        with ex_col2:
            if st.button("Load legit example", use_container_width=True):
                st.session_state["detect_subject"] = "Team lunch this Friday"
                st.session_state["detect_body"] = (
                    "Hi everyone,\n\nJust a reminder that we have our quarterly team lunch "
                    "this Friday at 12:30pm at the usual place on Main Street.\n\n"
                    "Please let me know if you have any dietary requirements by Thursday.\n\n"
                    "Looking forward to seeing everyone!\n\nBest,\nSarah"
                )
                st.rerun()

        analyse_btn = st.button("🔍 Analyse Email", type="primary", use_container_width=True)

    with col_r:
        if analyse_btn:
            if not subject_input and not body_input:
                st.warning("Please enter a subject or body to analyse.")
            else:
                with st.spinner("Analysing..."):
                    try:
                        predict_email = load_predictor()
                        result = predict_email(
                            subject=subject_input or "",
                            body=body_input or ""
                        )

                        is_phishing = result["label"] == 1
                        verdict     = result["verdict"]
                        conf        = result["confidence"]

                        # Verdict box
                        box_class = "verdict-phishing" if is_phishing else "verdict-legit"
                        icon      = "🚨" if is_phishing else "✅"
                        color     = "#c0392b" if is_phishing else "#27ae60"
                        st.markdown(f"""
                        <div class="{box_class}">
                          <div class="verdict-title" style="color:{color}">{icon} {verdict}</div>
                          <div style="font-size:14px;color:#555;">
                            Confidence: <strong>{conf*100:.1f}%</strong>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Top tokens
                        st.markdown('<div class="section-title">Top signals</div>',
                                    unsafe_allow_html=True)
                        top = result["top_tokens"][:8]
                        for token, score in top:
                            bar_color = "#c0392b" if score > 0 else "#2980b9"
                            bar_width = min(abs(score) * 100, 100)
                            direction = "Phishing" if score > 0 else "Legit"
                            st.markdown(f"""
                            <div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
                              <span style="font-family:monospace;min-width:120px;font-size:13px;">
                                {token}</span>
                              <div style="background:#f0f0f0;border-radius:4px;
                                          height:14px;flex:1;overflow:hidden;">
                                <div style="background:{bar_color};height:100%;
                                            width:{bar_width:.0f}%;border-radius:4px;
                                            opacity:0.75;"></div>
                              </div>
                              <span style="font-size:11px;color:#888;min-width:50px;">
                                {direction}</span>
                            </div>
                            """, unsafe_allow_html=True)

                        # Heatmap
                        st.markdown('<div class="section-title">Token heatmap</div>',
                                    unsafe_allow_html=True)
                        heatmap_html = render_token_heatmap(
                            result["tokens"], result["scores"]
                        )
                        st.markdown(heatmap_html, unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Error: {e}")
                        import traceback; st.code(traceback.format_exc())

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — GENERATE
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## Phishing Email Generator")
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

        gen_btn = st.button("⚡ Generate Phishing Email", type="primary",
                            use_container_width=True)

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
            st.text_input("Subject", value=st.session_state.get("gen_subject",""),
                          key="gen_subj_display")
            st.text_area("Body", value=st.session_state.get("gen_body",""),
                         height=280, key="gen_body_display")

            st.caption("⚠️ This is an AI-generated synthetic email for research purposes only.")

            if st.button("🔍 Detect this email", use_container_width=True):
                with st.spinner("Analysing..."):
                    try:
                        predict_email = load_predictor()
                        result = predict_email(
                            st.session_state.get("gen_subject",""),
                            st.session_state.get("gen_body","")
                        )
                        is_p = result["label"] == 1
                        icon = "🚨" if is_p else "✅"
                        col = "#c0392b" if is_p else "#27ae60"
                        st.markdown(f"""
                        <div class="{'verdict-phishing' if is_p else 'verdict-legit'}">
                          <div class="verdict-title" style="color:{col}">
                            {icon} {result['verdict']} — {result['confidence']*100:.1f}%
                          </div>
                        </div>""", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(str(e))
        else:
            st.markdown("""
            <div style="background:#f8f9fa;border-radius:8px;padding:40px;
                        text-align:center;color:#aaa;margin-top:20px;">
              <div style="font-size:40px">⚡</div>
              <div style="margin-top:8px">Generated email will appear here</div>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — EXAMPLES
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## Pre-loaded Example Analyses")
    st.markdown("These examples show the model detecting different types of phishing.")

    examples = [
        {
            "title": "Invoice Fraud (Synthetic)",
            "label": "Phishing",
            "subject": "Action Required: Unpaid Invoice #INV-2024-8821",
            "body": "Dear Valued Customer, Our records show an outstanding invoice of $847.50 that remains unpaid. Immediate action is required to avoid service suspension and late fees. Please review and pay your invoice within 24 hours by clicking the secure link below: http://amazon-invoice-alert.com/pay/INV-2024-8821 Failure to act immediately will result in account suspension and referral to collections. Amazon Billing Department",
        },
        {
            "title": "IT Security Alert (Synthetic)",
            "label": "Phishing",
            "subject": "URGENT: Your password expires in 2 hours",
            "body": "Dear Employee, Our security system has detected that your corporate password will expire in 2 hours. If you do not reset your password immediately, you will lose access to all company systems including email, VPN, and internal tools. Click here NOW to reset your password: http://it-helpdesk-alert.com/reset Ignoring this message will result in immediate account lockout. IT Security Team",
        },
        {
            "title": "Team Lunch Invite (Legitimate)",
            "label": "Legitimate",
            "subject": "Team lunch this Friday at 12:30",
            "body": "Hi everyone, Just a quick note to confirm our team lunch this Friday at 12:30pm at Bella Italia on High Street. We have a reservation for 8 people. Please let me know by Thursday if you can make it or if you have any dietary requirements. Looking forward to seeing everyone! Best, Sarah",
        },
        {
            "title": "Meeting Notes (Legitimate)",
            "label": "Legitimate",
            "subject": "Notes from today's sprint planning",
            "body": "Hi team, Please find attached the notes from today's sprint planning session. Key decisions: 1) We agreed to prioritise the API integration for next sprint. 2) The design review is moved to Wednesday at 3pm. 3) John will lead the customer demo next Friday. Please review and add any corrections by EOD tomorrow. Thanks, Mike",
        },
    ]

    for ex in examples:
        with st.expander(f"{'🚨' if ex['label']=='Phishing' else '✅'} {ex['title']} — {ex['label']}"):
            st.markdown(f"**Subject:** {ex['subject']}")
            st.markdown(f"**Body:**\n\n{ex['body']}")
            if st.button(f"Analyse: {ex['title']}", key=f"ex_{ex['title']}"):
                with st.spinner("Analysing..."):
                    try:
                        predict_email = load_predictor()
                        result = predict_email(ex["subject"], ex["body"])
                        is_p   = result["label"] == 1
                        icon   = "🚨" if is_p else "✅"
                        col    = "#c0392b" if is_p else "#27ae60"
                        correct = result["label"] == (1 if ex["label"]=="Phishing" else 0)
                        st.markdown(f"""
                        <div class="{'verdict-phishing' if is_p else 'verdict-legit'}">
                          <div class="verdict-title" style="color:{col}">
                            {icon} {result['verdict']} ({result['confidence']*100:.1f}%)
                            {'✓ Correct' if correct else '✗ Wrong'}
                          </div>
                        </div>""", unsafe_allow_html=True)

                        hm = render_token_heatmap(result["tokens"], result["scores"])
                        st.markdown(hm, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(str(e))
