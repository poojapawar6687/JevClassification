import json
import os
import time

import pandas as pd
import streamlit as st

EMAILS_FILE = "emails.json"
RESULTS_FILE = "classified_emails.json"
LABELED_FILE = "emails_labeled.csv"
EXPORTED_FILE = "exported_output.csv"
MODEL_ID = "Mapika/decider-2b"

CATEGORIES = {
    "job_alert": "Bulk/automated job board listings and hiring notifications (e.g. beBee, Naukri, Indeed, Foundit)",
    "recruiter_outreach": "A recruiter or company individually inviting you to apply for a specific role",
    "job_application_update": "Status updates on jobs you already applied to, or recruiter interest in your profile",
    "newsletter": "Editorial digests, tech newsletters, blog content (e.g. Medium Daily Digest, Quora Digest)",
    "event_invite": "Invitations to webinars, competitions, consultations, or other events",
    "promotion": "Marketing, product announcements, sales, and content-marketing emails",
    "transactional": "Bank/broker statements, order confirmations, payment/transaction alerts",
    "account_security": "Login, verification, fraud warnings, and data-sharing alerts",
    "social_notification": "Mentions, social platform notifications (e.g. Academia.edu mentions, Reddit)",
}


@st.cache_resource
def load_model():
    from decider.infer import Decider

    return Decider(MODEL_ID, use_graphs=False)


def load_emails():
    with open(EMAILS_FILE, encoding="utf-8") as f:
        return json.load(f)


def classify_email(model, email):
    text = f"From: {email['from']}\nSubject: {email['subject']}\nSnippet: {email['snippet']}"
    result = model.decide(
        text,
        [
            {
                "question": "Which category best describes this email?",
                "options": list(CATEGORIES.keys()),
            }
        ],
    )
    answer = result[0]
    return answer["choice"], answer["confidence"]


st.set_page_config(page_title="Jev Email Classifier", layout="wide")
st.title("Email Classification — decider-2b")

emails = load_emails()
st.write(f"Loaded {len(emails)} emails from `{EMAILS_FILE}`")

with st.expander("Categories"):
    for key, desc in CATEGORIES.items():
        st.markdown(f"- **{key}**: {desc}")

if st.button("Run classification", type="primary"):
    model = load_model()
    progress = st.progress(0.0)
    status = st.empty()
    metrics_placeholder = st.empty()
    table_placeholder = st.empty()
    results = []
    run_start = time.perf_counter()
    for i, email in enumerate(emails):
        status.write(f"Classifying {i + 1}/{len(emails)}: {email['subject'][:80]}")
        start = time.perf_counter()
        choice, confidence = classify_email(model, email)
        elapsed = time.perf_counter() - start
        results.append(
            {
                "from": email["from"],
                "subject": email["subject"],
                "date": email["date"],
                "category": choice,
                "confidence": round(confidence, 4),
                "time_sec": round(elapsed, 3),
            }
        )
        progress.progress((i + 1) / len(emails))

        live_df = pd.DataFrame(results)
        with metrics_placeholder.container():
            col1, col2, col3 = st.columns(3)
            col1.metric("Elapsed", f"{time.perf_counter() - run_start:.1f} s")
            col2.metric("Avg per email", f"{live_df['time_sec'].mean():.3f} s")
            remaining = len(emails) - (i + 1)
            eta = remaining * live_df["time_sec"].mean()
            col3.metric("ETA remaining", f"{eta:.1f} s")
        table_placeholder.dataframe(live_df.iloc[::-1], use_container_width=True)

    status.write("Done.")
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    st.session_state["results"] = results

if "results" not in st.session_state and os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, encoding="utf-8") as f:
        st.session_state["results"] = json.load(f)

if "results" in st.session_state:
    df = pd.DataFrame(st.session_state["results"])

    if "time_sec" in df.columns:
        st.subheader("Timing")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total time", f"{df['time_sec'].sum():.1f} s")
        col2.metric("Avg per email", f"{df['time_sec'].mean():.3f} s")
        col3.metric("Slowest email", f"{df['time_sec'].max():.3f} s")
        st.line_chart(df["time_sec"])

    st.subheader("Category breakdown")
    st.bar_chart(df["category"].value_counts())

    st.subheader("Classified emails")
    col1, col2 = st.columns(2)
    with col1:
        selected = st.multiselect("Filter by category", options=sorted(df["category"].unique()))
    with col2:
        min_confidence = st.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)

    filtered = df
    if selected:
        filtered = filtered[filtered["category"].isin(selected)]
    filtered = filtered[filtered["confidence"] >= min_confidence]
    st.dataframe(filtered, use_container_width=True)

st.divider()
st.header("Accuracy vs labeled data")

if os.path.exists(LABELED_FILE) and os.path.exists(EXPORTED_FILE):
    labeled = pd.read_csv(LABELED_FILE)
    predicted = pd.read_csv(EXPORTED_FILE)

    # Align row-by-row (both files follow emails.json order). Merging on subject
    # duplicates rows because many emails share the same subject line.
    n = min(len(predicted), len(labeled))
    merged = predicted.iloc[:n].reset_index(drop=True).copy()
    merged = merged.rename(columns={"category": "category_pred"})
    merged["category_true"] = labeled["category"].iloc[:n].values
    if not (merged["subject"].values == labeled["subject"].iloc[:n].values).all():
        st.warning("Row order differs between predicted and labeled files; accuracy may be wrong.")

    if merged.empty:
        st.warning("No matching subjects found between predicted and labeled files.")
    else:
        merged["correct"] = merged["category_pred"] == merged["category_true"]
        accuracy = merged["correct"].mean()

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy:.1%}")
        col2.metric("Correct", int(merged["correct"].sum()))
        col3.metric("Total compared", len(merged))

        if len(predicted) != len(labeled):
            st.caption(
                f"Note: {len(predicted)} predicted rows vs {len(labeled)} labeled rows — "
                f"{len(merged)} matched by subject."
            )

        st.subheader("Per-category accuracy")
        per_cat = (
            merged.groupby("category_true")["correct"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "accuracy", "count": "n"})
            .sort_values("accuracy")
        )
        st.bar_chart(per_cat["accuracy"])
        st.dataframe(per_cat, use_container_width=True)

        st.subheader("Misclassified emails")
        wrong = merged[~merged["correct"]][["subject", "category_true", "category_pred", "confidence"]]
        st.dataframe(wrong, use_container_width=True)
else:
    missing = [f for f in (LABELED_FILE, EXPORTED_FILE) if not os.path.exists(f)]
    st.info(f"Missing file(s) for accuracy comparison: {', '.join(missing)}")
