import os
import json
import re
import streamlit as st
from groq import Groq
from rank_bm25 import BM25Okapi



# ================================
# 1. GROQ CLIENT
# ================================
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# ================================
# 2. LOAD GYB CHUNKS
# ================================
def _tokenize(text: str):
    """Very simple word tokenizer for retrieval (list of tokens)."""
    return re.findall(r"\w+", text.lower())



GYB_SNIPPETS = []
gyb_path = "gyb_chunks.json"

if os.path.exists(gyb_path):
    with open(gyb_path, "r", encoding="utf-8") as f:
        GYB_SNIPPETS = json.load(f)

# Build BM25 index over all chunk contents
BM25_INDEX = None
CORPUS_TOKENS = []

if GYB_SNIPPETS:
    for snippet in GYB_SNIPPETS:
        tokens = _tokenize(snippet.get("content", ""))
        snippet["tokens"] = tokens  # keep if you want for debugging
        CORPUS_TOKENS.append(tokens)

    BM25_INDEX = BM25Okapi(CORPUS_TOKENS)


def get_relevant_snippets(query: str, k: int = 3, min_score: float = 0.5):
    """
    BM25-based retrieval:
    - Tokenize user query
    - Use BM25Okapi to score chunks
    - Return top-k chunks with score above min_score
    """
    if not query or not GYB_SNIPPETS or BM25_INDEX is None:
        return []

    query_tokens = _tokenize(query)
    scores = BM25_INDEX.get_scores(query_tokens)

    # Sort chunks by score (highest first)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in ranked:
        if score < min_score:
            # below threshold → ignore
            continue
        snippet = dict(GYB_SNIPPETS[idx])  # copy
        snippet["score"] = float(score)
        results.append(snippet)
        if len(results) >= k:
            break

    return results

# ================================
# Helper: detect if user is asking a question
# ================================
def looks_like_question(text: str) -> bool:
    """Very simple heuristic to detect a user question."""
    if not text:
        return False
    t = text.strip().lower()
    if t.endswith("?"):
        return True
    starters = (
        "how", "what", "why", "when", "where", "who", "which",
        "can ", "could ", "should ", "is ", "are ", "do ", "does ",
    )
    return any(t.startswith(s) for s in starters)



# ================================
# 3. LLM FEEDBACK
# ================================
def llm_feedback(stage, user_answer, gyb_data):
    ...
    # Build a richer retrieval query using the business idea + stage
    idea = gyb_data.get("idea", "")
    retrieval_query = f"Stage: {stage}. Idea: {idea}. Learner text: {user_answer}"

    chunks = get_relevant_snippets(retrieval_query, k=3)


    if chunks:
        context = "\n\n".join([f"- {c['content'][:300]}..." for c in chunks])
    else:
        context = "No relevant chunks found. Give general GYB advice."

system_msg = {
    "role": "system",
    "content": (
        "You are a friendly SIYB GYB (Generate Your Business Idea) coach from "
        "the ILO Start and Improve Your Business (SIYB) programme.\n"
        "Use ONLY generic SIYB-style language. Do NOT invent or mention:\n"
        "- external book titles or authors,\n"
        "- other training programmes,\n"
        "- fake expansions of GYB like 'Growth for Youth in Business'.\n"
        "Just speak as a neutral SIYB trainer.\n\n"
        f"{stage_instruction}\n\n"
        "Relevant GYB manual text:\n" + context
    )
}


    user_msg = {"role": "user", "content": user_answer}

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[system_msg, user_msg],
        temperature=0.7,
    )
    return resp.choices[0].message.content


# ================================
# 4. SUMMARY GENERATION
# ================================
def generate_summary(gyb):
    ctx = f"""
Background: {gyb.get('background','')}
Idea: {gyb.get('idea','')}
Customers: {gyb.get('customers','')}
Competitors: {gyb.get('competitors','')}
Location: {gyb.get('location','')}
"""
    system_msg = {
        "role": "system",
        "content": (
            "Write a simple SIYB-style business idea summary. "
"Do NOT mention any specific books, authors, or external programmes. "
"Just speak as a neutral SIYB trainer.\n"
"Include:\n"
"1) Idea title\n2) One-line description\n..."

        )
    }
    user_msg = {"role": "user", "content": ctx}

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[system_msg, user_msg],
        temperature=0.7,
    )
    return resp.choices[0].message.content


# ================================
# 5. STREAMLIT APP
# ================================
def main():
    st.title("SIYB GYB Coach – Clean Version")

    # Init session
    if "stage" not in st.session_state:
        st.session_state.stage = "ask_background"
        st.session_state.messages = []
        st.session_state.data = {}
        st.session_state.summary = None
        st.session_state.last_question = (
            "First, tell me about yourself — skills, experience, situation?"
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": "Assalam o alaikum! 👋\n\n"
             + st.session_state.last_question}
        )

    finished = st.session_state.stage == "finished"
    user_input = st.chat_input(
        "Ask or answer…" if finished else "Answer the coach or ask a side question…"
    )

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        stage = st.session_state.stage
        data = st.session_state.data

        # Hybrid Q&A
        if stage != "finished" and looks_like_question(user_input):
            reply = llm_feedback("general", user_input, data)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.session_state.messages.append(
                {"role": "assistant", "content": st.session_state.last_question}
            )

        else:
            # Guided flow
            if stage == "ask_background":
                data["background"] = user_input
                fb = llm_feedback(stage, user_input, data)
                st.session_state.messages.append({"role": "assistant", "content": fb})

                q = "Great — now describe one business idea you have?"
                st.session_state.messages.append({"role": "assistant", "content": q})
                st.session_state.last_question = q
                st.session_state.stage = "ask_idea"

            elif stage == "ask_idea":
                data["idea"] = user_input
                fb = llm_feedback(stage, user_input, data)
                st.session_state.messages.append({"role": "assistant", "content": fb})

                q = "Nice. Who are your main customers?"
                st.session_state.messages.append({"role": "assistant", "content": q})
                st.session_state.last_question = q
                st.session_state.stage = "ask_customers"

            elif stage == "ask_customers":
                data["customers"] = user_input
                fb = llm_feedback(stage, user_input, data)
                st.session_state.messages.append({"role": "assistant", "content": fb})

                q = "Now tell me about your competitors."
                st.session_state.messages.append({"role": "assistant", "content": q})
                st.session_state.last_question = q
                st.session_state.stage = "ask_competitors"

            elif stage == "ask_competitors":
                data["competitors"] = user_input
                fb = llm_feedback(stage, user_input, data)
                st.session_state.messages.append({"role": "assistant", "content": fb})

                q = "Where will you run this business (home, shop, online)?"
                st.session_state.messages.append({"role": "assistant", "content": q})
                st.session_state.last_question = q
                st.session_state.stage = "ask_location"

            elif stage == "ask_location":
                data["location"] = user_input
                fb = llm_feedback(stage, user_input, data)
                st.session_state.messages.append({"role": "assistant", "content": fb})

                summary = generate_summary(data)
                st.session_state.summary = summary
                st.session_state.messages.append(
                    {"role": "assistant", "content": "Here is your SIYB summary below:"}
                )
                st.session_state.stage = "finished"

            else:
                # After finished — normal Q&A
                fb = llm_feedback("general", user_input, data)
                st.session_state.messages.append({"role": "assistant", "content": fb})

    # Render chat
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if st.session_state.summary:
        st.subheader("📄 Your Business Idea Summary")
        st.markdown(st.session_state.summary)


if __name__ == "__main__":
    main()
