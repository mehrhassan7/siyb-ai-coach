import os
import json
import re
import streamlit as st
from groq import Groq


# ================================
# 1. GROQ CLIENT
# ================================
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# ================================
# 2. LOAD GYB CHUNKS
# ================================
def _tokenize(text):
    return set(re.findall(r"\w+", text.lower()))

GYB_SNIPPETS = []
if os.path.exists("gyb_chunks.json"):
    with open("gyb_chunks.json", "r", encoding="utf-8") as f:
        GYB_SNIPPETS = json.load(f)

for snip in GYB_SNIPPETS:
    snip["tokens"] = _tokenize(snip["content"])


def get_relevant_chunks(query, k=3):
    if not query or not GYB_SNIPPETS:
        return []
    q = _tokenize(query)
    scored = []
    for s in GYB_SNIPPETS:
        overlap = len(q & s["tokens"])
        scored.append((overlap, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for score, s in scored if score > 0][:k]


def looks_like_question(text):
    if not text:
        return False
    t = text.lower().strip()
    if t.endswith("?"):
        return True
    return t.startswith(("how", "what", "why", "when", "where", "who", "can", "could", "should"))


# ================================
# 3. LLM FEEDBACK
# ================================
def llm_feedback(stage, user_answer, gyb_data):
    stage_instruction = {
        "ask_background":
            "Appreciate the learner’s background and say why skills/experience help in choosing a business idea.",
        "ask_idea":
            "Summarize the idea in 3–4 bullet points, give one strength and one improvement point.",
        "ask_customers":
            "Check if the customer group is specific. Suggest 2–3 improvements.",
        "ask_competitors":
            "Give 2 simple suggestions on how to stand out vs competitors.",
        "ask_location":
            "Explain (briefly) why location matters and what to observe.",
        "general":
            "Give simple, friendly entrepreneurship advice in 4–6 sentences.",
    }.get(stage, "Give friendly advice.")

    chunks = get_relevant_chunks(user_answer, k=3)
    if chunks:
        context = "\n\n".join([f"- {c['content'][:300]}..." for c in chunks])
    else:
        context = "No relevant chunks found. Give general GYB advice."

    system_msg = {
        "role": "system",
        "content": (
            "You are a friendly SIYB GYB coach.\n"
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
            "Write a simple SIYB-style business idea summary including:\n"
            "1) Idea title\n2) One-line description\n"
            "3) Main customers\n4) Problem solved\n"
            "5) Why idea fits this location\n6) 3 simple next steps."
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
