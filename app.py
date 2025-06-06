import streamlit as st
import openai
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(page_title="UAE VAT Companion", page_icon="🧾", layout="wide")

# Header section
st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1;'>🧾 UAE VAT Companion</h1>
    <p style='text-align: center; font-size: 18px;'>Your trusted UAE VAT companion.</p>
    <hr>
    """,
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.header("About this tool")
    st.info("Get UAE VAT answers backed by legal references from Decree-Law & Executive Regulations.")
    st.markdown("**Examples:**")
    st.markdown("- What is the VAT on free samples?\n- What is deemed supply?\n- Is VAT applicable on imported goods for repair and re-export?")


# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load data
@st.cache_resource

def load_embeddings():
    def load_and_prepare(path, source_name):
        df = pd.read_pickle(path)
        df = df[df["embedding"].notna()].copy()
        df["embedding"] = df["embedding"].apply(lambda x: np.array(x))
        df["source"] = source_name
        return df

    df_law = load_and_prepare("vat_law_exe_embeddings.pkl", "VAT Decree Law or Executive Regulations")
    return pd.concat([df_law], ignore_index=True)

with st.spinner("Loading VAT legal data..."):
    df = load_embeddings()

# Input box
question = st.text_area("💬 Ask your VAT question:", placeholder="e.g., What is the VAT treatment of import of goods for repair and re-export?")

if question:
    with st.spinner("Generating answer..."):
        question_embedding = openai.embeddings.create(
            input=[question], model="text-embedding-3-small"
        ).data[0].embedding
        question_embedding = np.array(question_embedding)

        df["similarity"] = df["embedding"].apply(lambda x: cosine_similarity([x], [question_embedding])[0][0])
        top_chunks = df.sort_values("similarity", ascending=False).head(10)

        context = ""
        for _, row in top_chunks.iterrows():

            context += f"""
📜 Reference: {row['source']}  
📘 Title {row.get('title_number', '')}: {row.get('title_name', '')}  
📘 Chapter {row.get('chapter_number', '')}: {row.get('chapter_name', '')}  
📘 Article {row.get('article_number', '')}: {row.get('article_name', '')}  
📘 Clause {row.get('clause_number', '')}:  
{row.get('clause_text', row.get('text', ''))}

---
"""

        # Prepare prompt
        prompt = f"""
You are a professional and knowledgeable UAE VAT virtual companion.
The VAT Decree Law and Executive Regulations have been combined into a single reference file. Before providing an answer, ensure you thoroughly review and refer to all related articles and clauses from both the VAT Decree Law and the Executive Regulations.

Your response must:

Include all relevant legal provisions necessary to give the user a complete and accurate answer.

Mention the corresponding article numbers, clause numbers, and their full legal text.

Ensure that any interlinked articles or cross-referenced clauses are also included to provide proper context and clarity.

Deliver a response that is legally thorough yet easy to understand for the user.
"""

        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        result = response.choices[0].message.content
        st.success("✅ Here’s your answer:")
        st.markdown(result)

        # Feedback section
        with st.expander("💬 Was this answer helpful?"):
            feedback = st.radio("Please let us know:", ["👍 Yes", "👎 No"], horizontal=True)
            if feedback == "👎 No":
                issue = st.text_area("What could be improved? (Optional)")
                if st.button("Submit Feedback"):
                    st.success("Thank you! Your feedback helps us improve.")
