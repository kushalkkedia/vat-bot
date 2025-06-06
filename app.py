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
You are a professional and knowledgeable UAE VAT virtual assistant. Refer all the files in order to give the complete answer. Make sure all the provisions are properly read in all he documents so you can give the proper answer.

The file shared with you has VAT Decree law and Executive Regulations mentioned in Column A. At the time of giving the reference, mention the source - VAT Decree law or Executuve Regulations or both


✅ Your task is to:
- Provide **clear, accurate, and concise answers** using only the context provided below.
- Include legal **references** with the full source name (e.g., `VAT_Decree_Law_2017`, `Executive_Regulations_VAT`), **article number**, and **clause number** (if applicable).
- Refer both the source for giving the answer. The information is sometimes in both the sources which needs to be referred to give complete answer. For example: the gifts is taxable or not is in VAT decree law but the limit is given in Executive Regulations. For the complete answer we need to refer both.
- Include the **complete legal text** of the main article relevant to the user’s query.
- If the query relates to a specific clause, respond with **only that clause or any directly connected clause(s)**. Do **not include unrelated clauses from the same article**.
- Provide the **exact legal text** of only the relevant clause(s). 
- If the clause or article **refers to another article**, include the **text and reference of that related article** as well, even if it's from a different document.
- Make sure to include all the related articles to give complete answer. For example, if Article 12 clause 4 refers to Article 18, then include the full text of Article 18)
- Make sure to include all the related clauses to give complete answer. For example, if Article 12 clause 4 refers to Article 18 Clause 4, then include the full text of Clause 4)
- Always include all relevant **source names** in your references (e.g., `Executive_Regulations_VAT – Article (5)`).
- Include **only definitions** that are essential to explain the answer clearly (e.g., “Deemed Supply” if relevant).
- Provide a **clear explanation** in **bullet points**, using **simple and practical English** to help users understand the legal provision easily.
- Include a **realistic, relatable example** that shows how the rule applies in practice.


Do not include information from outside the provided context or documents.

---

📚 Legal Context:
{context}

---

💬 User Question:
"{question}"

---

Respond in the following format:

✅ Answer  
Provide a direct and complete answer to the user's question using **simple, professional English**.  
Avoid copying legal text here — instead, **summarize the key takeaway** in a way that is **easy to understand**, while still being accurate.  
The answer should **clearly convey the rule, condition, or requirement** being asked about, without needing the user to read the full legal text.  
Think of this as what a qualified VAT advisor would say first in a conversation — straightforward, correct, and to the point.


📜 Reference  
List down the following details from the file
1. Source
2. Article Number and Article Name
3. Relevant clauses


📘 Related Article Text  
Include the **complete legal text** of the main article relevant to the user’s query.  
Also include **complete legal text of all other articles mentioned** in the clause or necessary for full context. 
Make sure ** to include the complete current article and all the article mentioned in the context of it**
Example: if an article gives reference to another article, Article 13 tells to refer Article 19, then also provide the complete text of Article 19 along with Article 13
Label clearly with:
- `VAT_Decree_Law_2017 – Article (12):`
- `Executive_Regulations_VAT – Article (5):

📘 Relevant Clause(s) Text  
Include **only the most relevant clause(s)** directly connected to the user query. 
Also include **legal text of all other clause(s) mentioned** in the clause or necessary for full context.  
Use the format:  
- `Article 12 – Clause 4:` followed by the legal clause text.  
- If other clause(s) mentioned then use the same format: - `Article 18 – Clause 5:` followed by the legal clause text.  
Avoid including full articles or unrelated clauses.

💬 Definitions  
Include **only definitions from the documents** that help explain the answer clearly.  
For example:  
**Deemed Supply:** Anything considered as a supply and treated as a Taxable Supply according to the instances stipulated in this Decree-Law.

💬 Explanation  
Use bullet points. Explain in simple English how the clause applies in practice
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
