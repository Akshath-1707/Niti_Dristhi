import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# 1. Load API Key from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key or "YourActualKey" in api_key:
    raise ValueError("GEMINI_API_KEY is missing or invalid in your .env file!")

# 2. Extract Real Indicators from data/database_ready_schools_urc.csv
csv_path = os.path.join("data", "database_ready_schools_urc.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Could not find dataset at '{csv_path}'")

print("=" * 60)
print(f"Reading empirical dataset: {csv_path}")
print("=" * 60)
df = pd.read_csv(csv_path)

total_schools = len(df)
active_schools = int(df["is_operational"].sum()) if "is_operational" in df.columns else total_schools
total_students = int(df["total_enrolment"].sum())
total_boys = int(df["total_boys"].sum())
total_girls = int(df["total_girls"].sum())
promo_avg = round(float(df[df["total_students_promo"] > 0]["promotion_rate"].mean()), 2)
pending_students = int(df["pending_students"].sum())

# Simulated readiness index (combining promotion rate and gender balance)
readiness_score = round(promo_avg * 0.75 + (total_girls / max(total_boys, 1) * 100) * 0.25, 1)

metrics = {
    "Total Schools (URC-1 & URC-2)": f"{total_schools} (Active: {active_schools})",
    "Total Student Enrolment": f"{total_students:,}",
    "Gender Breakdown": f"Boys: {total_boys:,} | Girls: {total_girls:,}",
    "Primary Enrolment (Classes 1-5)": f"{int(df['enrolment_primary'].sum()):,}",
    "Upper Primary Enrolment (Classes 6-8)": f"{int(df['enrolment_upper_primary'].sum()):,}",
    "Secondary Enrolment (Classes 9-10)": f"{int(df['enrolment_secondary'].sum()):,}",
    "Higher Secondary Enrolment (Classes 11-12)": f"{int(df['enrolment_higher_secondary'].sum()):,}",
    "Average Promotion Rate": f"{promo_avg}%",
    "Students with Pending Promotion": f"{pending_students:,}"
}

for k, v in metrics.items():
    print(f"  * {k}: {v}")

# 3. Retrieve Context from Local FAISS Vector Store
vectorstore_path = os.path.join("backend", "rag", "vectorstore")
print("\n" + "=" * 60)
print(f"Querying local FAISS index: '{vectorstore_path}'...")
print("=" * 60)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

query = "pupil teacher ratio RTE norms remedial training dropout curtailing secondary transition infrastructure"
retrieved_docs = retriever.invoke(query)

policy_context = "\n\n---\n\n".join([
    f"[{doc.metadata.get('source', 'PDF')} | Page {doc.metadata.get('page', 'N/A')}]:\n{doc.page_content.strip()}"
    for doc in retrieved_docs
])

print(f"Retrieved {len(retrieved_docs)} policy passages directly from your PDFs.")

# 4. Prompt Assembly & Gemini Generation
prompt_template = PromptTemplate.from_template(
"""You are an executive urban policy advisor for the NITI Drishti decision-support system.
Synthesize the real empirical infrastructure metrics for Chhatrapati Sambhajinagar against the statutory government education frameworks and produce an actionable policy recommendation brief.

### Real Empirical Data Audit:
- Sector: Education (URC-1 & URC-2 Urban Core)
- Infrastructure Readiness Score: {readiness_score} / 100
- Granular Indicators:
{metrics}

### Retrieved Statutory Policy Guidelines:
{policy_context}

### Required Deliverable:
Produce a structured, professional policy recommendation report:
1. Executive Diagnosis: Direct analysis of the cohort progression, secondary stage bottlenecks, and pending promotion backlogs referencing the numbers.
2. Targeted Policy Directives: 3 actionable interventions directly citing the retrieved statutory norms (RTE, NEP 2020, Samagra Shiksha, PM SHRI).
3. Phased Implementation Roadmap:
   - Immediate (0 to 6 Months)
   - Medium-Term (6 to 24 Months)
   - Long-Term (2037 Vision Target)
"""
)

metrics_formatted = "\n".join([f"- {k}: {v}" for k, v in metrics.items()])
formatted_prompt = prompt_template.format(
    readiness_score=readiness_score,
    metrics=metrics_formatted,
    policy_context=policy_context
)

print("\n" + "=" * 60)
print("Synthesizing policy brief using Gemini 2.5 Flash...")
print("=" * 60)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key,
    temperature=0.2
)

response = llm.invoke(formatted_prompt)

print("\n" + "=" * 70)
print("GENERATED NITI DRISHTI POLICY REPORT")
print("=" * 70)
print(response.content)

output_file = "generated_policy_report.md"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(response.content)
print(f"\n[SUCCESS] Report written to '{output_file}'.")