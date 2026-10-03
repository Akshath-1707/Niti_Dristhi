import os
import pandas as pd
import numpy as np
import pdfplumber

# 1. Parse Promotion PDFs (URC-1 and URC-2)
def parse_promotion_pdf(pdf_path, urc_label):
    records = []
    if not os.path.exists(pdf_path):
        print(f"Warning: File not found -> {pdf_path}")
        return pd.DataFrame(records)
        
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if not table:
                continue
            for r in table:
                if not r or len(r) < 7:
                    continue
                raw_udise = str(r[1]).replace("\n", "").strip() if r[1] else ""
                # Extract numeric string of length >= 8
                if raw_udise.isdigit() and len(raw_udise) >= 8:
                    cluster = str(r[0]).replace("\n", " ").strip() if r[0] else ""
                    school_name = str(r[2]).replace("\n", " ").strip() if r[2] else ""
                    total_std = pd.to_numeric(str(r[3]).replace(",", "").strip(), errors="coerce") or 0
                    promoted = pd.to_numeric(str(r[4]).replace(",", "").strip(), errors="coerce") or 0
                    pending = pd.to_numeric(str(r[5]).replace(",", "").strip(), errors="coerce") or 0
                    promo_rate = str(r[6]).replace("%", "").strip()
                    finalized = str(r[-1]).replace("\n", " ").strip() if r[-1] else "NO"
                    
                    records.append({
                        "urc": urc_label,
                        "cluster_name_promo": cluster,
                        "udise_code": str(int(raw_udise)),  # Force canonical clean string
                        "school_name_promo": school_name,
                        "total_students_promo": int(total_std),
                        "promoted_students": int(promoted),
                        "pending_students": int(pending),
                        "promotion_rate": float(promo_rate) if promo_rate.replace(".", "", 1).isdigit() else 0.0,
                        "is_finalized": finalized
                    })
    return pd.DataFrame(records)

print("1. Extracting URC-1 & URC-2 Promotion Lists...")
df_u1 = parse_promotion_pdf("data/PROMOTION U1 LIST.pdf", "URC-1")
df_u2 = parse_promotion_pdf("data/PROMOTION U2 LIST.pdf", "URC-2")
df_promotions = pd.concat([df_u1, df_u2], ignore_index=True).drop_duplicates(subset=["udise_code"])
print(f"   -> Extracted {len(df_promotions)} unique school promotion records.")

# 2. Read Enrolment Excel (skipping 3 metadata rows)
excel_path = "data/class wise 1 to 12 urc1-2.xlsx"
print("2. Reading Excel Enrolment details (Row 4 header)...")
df_excel = pd.read_excel(excel_path, skiprows=3)
df_excel.columns = df_excel.columns.astype(str).str.strip()

# Drop trailing empty/summary rows
df_excel = df_excel.dropna(subset=["UDISE Code"]).copy()

# Force identical clean string representation
df_excel["udise_code"] = df_excel["UDISE Code"].apply(lambda x: str(int(float(x))) if pd.notnull(x) and str(x).replace('.','',1).isdigit() else "")

# 3. Calculate Gender Totals (Boys, Girls, and Transgender)
df_excel["total_boys"] = pd.to_numeric(df_excel["Total Boys"], errors="coerce").fillna(0).astype(int)
df_excel["total_girls"] = pd.to_numeric(df_excel["Total Girls"], errors="coerce").fillna(0).astype(int)

if "Total Trans" in df_excel.columns:
    df_excel["total_trans"] = pd.to_numeric(df_excel["Total Trans"], errors="coerce").fillna(0).astype(int)
else:
    trans_cols = [c for c in df_excel.columns if "(Trans)" in c or "Trans" in c]
    df_excel["total_trans"] = df_excel[trans_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1).astype(int)

df_excel["total_enrolment"] = df_excel["total_boys"] + df_excel["total_girls"] + df_excel["total_trans"]

# 4. Educational Cohort Volumes
primary_cols = [f"Class {i}(Total)" for i in range(1, 6) if f"Class {i}(Total)" in df_excel.columns]
upper_pri_cols = [f"Class {i}(Total)" for i in range(6, 9) if f"Class {i}(Total)" in df_excel.columns]
sec_cols = [c for c in ["Class 9(Total)", "Class 10(Total)"] if c in df_excel.columns]
hi_sec_cols = [c for c in ["Class 11(Total)", "Class 12(Total)"] if c in df_excel.columns]

df_excel["enrolment_primary"] = df_excel[primary_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1).astype(int)
df_excel["enrolment_upper_primary"] = df_excel[upper_pri_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1).astype(int)
df_excel["enrolment_secondary"] = df_excel[sec_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1).astype(int)
df_excel["enrolment_higher_secondary"] = df_excel[hi_sec_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1).astype(int)

# 5. Inclusion & Equity Ratios
df_excel["gender_parity_index"] = (df_excel["total_girls"] / df_excel["total_boys"].replace(0, 1)).round(2)
df_excel["female_enrolment_ratio"] = (df_excel["total_girls"] / df_excel["total_enrolment"].replace(0, np.nan) * 100).round(2).fillna(0)
df_excel["trans_enrolment_ratio"] = (df_excel["total_trans"] / df_excel["total_enrolment"].replace(0, np.nan) * 100).round(3).fillna(0)

# 6. Merge on string udise_code
print("3. Merging Enrolment with Promotion records on UDISE Code...")
df_merged = pd.merge(
    df_excel,
    df_promotions[[
        "udise_code", "urc", "cluster_name_promo",
        "total_students_promo", "promoted_students",
        "pending_students", "promotion_rate", "is_finalized"
    ]],
    on="udise_code",
    how="left"
)

# 7. Select Curated Features
model_cols = [
    "udise_code", "School Name", "Cluster Code", "Block Name", "urc",
    "School Management", "School Category",
    "total_enrolment", "total_boys", "total_girls", "total_trans",
    "gender_parity_index", "female_enrolment_ratio", "trans_enrolment_ratio",
    "enrolment_primary", "enrolment_upper_primary",
    "enrolment_secondary", "enrolment_higher_secondary",
    "total_students_promo", "promoted_students",
    "pending_students", "promotion_rate"
]

clean_features_df = df_merged[model_cols].rename(columns={
    "School Name": "school_name",
    "Cluster Code": "cluster_code",
    "Block Name": "block_name",
    "School Management": "school_management",
    "School Category": "school_category"
})

output_path = "data/model_school_features.csv"
clean_features_df.to_csv(output_path, index=False)

matched_promo = clean_features_df['promotion_rate'].notnull().sum()
print(f"\nSUCCESS: Generated {output_path}!")
print(f"Total Schools: {len(clean_features_df)}")
print(f"Schools successfully matched with Promotion records: {matched_promo}")
print(f"Total Enrolled Students: {clean_features_df['total_enrolment'].sum():,}")
print(f"Total Transgender Students: {clean_features_df['total_trans'].sum():,}")
