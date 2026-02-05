# SPDX-License-Identifier: MIT
# timesheet-audit-app: Streamlit web app for quarterly timesheet checks

import io
import re
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Quarterly Timesheet Auditor", layout="wide")

# ----------------------------- Helpers --------------------------------- #

def normalize_text(x: str) -> str:
    """Case-insensitive, trim spaces, normalize en/em dashes to hyphen."""
    if pd.isna(x):
        return ""
    s = str(x)
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def to_date(series, errors="coerce"):
    return pd.to_datetime(series, errors=errors).dt.normalize()

def to_numeric(series):
    return pd.to_numeric(series, errors="coerce")

def style_overlimit(df: pd.DataFrame, col_name: str):
    """Return a Styler with red bold for positive 'Over the Limit (hrs)'."""
    def hilite(s):
        return [
            "color: #b60000; font-weight: 700" if (col_name in s.index and pd.notna(s[col_name]) and float(s[col_name]) > 0)
            else "" for _ in s
        ]
    styler = df.style.apply(hilite, axis=1)
    return styler

def pick_sheet_ui(label: str, file):
    """Let user pick a sheet from an uploaded Excel file."""
    with pd.ExcelFile(file) as xls:
        sheet_names = xls.sheet_names
    return st.selectbox(label, options=sheet_names)

def map_columns_ui(df: pd.DataFrame, mapping_label: str, required: Dict[str, str]) -> Dict[str, str]:
    """
    Ask the user to map current df columns to required semantic names.
    `required` = {"Resource": "", "Project Code": "", ...}
    """
    st.markdown(f"**Map columns – {mapping_label}**")
    cols_map = {}
    options = ["(not set)"] + list(df.columns)
    columns = st.columns(len(required))
    for i, req_col in enumerate(required.keys()):
        with columns[i]:
            pick = st.selectbox(req_col, options=options, key=f"{mapping_label}-{req_col}")
            if pick == "(not set)":
                cols_map[req_col] = None
            else:
                cols_map[req_col] = pick
    # Validate
    missing = [k for k, v in cols_map.items() if not v]
    if missing:
        st.warning(f"Select columns for: {', '.join(missing)}")
    return cols_map

def compute_quarter_ranges(plan_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per resource, use min(Start Date) and max(End Date) from plan as the "quarter window".
    """
    res_ranges = plan_df.groupby("res_key", as_index=False).agg(
        Quarter_Start=("Start Date", "min"),
        Quarter_End=("End Date", "max"),
        Planned_Total=("Planned Total Hours", "sum"),
    )
    return res_ranges

def build_reports(plan_df: pd.DataFrame, act_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build (A) assignment-level, (B) resource-level, (C) mismatches dataframes.
    Assumptions:
    - plan_df has semantic columns already standardized.
    - act_df same.
    """
    # Normalize keys
    for df in (plan_df, act_df):
        df["res_key"] = df["Resource"].astype(str).str.strip()
        df["proj_key"] = df["Project Code"].map(normalize_text)
        df["task_key"] = df["Task code"].map(normalize_text)

    # Quarter window per resource
    res_ranges = compute_quarter_ranges(plan_df.rename(columns={"Total Hours": "Planned Total Hours"}))
    act_df = act_df.merge(res_ranges[["res_key", "Quarter_Start", "Quarter_End"]], on="res_key", how="left")
    act_df["In_Quarter"] = (act_df["Date"] >= act_df["Quarter_Start"]) & (act_df["Date"] <= act_df["Quarter_End"])
    act_in_q = act_df[act_df["In_Quarter"].fillna(False)].copy()

    # Match actuals to specific assignments (resource+project+task + within Start/End)
    plan_assign = plan_df[[
        "res_key", "proj_key", "task_key", "Start Date", "End Date", "Resource", "Project Code", "Task code", "Planned Total Hours"
    ]].rename(columns={"Start Date": "StartDate", "End Date": "EndDate"}).copy()

    merged = act_in_q.merge(plan_assign, on=["res_key", "proj_key", "task_key"], how="left", suffixes=("", "_plan"))
    merged["Within_Assignment"] = (merged["Date"] >= merged["StartDate"]) & (merged["Date"] <= merged["EndDate"])
    matched = merged[merged["Within_Assignment"].fillna(False)].copy()

    # Assignment-level actuals
    assign_actual = matched.groupby(
        ["res_key", "proj_key", "task_key", "StartDate", "EndDate", "Resource", "Project Code", "Task code"],
        dropna=False
    ).agg(Actual_Hours=("Total Hours", "sum")).reset_index()

    assignment = plan_assign.merge(assign_actual[
        ["res_key", "proj_key", "task_key", "StartDate", "EndDate", "Actual_Hours"]
    ], on=["res_key", "proj_key", "task_key", "StartDate", "EndDate"], how="left")

    assignment["Actual_Hours"] = assignment["Actual_Hours"].fillna(0.0)
    assignment["Variance (Actual - Planned)"] = assignment["Actual_Hours"] - assignment["Planned Total Hours"]
    assignment["Over the Limit (hrs)"] = assignment["Variance (Actual - Planned)"].clip(lower=0)
    assignment["Remaining (hrs)"] = (-assignment["Variance (Actual - Planned)"]).clip(lower=0)
    assignment["Within Limit?"] = np.where(assignment["Over the Limit (hrs)"] > 0, "No", "Yes")

    assignment_view = assignment.rename(columns={
        "StartDate": "Start Date", "EndDate": "End Date", "Actual_Hours": "Actual Hours"
    })[
        ["Resource", "Project Code", "Task code", "Start Date", "End Date",
         "Planned Total Hours", "Actual Hours", "Variance (Actual - Planned)",
         "Over the Limit (hrs)", "Remaining (hrs)", "Within Limit?"]
    ].sort_values(["Resource", "Project Code", "Task code", "Start Date"])

    # Resource-level
    resource = assignment_view.groupby("Resource", as_index=False).agg(
        Planned_Total_Hours=("Planned Total Hours", "sum"),
        Actual_Total_Hours=("Actual Hours", "sum")
    )
    resource["Variance (Actual - Planned)"] = resource["Actual_Total_Hours"] - resource["Planned_Total_Hours"]
    resource["Over the Limit (hrs)"] = resource["Variance (Actual - Planned)"].clip(lower=0)
    resource["Remaining (hrs)"] = (-resource["Variance (Actual - Planned)"]).clip(lower=0)
    resource["Within Limit?"] = np.where(resource["Over the Limit (hrs)"] > 0, "No", "Yes")

    # Mismatches (within quarter but (proj, task) not in plan for that resource)
    planned_pairs = plan_assign[["res_key", "proj_key", "task_key"]].drop_duplicates()
    actual_pairs = act_in_q[["res_key", "proj_key", "task_key", "Resource", "Project Code", "Task code", "Date", "Total Hours"]].copy()
    mm = actual_pairs.merge(planned_pairs, on=["res_key", "proj_key", "task_key"], how="left", indicator=True)
    mm = mm[mm["_merge"] == "left_only"].drop(columns="_merge")

    # Classify: which part mismatched?
    planned_proj = plan_assign[["res_key", "proj_key"]].drop_duplicates().assign(ProjPlanned=True)
    planned_task = plan_assign[["res_key", "task_key"]].drop_duplicates().assign(TaskPlanned=True)
    mm = mm.merge(planned_proj, on=["res_key", "proj_key"], how="left")
    mm = mm.merge(planned_task, on=["res_key", "task_key"], how="left")

    conditions = [
        (mm["ProjPlanned"].isna() & mm["TaskPlanned"].isna()),
        (mm["ProjPlanned"].notna() & mm["TaskPlanned"].isna()),
        (mm["ProjPlanned"].isna() & mm["TaskPlanned"].notna()),
    ]
    labels = [
        "Both Project & Task code not in plan",
        "Task code not in plan for this Project",
        "Project code not in plan for this Task",
    ]
    mm["Issue"] = np.select(conditions, labels, default="Not in plan")
    mismatches = mm[["Resource", "Date", "Project Code", "Task code", "Total Hours", "Issue"]].sort_values(["Resource", "Date"])

    return assignment_view, resource, mismatches

def download_excel(assignment: pd.DataFrame, resource: pd.DataFrame, mismatches: pd.DataFrame) -> bytes:
    """Create a multi-sheet Excel report in memory."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # Section A
        assignment.to_excel(writer, sheet_name="Quarter Check", index=False, startrow=1)
        ws = writer.sheets["Quarter Check"]
        ws["A1"] = "A) Assignment-level comparison"

        # Section B
        startrow_b = len(assignment) + 4
        ws.cell(row=startrow_b-1, column=1, value="B) Resource-level summary (quarter)")
        resource.to_excel(writer, sheet_name="Quarter Check", index=False, startrow=startrow_b)

        # Section C
        startrow_c = startrow_b + len(resource) + 4
        ws.cell(row=startrow_c-1, column=1, value="C) Timesheet rows with mismatched Project/Task codes (within quarter)")
        if mismatches.empty:
            ws.cell(row=startrow_c, column=1, value="No mismatches found within the quarter window.")
        else:
            mismatches.to_excel(writer, sheet_name="Quarter Check", index=False, startrow=startrow_c)

    buf.seek(0)
    return buf.read()

# ----------------------------- UI ------------------------------------- #

st.title("Quarterly Timesheet Auditor")
st.caption("Upload planned assignments and actual timesheets to validate project/task codes and hour limits per resource.")

with st.expander("Instructions", expanded=True):
    st.markdown("""
1. **Prepare two Excel files**  
   - **Plan file**: Contains **Resource, Project Code, Task code, Start Date, End Date, Total Hours** for the quarter.  
   - **Actuals file**: Contains **Resource, Project Code, Task code, Date, Total Hours** (daily or granular timesheet entries).  
2. Upload both files below, select the **sheets**, and **map columns** if your headers differ.  
3. Click **Run comparison** to see:
   - **A) Assignment-level**: Planned vs Actual, with **Over the Limit (hrs)** shown in **red**  
   - **B) Resource-level** summary for the quarter  
   - **C) Mismatches** where Project/Task codes don’t exist in plan for that resource (within the quarter window).  
4. Use **Download Excel report** to export all sections into a single workbook.
""")

plan_file = st.file_uploader("Upload Plan Excel", type=["xlsx", "xls"])
actual_file = st.file_uploader("Upload Actuals Excel", type=["xlsx", "xls"])

if plan_file and actual_file:
    # Pick sheets
    plan_sheet = pick_sheet_ui("Pick sheet in Plan file", plan_file)
    actual_sheet = pick_sheet_ui("Pick sheet in Actuals file", actual_file)

    # Read data
    try:
        plan_raw = pd.read_excel(plan_file, sheet_name=plan_sheet, engine="openpyxl")
        actual_raw = pd.read_excel(actual_file, sheet_name=actual_sheet, engine="openpyxl")
    except Exception as e:
        st.error(f"Error reading Excel: {e}")
        st.stop()

    st.markdown("### Preview")
    tab1, tab2 = st.tabs(["Plan (first 5 rows)", "Actuals (first 5 rows)"])
    with tab1:
        st.dataframe(plan_raw.head())
    with tab2:
        st.dataframe(actual_raw.head())

    # Column mapping
    plan_required = {
        "Resource": "",
        "Project Code": "",
        "Task code": "",
        "Start Date": "",
        "End Date": "",
        "Total Hours": "",
    }
    actual_required = {
        "Resource": "",
        "Project Code": "",
        "Task code": "",
        "Date": "",
        "Total Hours": "",
    }

    st.divider()
    cols_map_plan = map_columns_ui(plan_raw, "Plan", plan_required)
    cols_map_act = map_columns_ui(actual_raw, "Actuals", actual_required)

    run = st.button("Run comparison", type="primary", use_container_width=True)

    if run:
        # Validate mappings
        if any(v is None for v in cols_map_plan.values()) or any(v is None for v in cols_map_act.values()):
            st.error("Please map all required columns for both Plan and Actuals.")
            st.stop()

        # Standardize plan
        plan = plan_raw.rename(columns={v: k for k, v in cols_map_plan.items() if v})
        plan = plan[list(plan_required.keys())].copy()

        # Types
        plan["Start Date"] = to_date(plan["Start Date"])
        plan["End Date"] = to_date(plan["End Date"])
        plan["Total Hours"] = to_numeric(plan["Total Hours"])
        plan["Planned Total Hours"] = plan["Total Hours"]

        # Standardize actuals
        actual = actual_raw.rename(columns={v: k for k, v in cols_map_act.items() if v})
        actual = actual[list(actual_required.keys())].copy()
        actual["Date"] = to_date(actual["Date"])
        actual["Total Hours"] = to_numeric(actual["Total Hours"])

        # Compute
        try:
            assignment_df, resource_df, mismatch_df = build_reports(plan, actual)
        except Exception as e:
            st.exception(e)
            st.stop()

        st.success("Comparison complete.")

        st.markdown("### A) Assignment-level comparison")
        st.dataframe(style_overlimit(assignment_df, "Over the Limit (hrs)"), use_container_width=True)

        st.markdown("### B) Resource-level summary (quarter)")
        st.dataframe(style_overlimit(resource_df, "Over the Limit (hrs)"), use_container_width=True)

        st.markdown("### C) Timesheet rows with mismatched Project/Task codes (within quarter)")
        if mismatch_df.empty:
            st.info("No mismatches found within the quarter window.")
        else:
            st.dataframe(mismatch_df, use_container_width=True)

        # Download report
        xls_bytes = download_excel(assignment_df, resource_df, mismatch_df)
        st.download_button(
            label="Download Excel report",
            data=xls_bytes,
            file_name=f"quarter_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

else:
    st.info("Upload both the Plan and Actuals Excel files to begin.")
