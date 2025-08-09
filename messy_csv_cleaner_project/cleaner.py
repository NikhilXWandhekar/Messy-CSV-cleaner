"""Messy CSV Cleaner
Usage: python cleaner.py --input data/messy_data.csv --output data/cleaned_data.csv
Produces a cleaning report (report.json) and log file (cleaning.log).
"""
import argparse, json, logging
from datetime import datetime
import pandas as pd
import numpy as np
from dateutil import parser as dateparser
from tabulate import tabulate

logging.basicConfig(filename='cleaning.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def normalize_name(name):
    if pd.isna(name):
        return None
    name = str(name).strip()
    if name == '':
        return None
    # Title case and remove duplicate spaces
    return ' '.join(part.capitalize() for part in name.split())

def parse_date(val):
    if pd.isna(val) or str(val).strip()=='' :
        return None
    try:
        # dateutil can parse many formats
        dt = dateparser.parse(str(val), dayfirst=False, yearfirst=False)
        return dt.date().isoformat()
    except Exception:
        return None

def to_numeric(val):
    try:
        if pd.isna(val) or str(val).strip()=='' :
            return np.nan
        v = float(str(val).replace(',','').strip())
        return v
    except Exception:
        return np.nan

def generate_report(before_df, after_df, actions):
    report = {}
    report['before_rows'] = len(before_df)
    report['after_rows'] = len(after_df)
    report['removed_rows'] = report['before_rows'] - report['after_rows']
    report['columns'] = list(after_df.columns)
    report['actions'] = actions
    # summary stats
    report['summary_before'] = before_df.describe(include='all', datetime_is_numeric=False).to_dict()
    report['summary_after'] = after_df.describe(include='all', datetime_is_numeric=False).to_dict()
    return report

def clean_dataframe(df, logger=logging.getLogger()):
    df = df.copy()
    actions = []

    # 1) Normalize column names (strip, lower, replace spaces)
    original_cols = list(df.columns)
    new_cols = [c.strip() for c in original_cols]
    df.columns = new_cols
    actions.append(f'Normalized column names: {original_cols} -> {new_cols}')
    logger.info(actions[-1])

    # 2) Normalize names (case-insensitive dedupe)
    if 'Name' in df.columns or 'name' in df.columns:
        col = [c for c in df.columns if c.lower()=='name'][0]
        df[col] = df[col].apply(normalize_name)
        # dedupe case-insensitive keeping first occurrence (by all columns)
        before = len(df)
        df = df.drop_duplicates(subset=[col] + [c for c in df.columns if c!=col], keep='first')
        after = len(df)
        actions.append(f'Normalized names and dropped duplicates based on all columns (rows {before}->{after})')
        logger.info(actions[-1])

    # 3) Parse dates to ISO format
    date_cols = [c for c in df.columns if 'date' in c.lower()]
    for dc in date_cols:
        df[dc] = df[dc].apply(parse_date)
        actions.append(f'Parsed date column {dc} to ISO format')
        logger.info(actions[-1])

    # 4) Convert numeric columns (attempt)
    # Heuristic: columns containing 'age' or 'salary' -> numeric
    numeric_cols = [c for c in df.columns if any(k in c.lower() for k in ['age','salary','price','amount'])]
    for nc in numeric_cols:
        df[nc] = df[nc].apply(to_numeric)
        actions.append(f'Converted column {nc} to numeric')
        logger.info(actions[-1])

    # 5) Fill missing ages with median if age column present
    age_cols = [c for c in df.columns if 'age' in c.lower()]
    for ac in age_cols:
        median_age = int(np.nanmedian(df[ac].astype(float)))
        df[ac] = df[ac].fillna(median_age)
        actions.append(f'Filled missing values in {ac} with median: {median_age}')
        logger.info(actions[-1])

    # 6) Drop rows with too many missing values (threshold: >50% missing)
    thresh = int(0.5 * len(df.columns))
    before = len(df)
    df = df.dropna(thresh=thresh)
    after = len(df)
    actions.append(f'Dropped rows with fewer than {thresh} non-NA values (rows {before}->{after})')
    logger.info(actions[-1])

    # 7) Remove exact duplicates (all columns)
    before = len(df)
    df = df.drop_duplicates(keep='first')
    after = len(df)
    actions.append(f'Removed exact duplicate rows (rows {before}->{after})')
    logger.info(actions[-1])

    # 8) Trim strings and standardize empty strings to NaN
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].apply(lambda x: x.strip() if isinstance(x, str) else x)
        df[c] = df[c].replace('', None)
    actions.append('Trimmed string columns and standardized empty strings to NaN')
    logger.info(actions[-1])

    # 9) Reorder columns: Name, Age, Join Date, Salary if present
    preferred = ['Name','name','Age','age','Join Date','join date','Salary','salary']
    cols = df.columns.tolist()
    ordered = []
    for p in preferred:
        for c in cols:
            if c.lower()==p.lower() and c not in ordered:
                ordered.append(c)
    for c in cols:
        if c not in ordered:
            ordered.append(c)
    df = df[ordered]
    actions.append('Reordered columns for readability')
    logger.info(actions[-1])

    return df, actions

def main():
    parser = argparse.ArgumentParser(description='Clean a messy CSV file')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file path')
    parser.add_argument('--output', '-o', required=False, help='Output cleaned CSV file path', default='data/cleaned_data.csv')
    parser.add_argument('--report', '-r', required=False, help='Output JSON report path', default='report.json')
    args = parser.parse_args()

    logger = logging.getLogger('cleaner')
    logger.info(f'Starting cleaning for {args.input}')

    df = pd.read_csv(args.input)
    before_df = df.copy()
    cleaned_df, actions = clean_dataframe(df, logger=logger)

    # save cleaned CSV
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cleaned_df.to_csv(args.output, index=False)

    report = generate_report(before_df, cleaned_df, actions)
    with open(args.report, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print('Cleaning complete. Cleaned file:', args.output)
    print('Report saved to', args.report)
    logger.info('Cleaning complete')


if __name__ == '__main__':
    main()
