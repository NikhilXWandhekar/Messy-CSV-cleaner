# Messy CSV Cleaner

This project provides a small utility to clean messy CSV files using **pandas** and friends.

## Features
- Normalizes column names and string values
- Parses various date formats into ISO (YYYY-MM-DD)
- Converts numeric columns and fills missing numeric values (median for age)
- Removes duplicates and rows with too many missing values
- Produces a JSON report describing cleaning actions

## Usage
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run cleaner:
   ```bash
   python cleaner.py --input data/messy_data.csv --output data/cleaned_data.csv --report report.json
   ```
3. Open `data/cleaned_data.csv` and `report.json`.

## Files
- `cleaner.py` - main cleaning script (CLI)
- `data/messy_data.csv` - sample messy dataset
- `messy_cleaner_demo.ipynb` - notebook demo
- `requirements.txt` - dependencies


