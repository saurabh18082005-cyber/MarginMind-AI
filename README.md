# MarginMind AI — CSV Auditor

A lightweight Flask app for Indian small online sellers to inspect estimated product profitability from a CSV file.

## Features
- Upload a sales CSV (maximum 5 MB)
- Validate required columns and numeric inputs
- Calculate estimated net revenue, estimated profit, margin and return rate
- Compare products by estimated profit
- Download sample data and view a preview on the homepage
- No AI model or API required for this first version

## Run locally (Windows PowerShell)
```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000.

## CSV columns
`product, units_sold, selling_price, product_cost, shipping_cost, platform_fee, discount, returns`

Assumptions: selling price, product cost, and shipping cost are per unit; platform fee and discount are total row-level amounts; returns is a count of returned units. This MVP deducts returned sales value but does not credit back product or shipping costs for returned units. Other costs are excluded unless included in the file. Results are estimates, not formal accounting advice.

## Privacy
This version processes the CSV in the app and does not call an AI API. Do not upload sensitive customer information to a public demo.
