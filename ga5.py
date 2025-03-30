# Standardized country mappings
import re
from datetime import datetime

import numpy as np
import pandas as pd

COUNTRY_MAP = {
    "USA": "US", "U.S.A": "US", "United States": "US",
    "FRANCE": "FR", "FR": "FR", "FRA": "FR",
    "UK": "GB", "United Kingdom": "GB", "GB": "GB"
}


def extract_parameters(task_description):
    """
    Extracts the cutoff time, product name, and country code from the task description.
    """
    # Extract date
    date_match = re.search(
        r'up to and including (.+?\d{4} \d{2}:\d{2}:\d{2})', task_description)
    cutoff_time = date_match.group(1) if date_match else None
    if cutoff_time:
        cutoff_time = datetime.strptime(cutoff_time, "%a %b %d %Y %H:%M:%S")

    # Extract product
    product_match = re.search(
        r'Product Filter: Transactions for a specific product \((.+?)\)', task_description)
    product_name = product_match.group(1) if product_match else None

    # Extract country
    country_match = re.search(
        r'Country Filter: Transactions from a specific country \((.+?)\)', task_description)
    country_code = country_match.group(1) if country_match else None

    return cutoff_time, product_name, country_code


def excell_process(excel_path, product_name, country_code, cutoff_time):
    """
    Cleans an Excel file and calculates the total margin for a given product and country up to a specific date.
    """
    df = pd.read_excel(excel_path)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Country'] = df['Country'].str.upper().map(
        COUNTRY_MAP).fillna(df['Country'])
    df['Product'] = df['Product'].apply(
        lambda x: x.split('/')[0] if isinstance(x, str) else x)
    df['Sales'] = df['Sales'].replace('[^0-9.]', '', regex=True).astype(float)
    df['Cost'] = df['Cost'].replace('[^0-9.]', '', regex=True).astype(float)
    df['Cost'].fillna(df['Sales'] * 0.5, inplace=True)

    df_filtered = df[(df['Date'] <= cutoff_time) & (
        df['Product'] == product_name) & (df['Country'] == country_code)]
    total_sales = df_filtered['Sales'].sum()
    total_cost = df_filtered['Cost'].sum()
    total_margin = (total_sales - total_cost) / \
        total_sales if total_sales > 0 else 0

    return f"{total_margin:.2%}"


def ga_5_1(excel_path, task_description):
    """
    Extracts parameters from the task description and calculates the total margin.
    """
    cutoff_time, product_name, country_code = extract_parameters(
        task_description)
    return excell_process(excel_path, product_name, country_code, cutoff_time)


task = '''
Your solution should address the following challenges:

Trim and Normalize Strings: Remove extra spaces from the Customer Name and Country fields. Map inconsistent country names (e.g., "USA", "U.S.A", "US") to a standardized format.
Standardize Date Formats: Detect and convert dates from "MM-DD-YYYY" and "YYYY/MM/DD" into a consistent date format (e.g., ISO 8601).
Extract the Product Name: From the Product field, extract the portion before the slash (e.g., extract "Theta" from "Theta/5x01vd").
Clean and Convert Sales and Cost: Remove the "USD" text and extra spaces from the Sales and Cost fields. Convert these fields to numerical values. Handle missing Cost values appropriately (50% of Sales).
Filter the Data: Include only transactions up to and including Fri Nov 04 2022 15:11:27 GMT+0530 (India Standard Time), matching product Kappa, and country FR.
Calculate the Margin: Sum the Sales and Cost for the filtered transactions. Compute the overall margin using the formula provided.
By cleaning the data and calculating accurate margins, RetailWise Inc. can:

Improve Decision Making: Provide clients with reliable margin analyses to optimize pricing and inventory.
Enhance Reporting: Ensure historical data is consistent and accurate, boosting stakeholder confidence.
Streamline Operations: Reduce the manual effort needed to clean data from legacy sources.
Download the Sales Excel file: 

What is the total margin for transactions before Fri Nov 04 2022 15:11:27 GMT+0530 (India Standard Time) for Kappa sold in FR (which may be spelt in different ways)?
You can enter the margin as a percentage (e.g. 12.34%) or a decimal (e.g. 0.1234).'''
excel_file = "data/q-clean-up-excel-sales-data.xlsx"
# margin = ga_5_1(excel_file, task)
# print(margin)


def ga_5_2(file_path):
    """
    Reads a text file, extracts unique student IDs, and returns the count of unique students.

    :param file_path: Path to the text file containing student data.
    :return: The count of unique student IDs.
    """
    unique_students = set()

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = re.search(r'Student ID: (\d+)', line)
            if match:
                unique_students.add(match.group(1))

    return len(unique_students)


# Example usage:
file_path = "data/q-clean-up-student-marks.txt"
print(ga_5_2(file_path))


def ga_5_3():
    pass


def ga_5_4():
    pass


def ga_5_5():
    pass


def ga_5_6():
    pass


def ga_5_7():
    pass


def ga_5_8():
    pass


def ga_5_9():
    pass


def ga_5_10():
    pass
