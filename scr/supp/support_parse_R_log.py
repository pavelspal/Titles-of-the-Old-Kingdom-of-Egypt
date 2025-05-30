import pandas as pd
import re
import os

import pandas as pd


# Function to extract main data, coefficients, and predicted probabilities
def parse_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Extract relevant information using regex
    formula = re.search(r"fomula:\s*(.*)", content)
    response = re.search(r"response:\s*(.*)", content)
    predictor = re.search(r"predictor:\s*(.*)", content)
    date = re.search(r"date:\s*(.*)", content)
    file_name = re.search(r"file name:\s*(.*)", content)
    data_source_path = re.search(r"data source path:\s*(.*)", content)
    aic = re.search(r"AIC:\s*([\d.]+)", content)
    null_dev = re.search(r"Null deviance:\s*([\d.]+)", content)
    residual_dev = re.search(r"Residual deviance:\s*([\d.]+)", content)

    # Extract coefficients block between "Coefficients:" and an empty line
    coeffs_match = re.search(r"Coefficients:\s*\n(.*?)(\n\s*\n|\Z)", content, re.DOTALL)
    coefficients_df = pd.DataFrame()

    if coeffs_match:
        coeffs_lines = coeffs_match.group(1).strip().splitlines()

        # Ensure (Intercept) is included in the lines
        coeffs_data = []
        for line in coeffs_lines:
            # Match lines with variable, estimate, std. error, z value, and p-value, allowing for significance markers
            match = re.match(r"(\S+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+<*\s*([-\d.eE+]+)\s*(\*\*\*|\*\*|\*|\.|\s*)", line)
            if match:
                coeffs_data.append(match.groups()[:5])

        # Create DataFrame if any coefficients were found
        if coeffs_data:
            coefficients_df = pd.DataFrame(coeffs_data,
                                           columns=['Variable', 'Estimate', 'Std. Error', 'z value', 'Pr(>|z|)'])
            # Convert numeric columns, handling scientific notation for p-values
            coefficients_df[['Estimate', 'Std. Error', 'z value']] = coefficients_df[
                ['Estimate', 'Std. Error', 'z value']].apply(pd.to_numeric)
            coefficients_df['Pr(>|z|)'] = pd.to_numeric(coefficients_df['Pr(>|z|)'], errors='coerce')

    # Extract predicted probabilities block
    probs_match = re.search(r"predicted probability:\s*\n(.*?)(\n\s*\n|\Z)", content, re.DOTALL)
    predicted_probs_df = pd.DataFrame()

    if probs_match:
        probs_lines = probs_match.group(1).strip().splitlines()
        # Assuming the first line contains headers
        headers = probs_lines[0].split()
        probs_data = [line.split()[1:] for line in probs_lines[1:]]
        predicted_probs_df = pd.DataFrame(probs_data, columns=headers)

        # Convert numeric columns
        numeric_columns = ['count', 'predicted_probability']
        for col in numeric_columns:
            if col in predicted_probs_df.columns:
                predicted_probs_df[col] = pd.to_numeric(predicted_probs_df[col])

    # Return the extracted data
    return {
        'file_path': file_path,
        'formula': formula.group(1) if formula else None,
        'response': response.group(1) if response else None,
        'predictor': [item.strip() for item in predictor.group(1).split(';')] if predictor else None,
        'date': date.group(1) if date else None,
        'file_name': file_name.group(1) if file_name else None,
        'data_source_path': data_source_path.group(1) if data_source_path else None,
        'AIC': float(aic.group(1)) if aic else None,
        'Null Deviance': float(null_dev.group(1)) if null_dev else None,
        'Residual Deviance': float(residual_dev.group(1)) if residual_dev else None,
        'coefficients_df': coefficients_df,
        'predicted_probs_df': predicted_probs_df
    }