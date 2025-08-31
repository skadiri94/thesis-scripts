import os
import pandas as pd
from typing import Dict, List, Optional, Any
import json
import logging
from logging.handlers import RotatingFileHandler
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic


# Set up logging configuration
def setup_logging():
    """Configure logging for the security rule analyzer."""
    logger = logging.getLogger('rule_mapper')
    logger.setLevel(logging.INFO)

    # Create a handler that writes to a file, with rotation enabled
    handler = RotatingFileHandler(
        'rule-mapper1.log',
        maxBytes=50*1024*1024,  # 10MB max size
        backupCount=5  # Keep up to 5 backup logs
    )

    # Define the format for log messages
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    return logger


# Initialize logger
logger = setup_logging()

# Use Gemini for LLM
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash", temperature=0)

# Use Claude for LLM
# llm = ChatAnthropic(model="claude-3-7-sonnet-20250219", temperature=0)


main_prompt = ChatPromptTemplate.from_template("""
You are a cybersecurity expert specialized in MITRE ATT&CK mapping.
Your task is to analyze security rules and predict the most likely MITRE ATT&CK techniques they detect.
Always reason step by step before selecting techniques.
Only recommend techniques when there is clear evidence.
Output strict JSON.

Security Rule:

{input_rule}

---

Step-by-step reasoning:
1. Extract observable behaviors, artifacts, indicators.
2. Identify possible attack techniques.
3. Justify each technique selected.

Output JSON format:

[
  {{
    "technique_id": "Txxxx",
    "technique_name": "",
    "tactic": "",
    "justification": "Explanation based on rule content"
  }}
]
""")

chain = (
    {"input_rule": lambda x: x["input_rule"]}
    | main_prompt
    | llm
    | JsonOutputParser()
)


def analyze_security_rule(input_rule):
    logger.info(f"Analyzing rule: {input_rule.strip()}")
    try:
        result = chain.invoke({"input_rule": input_rule})
        logger.info(f"Analysis result: {result}")
        return result
    except Exception as e:
        logger.error(f"Unexpected error during rule analysis: {e}")
        return None


def analyze_rules_from_excel(input_excel_path: str, rule_column: str, output_excel_path: str):
    """
    Analyze all rules from a specified column in an Excel file and save results to a new Excel file.
    Output columns: input_rule, predicted_techniques, predicted_technique_ids, ground_truth_id
    """
    logger.info(
        f"Loading rules from {input_excel_path}, column '{rule_column}'")
    df = pd.read_excel(input_excel_path)
    if rule_column not in df.columns:
        logger.error(f"Column '{rule_column}' not found in the Excel file.")
        raise ValueError(
            f"Column '{rule_column}' not found in the Excel file.")

    # Check for ground truth column
    ground_truth_col = None
    for col in df.columns:
        if col.lower() in ["ground_truth_id", "groundtruth_id", "mitre_attack_id"]:
            ground_truth_col = col
            break

    results = []
    for idx, rule_text in enumerate(df[rule_column]):
        if pd.isna(rule_text) or not str(rule_text).strip():
            logger.info(f"Skipping empty rule at row {idx}")
            continue
        logger.info(f"Analyzing rule at row {idx}")
        try:
            predicted_techniques = analyze_security_rule(str(rule_text))
            predicted_techniques_str = json.dumps(predicted_techniques)
            # Extract only the technique_id values as a comma-separated string
            predicted_technique_ids = ", ".join(
                t.get("technique_id", "") for t in predicted_techniques if t.get("technique_id", "")
            )
            ground_truth = df.loc[idx,
                                  ground_truth_col] if ground_truth_col else ""
            result_row = {
                "input_rule": rule_text,
                "predicted_techniques": predicted_techniques_str,
                "predicted_technique_ids": predicted_technique_ids,
                "ground_truth_id": ground_truth
            }
            results.append(result_row)
        except Exception as e:
            logger.error(f"Error analyzing rule at row {idx}: {str(e)}")
            ground_truth = df.loc[idx,
                                  ground_truth_col] if ground_truth_col else ""
            results.append({
                "input_rule": rule_text,
                "predicted_techniques": "",
                "predicted_technique_ids": "",
                "ground_truth_id": ground_truth,
                "error": str(e)
            })

    logger.info(f"Writing results to {output_excel_path}")
    results_df = pd.DataFrame(results)
    results_df.to_excel(output_excel_path, index=False)
    logger.info("Results written successfully.")


if __name__ == "__main__":
    try:
        logger.info("Starting Security Rule Analyzer")
        analyze_rules_from_excel(
            input_excel_path="test_excel.xlsx",
            rule_column="search",
            output_excel_path="test_cot_co3-7-test-result.xlsx"
        )
        logger.info("Program execution completed successfully")
        print("Program execution completed successfully")
    except Exception as e:
        logger.critical(
            f"Program terminated with error: {str(e)}", exc_info=True)
        print(f"An error occurred: {str(e)}")
