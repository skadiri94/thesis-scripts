import os
import pandas as pd
from typing import Dict, List, Optional, Any
import json
import logging
from logging.handlers import RotatingFileHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import JSONLoader
from langchain_google_genai import ChatGoogleGenerativeAI


# Set up logging configuration
def setup_logging():
    """Configure logging for the security rule analyzer."""
    logger = logging.getLogger('z1')
    logger.setLevel(logging.INFO)

    # Create a handler that writes to a file, with rotation enabled
    handler = RotatingFileHandler(
        'mapper_demo.log',
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


def log_state_after_step(step_name):
    def decorator(func):
        def wrapper(state):
            result = func(state)
            logger.info(
                f"State after '{step_name}': {json.dumps(result, default=str, indent=2)}")
            return result
        return wrapper
    return decorator


def extract_translation(resp):
    if isinstance(resp, dict):
        if "natural_language_translation" in resp:
            return resp["natural_language_translation"]
        elif "content" in resp:
            return resp["content"]
    return resp


class SecurityAnalysisState(dict):
    """State for the security rule analysis pipeline."""
    input_rule: str
    extracted_iocs: Optional[Dict[str, Any]] = None
    ioc_context_data: Optional[str] = None
    natural_language_rule: Optional[str] = None
    data_source_info: Optional[Dict[str, Any]] = None
    probable_techniques: Optional[List[Dict[str, Any]]] = None
    technique_refined: Optional[List[Dict[str, Any]]] = None
    confidence_scores: Optional[List[Dict[str, float]]] = None
    final_output: Optional[List[str]] = None


# Initialize the LLM
logger.info("Initializing ChatOpenAI model")
# llm = ChatOpenAI(model="gpt-4.1", temperature=0)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# 1. IoC Extractor Component
logger.info("Setting up IoC Extraction component")
ioc_extraction_prompt = ChatPromptTemplate.from_template("""
You are a cybersecurity expert specialized in extracting Indicators of Compromise from security siem rules.
Analyze the following SIEM rule and extract all possible indicators of compromise (IoCs) such as process names, file names, IP addresses, registry keys, etc. Return as a JSON dictionary with keys as IoC types and values as lists of extracted IoCs.

Security Rule:
{input_rule}
""")

ioc_extraction_chain = (
    {"input_rule": RunnablePassthrough()}
    | ioc_extraction_prompt
    | llm
    | JsonOutputParser()
)

# 2. IOC Context Information Retriever Component
logger.info("Setting up IOC Context Information Retriever Component")
context_retrieval_prompt = ChatPromptTemplate.from_template("""
For each Indicator of Compromise (IoC) in the list below, systematically search reputable web resources and provide a brief (one or two sentences) contextual information of each IoC (such as associated malware, threat actors, or recent activity).
Output the results as a JSON object where each key is the IoC and the value is its contextual information.

List of IoCs:
{extracted_iocs}
""")

ioc_context_retrieval_chain = (
    {"extracted_iocs": lambda x: json.dumps(x["extracted_iocs"], indent=2)}
    | context_retrieval_prompt
    | llm
    | JsonOutputParser()
)

# 3. Natural Language Translator Component
logger.info("Setting up Natural Language Translation component")

nl_translation_prompt = ChatPromptTemplate.from_template("""
You are a cybersecurity expert. Given the technical security rule below and the contextual information for its IoCs, 
provide a clear natural language translation of the rule.  Return as a JSON object.

SIEM Rule:
{input_rule}

IoC Contextual Information:
{ioc_context_data}
""")

nl_translation_chain = (
    {"input_rule": lambda x: x["input_rule"],
     "ioc_context_data": lambda x: x["ioc_context_data"]}
    | nl_translation_prompt
    | llm
    | JsonOutputParser()
)

# 4. Data Source/Mitigation Identifier Component
logger.info("Setting up Data Source Identifier component")

data_source_prompt = ChatPromptTemplate.from_template("""
You are a cybersecurity expert. Given the following SIEM rule explained in natural language, identify:
-The rule's data sources
-The corresponding mitigations that would be effective against this threat

SIEM Rule:
{natural_language_rule}

Return your analysis as a JSON object.
""")

data_source_retrieval_chain = (
    {
        "natural_language_rule": lambda x: x["natural_language_rule"]
    }
    | data_source_prompt
    | llm
    | JsonOutputParser()
)

logger.info("Setting up Techniques Recommender component")


# 5. Techniques Recommender Component

techniques_prompt = ChatPromptTemplate.from_template("""
You are a cybersecurity expert.
Based on the security rule natural language explanation, and identified data sources/mitigations,
get 11 of the most probable MITRE ATT&CK techniques that this rule might be detecting.

Natural language siem rule: {natural_language_rule}
Data source and mitigation info: {data_source_info}

Return a list of JSON objects, each containing the
MITRE technique ID, technique name, and technique description:

[
  {{
    "technique_id": "T1234",
    "technique_name": "Example Technique",
    "description": "description of why this technique is relevant"
  }},
  ...
]

""")

techniques_chain = (
    {
        "natural_language_rule": lambda x: x["natural_language_rule"],
        "data_source_info": lambda x: x["data_source_info"]
    }
    | techniques_prompt
    | llm
    | JsonOutputParser()  # type: ignore
)

# 6. Technique refiner component

techniques_refiner_prompt = ChatPromptTemplate.from_template("""
 You are a cyber threat detection and MITRE ATT&CK expert.
Perform all the following steps for the provided SIEM rule:
1. Compare the rule description and technique description.
2. Concisely explain (chain-of-thought) why the technique is or isn’t relevant, citing specific overlaps or gaps include the explanation.
3. Assign a confidence score (0 –1) for relevance with with be strict with scoring.
Be conservative: Only assign scores above 0.8 when the match is clear and substantial. Do not inflate scores for partial matches.

Output:Return only techniques with confidence > 0.8, using this JSON array format:

Inputs:
Siem Rule: {natural_language_rule}

Techniques: {probable_techniques}
""")

techniques_refiner_chain = (
    {
        "natural_language_rule": lambda x: x["natural_language_rule"],
        "probable_techniques": lambda x: json.dumps(x["probable_techniques"], indent=2)
    }
    | techniques_refiner_prompt
    | llm
    | JsonOutputParser()
)

# def generate_final_output(state):
#     logger.info("Generating final output")
#     scores = state["confidence_scores"]
#     # Sort by confidence score in descending order
#     sorted_scores = sorted(
#         scores, key=lambda x: x["confidence_score"], reverse=True)
#     # Format as list of technique IDs
#     final_output = [f"{score['technique_id']}" for score in sorted_scores]

#     logger.info(
#         f"Final techniques (in order of confidence): {', '.join(final_output)}")
#     return {"final_output": final_output}


def build_security_analysis_graph():
    logger.info("Building security analysis workflow graph")
    workflow = StateGraph(SecurityAnalysisState)

    # Nodes
    workflow.add_node("extract_iocs", lambda state: {
        **state,
        "extracted_iocs": ioc_extraction_chain.invoke(state["input_rule"])
    })
    workflow.add_node("retrieve_context_info", lambda state: {
        **state,
        "ioc_context_data": ioc_context_retrieval_chain.invoke(state)
    })
    workflow.add_node("translate_rule_to_nl", (lambda state: {
        **state,
        "natural_language_rule": extract_translation(nl_translation_chain.invoke({
            "input_rule": state["input_rule"],
            "ioc_context_data": state["ioc_context_data"]
        }))
    }))
    workflow.add_node(
        "identify_data_sources",
        (lambda state: {
            **state,
            "data_source_info": data_source_retrieval_chain.invoke({
                "natural_language_rule": state["natural_language_rule"]
            })
        }
        ))
    workflow.add_node(
        "recommend_techniques", (
            lambda state: {
                **state,
                "probable_techniques": techniques_chain.invoke({
                    "natural_language_rule": state["natural_language_rule"],
                    "data_source_info": state["data_source_info"]
                })
            })
    )
    workflow.add_node(
        "technique_refiner", (
            lambda state: {
                **state,
                "technique_refined": techniques_refiner_chain.invoke({
                    "natural_language_rule": state["natural_language_rule"],
                    "probable_techniques": state["probable_techniques"]
                })
            })
    )
    # workflow.add_node("compare_techniques", compare_rule_to_techniques)
    # workflow.add_node("calculate_confidence", calculate_confidence_scores)
    # workflow.add_node("generate_output", generate_final_output)

    # Edges
    workflow.add_edge("extract_iocs", "retrieve_context_info")
    workflow.add_edge("retrieve_context_info", "translate_rule_to_nl")
    workflow.add_edge("translate_rule_to_nl", "identify_data_sources")
    workflow.add_edge("identify_data_sources", "recommend_techniques")
    workflow.add_edge("recommend_techniques", "technique_refiner")
    workflow.add_edge("technique_refiner", END)
    # workflow.add_edge("compare_techniques", "calculate_confidence")
    # workflow.add_edge("calculate_confidence", "generate_output")
    # workflow.add_edge("generate_output", END)

    # Entry Point
    workflow.set_entry_point("extract_iocs")

    logger.info("Workflow graph built successfully")
    return workflow.compile()


# Create a function to run the entire pipeline


def analyze_security_rule(rule_text):
    logger.info(f"Starting analysis of rule: {rule_text.strip()}")
    try:
        graph = build_security_analysis_graph()
        state = {"input_rule": rule_text}
        logger.info("Invoking analysis graph")
        result = graph.invoke(state)
        logger.info("Analysis completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error during rule analysis: {str(e)}", exc_info=True)
        raise


def analyze_rules_from_excel(input_excel_path: str, rule_column: str, output_excel_path: str):
    """
    Analyze all rules from a specified column in an Excel file and save results to a new Excel file.
    Output columns: input_rule, predicted_techniques, technique_refined, ground_truth_id
    If output file exists, skip rules whose input_rule is already present.
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

    # If output file exists, load already processed id values
    already_processed = set()
    if os.path.exists(output_excel_path):
        try:
            out_df = pd.read_excel(output_excel_path)
            if "id" in out_df.columns:
                already_processed = set(str(x).strip()
                                        for x in out_df["id"] if pd.notna(x))
            logger.info(
                f"Loaded {len(already_processed)} already processed rules from output file.")
        except Exception as e:
            logger.warning(
                f"Could not read output file for deduplication: {e}")

    results = []
    try:
        for idx in range(len(df)):
            # Get id and rule_str from input file
            if "id" in df.columns:
                id = str(df.loc[idx, "id"])
            else:
                id = str(idx)
            rule_str = str(df.loc[idx, rule_column]).strip()
            if pd.isna(rule_str) or not rule_str:
                logger.info(f"Skipping empty rule at row {idx}")
                continue
            if id in already_processed:
                logger.info(
                    f"Skipping already processed rule at row {idx}: {id}")
                continue
            logger.info(f"Processing row {idx} with id {id}")
            logger.info(f"Analyzing rule at row {idx}")
            try:
                result = analyze_security_rule(rule_str)
                refined = result.get("technique_refined", [])
                if isinstance(refined, list):
                    predicted_techniques = ", ".join(
                        str(item.get("technique_id", "")) for item in refined if "technique_id" in item
                    )
                    technique_refined = json.dumps(refined, indent=2)
                else:
                    predicted_techniques = ""
                    technique_refined = ""
                ground_truth = df.loc[idx,
                                      ground_truth_col] if ground_truth_col else ""
                result_row = {
                    "id": id,
                    "input_rule": rule_str,
                    "predicted_techniques": predicted_techniques,
                    "technique_refined": technique_refined,
                    "ground_truth_id": ground_truth
                }
                results.append(result_row)
            except Exception as e:
                logger.error(f"Error analyzing rule at row {idx}: {str(e)}")
                ground_truth = df.loc[idx,
                                      ground_truth_col] if ground_truth_col else ""
                results.append({
                    "id": id,
                    "input_rule": rule_str,
                    "predicted_techniques": "",
                    "technique_refined": "",
                    "ground_truth_id": ground_truth,
                    "error": str(e)
                })
    except KeyboardInterrupt:
        logger.warning(
            "KeyboardInterrupt detected! Saving progress before exiting.")
        # Save progress immediately
        if os.path.exists(output_excel_path):
            try:
                out_df = pd.read_excel(output_excel_path)
                results_df = pd.DataFrame(results)
                combined_df = pd.concat(
                    [out_df, results_df], ignore_index=True)
                combined_df.to_excel(output_excel_path, index=False)
                logger.info(
                    "Partial results appended to existing output file successfully.")
            except Exception as e:
                logger.error(
                    f"Error appending to output file during KeyboardInterrupt: {e}")
        else:
            logger.info(f"Writing partial results to {output_excel_path}")
            results_df = pd.DataFrame(results)
            results_df.to_excel(output_excel_path, index=False)
            logger.info("Partial results written successfully.")
        raise

    # If output file exists, append new results to it
    if os.path.exists(output_excel_path):
        try:
            out_df = pd.read_excel(output_excel_path)
            results_df = pd.DataFrame(results)
            combined_df = pd.concat([out_df, results_df], ignore_index=True)
            combined_df.to_excel(output_excel_path, index=False)
            logger.info(
                "Results appended to existing output file successfully.")
        except Exception as e:
            logger.error(f"Error appending to output file: {e}")
    else:
        logger.info(f"Writing results to {output_excel_path}")
        results_df = pd.DataFrame(results)
        results_df.to_excel(output_excel_path, index=False)
        logger.info("Results written successfully.")


# Example usage
if __name__ == "__main__":
    try:
        logger.info("Starting Security Rule Analyzer")
        # Example usage for batch processing from Excel:
        # Uncomment and set your file paths and column name
        analyze_rules_from_excel(
            input_excel_path="splunk_detections.xlsx",
            rule_column="search",
            output_excel_path="MappedOutput.xlsx"
        )

        logger.info("Program execution completed successfully")
        print("Program execution completed successfully")
    except Exception as e:
        logger.critical(
            f"Program terminated with error: {str(e)}", exc_info=True)
        print(f"An error occurred: {str(e)}")
