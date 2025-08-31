import json
from openai import OpenAI
import pandas as pd
import logging
import re

# Setup logging to file
logging.basicConfig(
    filename="siem_rule_analysis.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

client = OpenAI(api_key="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")

SIEM_RULE = '''| tstats count min(_time) as firstTime max(_time) as lastTime from datamodel=Web where Web.url IN ("/AHT/AhtApiService.asmx/AuthUser") Web.status=200 Web.http_method=POST by Web.http_user_agent, Web.status Web.http_method, Web.url, Web.url_length, Web.src, Web.dest, sourcetype | `drop_dm_object_name("Web")` | `security_content_ctime(firstTime)` | `security_content_ctime(lastTime)` | `ws_ftp_remote_code_execution_filter`'''

COMPOSITE_PROMPT = """
You are an expert cybersecurity analyst and SIEM detection engineer.

Please perform all the following steps for the provided SIEM rule. Return your answer as a JSON object with the following keys:
- "iocs": JSON dictionary as in step 1
- "ioc_context": dictionary mapping each extracted IoC to a 1-sentence contextual explanation (step 2)
- "human_explanation": a concise, comprehensive, human-readable explanation of what the rule detects, using the IoC context (step 3)
- "likely_data_source": the single most relevant MITRE ATT&CK data source for this rule, with 1-sentence justification (step 4)
- "mitre_techniques": a JSON array, each entry: {{'id': ..., 'name': ...}}, listing the relevant MITRE ATT&CK techniques for the rule and data source (step 5)
- "technique_scoring": a dictionary with each technique ID as key and as value an object: {{'relevance': float, 'rationale': string}}, rating relevance from 0 (not related) to 1 (highly related).

Here are the steps:
1. Extract all indicators of compromise (IoCs) such as process names, file names, IP addresses, registry keys, etc, from the SIEM rule. Return as a JSON dictionary with keys as IoC types and values as lists of extracted IoCs.
2. For each extracted IoC, provide a short (max 1-sentence) contextual explanation of what it is or why it is suspicious.
3. Using the rule and IoC context, write a concise but comprehensive, human-readable explanation of what the rule detects.
4. From the MITRE ATT&CK Data Sources list, select the single most relevant one for this rule and justify your choice in one sentence.
5. Given the rule description and the data source, list all relevant MITRE ATT&CK techniques (with ID and name) that match the detection logic, as a JSON array.
6. For each MITRE technique, explain why the mapping is or isn't appropriate and rate the relevance on a scale from 0 (not related) to 1 (highly related).

Input SIEM rule: {siem_rule}

Format your output as a single JSON object.
"""


def extract_first_json_object(text):
    """
    Extracts the first valid JSON object from a string.
    Returns the JSON substring, or raises ValueError if not found.
    """
    import re
    stack = []
    start = None
    for i, c in enumerate(text):
        if c == '{':
            if not stack:
                start = i
            stack.append('{')
        elif c == '}':
            if stack:
                stack.pop()
                if not stack and start is not None:
                    return text[start:i+1]
    raise ValueError("No valid JSON object found in text.")


def clean_double_colon_keys(json_str):
    """
    Removes lines with invalid double-colon keys (e.g., "foo": "bar": "baz") inside ioc_context.
    """
    # Find the ioc_context object boundaries
    ioc_context_match = re.search(r'"ioc_context"\s*:\s*\{', json_str)
    if not ioc_context_match:
        return json_str
    start = ioc_context_match.end()
    # Find the closing } for ioc_context (not perfect, but works for flat objects)
    end = start
    brace_count = 1
    while end < len(json_str):
        if json_str[end] == '{':
            brace_count += 1
        elif json_str[end] == '}':
            brace_count -= 1
            if brace_count == 0:
                break
        end += 1
    # Split ioc_context into lines
    before = json_str[:start]
    ioc_context_body = json_str[start:end]
    after = json_str[end:]
    # Remove lines with double-colon keys
    cleaned_lines = []
    for line in ioc_context_body.splitlines():
        # Remove lines like: "foo": "bar": "baz",
        if re.match(r'\s*".+?":\s*".+?":\s*".+?",?\s*$', line):
            continue
        cleaned_lines.append(line)
    cleaned_body = "\n".join(cleaned_lines)
    return before + cleaned_body + after


def analyze_rule_and_output_excel(rule, rule_id, ground_truth, excel_path="siem_rule_analysis.xlsx"):
    prompt = COMPOSITE_PROMPT.format(siem_rule=rule)

    logging.info("Sending prompt to OpenAI: %s", prompt)

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    result = response.choices[0].message.content.strip()

    logging.info("Received response: %s", result)

    # Remove code block markers if present
    if result.startswith('```json'):
        result = result.lstrip('```json').rstrip('```').strip()
    # Extract only the first valid JSON object
    try:
        json_str = extract_first_json_object(result)
        # Clean up invalid double-colon keys in ioc_context
        json_str = clean_double_colon_keys(json_str)
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            logging.warning(
                "Standard JSON decode failed, trying json5. Error: %s", e)
            try:
                import json5
            except ImportError:
                logging.error(
                    "json5 is not installed. Please install it with 'pip install json5'.")
                raise
            try:
                parsed = json5.loads(json_str)
            except Exception as e2:
                logging.error(
                    "json5 also failed to parse. Problematic JSON string:\n%s", json_str)
                raise e2
    except Exception as e:
        logging.error("Could not parse response: %s", result)
        raise e

    # Prepare a flat dictionary for a single row
    row = {
        'rule_id': rule_id,
        'rule': rule,
        'ground_truth': ground_truth,  # Add ground_truth column
        'human_explanation': parsed.get('human_explanation', '')
        # 'likely_data_source' removed
    }
    # Add any extra top-level keys from the parsed JSON, except excluded columns
    for k, v in parsed.items():
        if k not in row and k not in (
            'iocs', 'ioc_context', 'mitre_techniques', 'technique_scoring',
            'likely_data_source'  # exclude this column
        ):
            row[k] = v

    # Flatten IoCs for columns, but exclude specific keys
    iocs = parsed.get('iocs', {})
    for ioc_type, ioc_values in iocs.items():
        if ioc_type not in ['url', 'http_method', 'http_status']:
            row[f'ioc_{ioc_type}'] = ', '.join(str(x) for x in ioc_values)

    # Dump ioc_context as JSON string, but do not add individual ioc_context columns
    ioc_context = parsed.get('ioc_context', {})
    row['ioc_context'] = json.dumps(ioc_context, ensure_ascii=False)

    # Dump mitre_techniques and technique_scoring as JSON strings
    mitre_techniques = parsed.get('mitre_techniques', [])
    row['mitre_techniques'] = json.dumps(mitre_techniques, ensure_ascii=False)
    row['technique_scoring'] = json.dumps(parsed.get(
        'technique_scoring', {}), ensure_ascii=False)

    # Add recommendations column: comma-separated list of mitre technique IDs
    recommendations = ",".join(
        t.get('id', '') for t in mitre_techniques if t.get('id')
    )
    row['recommendations'] = recommendations

    # Write or append to a single sheet
    try:
        df = pd.read_excel(excel_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        logging.info("Appended new row to existing Excel file: %s", excel_path)
    except FileNotFoundError:
        df = pd.DataFrame([row])
        logging.info("Created new Excel file: %s", excel_path)

    df.to_excel(excel_path, index=False, sheet_name="All_Rules")
    logging.info("Analysis written to %s", excel_path)
    print(f"Analysis written to {excel_path}")


if __name__ == "__main__":
    # Read SIEM rules from Excel file (column name 'search', lowercase)
    # Change this to your input file path
    input_excel = "splunk_endpoint_rules.xlsx"
    try:
        rules_df = pd.read_excel(input_excel)
    except Exception as e:
        logging.error("Could not read input Excel file: %s", input_excel)
        raise e

    # Accept lowercase 'search' as the column name
    if 'search' not in rules_df.columns:
        raise ValueError(
            "Input Excel file must contain a column named 'search'.")
    if 'id' not in rules_df.columns:
        raise ValueError(
            "Input Excel file must contain a column named 'id' (rule id).")
    if 'mitre_attack_id' not in rules_df.columns:
        raise ValueError(
            "Input Excel file must contain a column named 'mitre_attack_id'.")

    # Load existing output Excel if present, get set of already processed rule_ids
    output_excel = "siem_rule_analysis.xlsx"
    try:
        out_df = pd.read_excel(output_excel)
        existing_ids = set(out_df['rule_id'].astype(str))
    except Exception:
        existing_ids = set()

    for idx, row in rules_df.iterrows():
        rule_id = str(row['id'])
        rule = row['search']
        ground_truth = row['mitre_attack_id']
        logging.info(
            f"Analyzing Excel row {idx}: rule_id={rule_id}, ground_truth={ground_truth}, rule={rule[:120]}...")
        if rule_id in existing_ids:
            logging.info(f"Skipping rule_id {rule_id} (already processed)")
            continue
        analyze_rule_and_output_excel(
            rule, rule_id, ground_truth, output_excel)
