import requests
import csv
import time
import json
import logging
import random
from tqdm import tqdm
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import asyncio

PROMPT = """
You are a skilled content categorizer. Your task is to assign EACH TAG to a SPECIFIC CATEGORY from the list below. You MUST choose from these categories ONLY:

1. ACTION - Accessories
33. Other

This categorization system is designed to highlight SPECIFIC MOMENTS OR SECTIONS WITHIN VIDEOS (e.g., outfit changes, position shifts). Do NOT use any other words or explanations. Respond ONLY with the NUMBER of the category.

Example:
Q: cowgirl
2

Q: kitchen
21

Q: hardcore
18

Q: 1080p
20

Q: doggy style
2

Q: Brunette
10

Categorize this tag: "{tag}"
""".strip()

# Initialize logging
logging.basicConfig(
    filename='llm_categorization.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

CATEGORY_MAP = {
    "1": "ACTION - Accessories",
    "33": "Other"
}

OPENROUTER_MODELS = [
  "deepseek/deepseek-chat-v3-0324",
  "x-ai/grok-3-mini-beta",
  "microsoft/wizardlm-2-8x22b",
  "meta-llama/llama-4-maverick",
  "gryphe/mythomax-l2-13b",
  "sentientagi/dobby-mini-unhinged-plus-llama-3.1-8b",
  "raifle/sorcererlm-8x22b",
]

async def call_llm(tag: str, model_config: Dict, max_retries: int = 3) -> Tuple[str, int]:
    """
    Calls an LLM based on the provided model configuration.
    model_config should contain 'type' ('openrouter', 'venice', 'local') and other necessary details.
    """
    model_type = model_config["type"]
    model_name = model_config["name"]
    api_key = model_config.get("api_key")
    referer = model_config.get("referer")
    site_name = model_config.get("site_name")
    base_url = model_config["base_url"]

    url = f"{base_url}/chat/completions"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if referer:
        headers["HTTP-Referer"] = referer
    if site_name:
        headers["X-Title"] = site_name

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": PROMPT.format(tag=tag)
            }
        ],
        "max_tokens": 2000,
        "temperature": 0.9,
        "top_p": 0.9,
        "stop": ["\\d"],
    }

    if model_type == "openrouter":
        payload["reasoning"] = {"exclude": True}
        if model_name == "x-ai/grok-3-mini-beta":
            payload.pop("stop", None)
    elif model_type == "venice":
        payload["venice_parameters"] = {"strip_thinking_response": True,"disable_thinking": True,"enable_web_search": "off","enable_web_citations": False,"include_venice_system_prompt": True}
 # Ensure it's not present if default is True elsewhere
    elif model_type == "local":
        # Local model doesn't need API key in payload or include_reasoning
        pass # No specific removal needed, as it wasn't added for local previously

    for attempt in range(max_retries):
        try:
            # Use requests.post for synchronous calls within asyncio tasks
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                timeout=30
            )

            if response.status_code != 200:
                error_details = f"Received status code {response.status_code} for tag '{tag}' with model '{model_name}'. "
                try:
                    error_json = response.json()
                    # If it's JSON, try to extract specific error message and dump full JSON
                    # error_details += f"API Error: {error_json.get('error', {}).get('message', 'Unknown API error')}. "
                    # error_details += f"Full Response: {json.dumps(error_json, indent=2)}"
                except json.JSONDecodeError:
                    # If not JSON, just dump the raw text content
                    error_details += f"Non-JSON error response. Raw content: {response.text}"

                logging.error(f"{error_json}. Retrying in 10s...")
                await asyncio.sleep(10) # Use await for async sleep
                return "uncategorized", response.status_code

            response_data = response.json()
            logging.debug(f"Full LLM Response for '{tag}' with model '{model_name}': {json.dumps(response_data, indent=2)}")

            if "error" in response_data:
                error_code = response_data["error"].get("code")
                if error_code == 429:
                    wait_time = 60
                    if "metadata" in response_data["error"] and "headers" in response_data["error"]["metadata"]:
                        reset_timestamp = response_data["error"]["metadata"]["headers"].get("X-RateLimit-Reset")
                        if reset_timestamp:
                            wait_time = max(1, float(reset_timestamp) / 1000 - time.time())
                    logging.warning(f"Rate limit error in response for '{tag}' with model '{model_name}'. Waiting {wait_time:.2f}s...")
                    await asyncio.sleep(wait_time)
                    return "uncategorized", 429
                else:
                    logging.error(f"API Error for '{tag}' with model '{model_name}': {response_data['error'].get('message', 'Unknown error')}")
                    return "uncategorized", 500

            try:
                reply = response_data["choices"][0]["message"]["content"].strip()
                if reply in CATEGORY_MAP:
                    return reply, 200
                else:
                    logging.warning(f"Invalid category '{reply}' for tag '{tag}' from model '{model_name}'")
                    return "uncategorized", 200
            except (KeyError, IndexError) as e:
                logging.error(f"Invalid format for '{tag}' with model '{model_name}': {str(e)}")
                return "uncategorized", 500

        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed for '{tag}' with model '{model_name}' (attempt {attempt+1}/{max_retries}): {str(e)}")
            await asyncio.sleep(2 ** attempt) # Use await for async sleep
            return "uncategorized", 503

    return "uncategorized", 503

def get_majority_category(categories: List[str], model_results: Dict[str, str]) -> str:
    valid_categories = [cat for cat in categories if cat != "uncategorized"]

    if not valid_categories:
        return "uncategorized"

    category_counts = Counter(valid_categories)
    max_count = max(category_counts.values())
    majority_categories = [cat for cat, count in category_counts.items() if count == max_count]

    if len(majority_categories) == 1:
        return CATEGORY_MAP[majority_categories[0]]
    else:
        # Tie-breaker: Prioritize Venice's result, then Local, then OpenRouter
        venice_result = model_results.get("qwen3-235b")
        if venice_result in majority_categories:
            return CATEGORY_MAP[venice_result]

        local_result = model_results.get("cognitivecomputations_dolphin-mistral-24b-venice-edition")
        if local_result in majority_categories:
            return CATEGORY_MAP[local_result]

        # If Venice or Local didn't break the tie, just pick the first majority
        logging.warning(f"Tie in categorization, picking arbitrary one from {majority_categories}.")
        return CATEGORY_MAP[majority_categories[0]]

async def categorize_tag(tag: str, openrouter_api_key: str, venice_api_key: str, referer: str, site_name: str) -> str:
    # Select one random OpenRouter model
    selected_openrouter_model = random.choice(OPENROUTER_MODELS)

    # Define the configurations for the three models
    model_configs = [
        {
            "type": "openrouter",
            "name": selected_openrouter_model,
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": openrouter_api_key,
            "referer": referer,
            "site_name": site_name,
        },
        {
            "type": "venice",
            "name": "qwen3-235b",
            "base_url": "https://api.venice.ai/api/v1",
            "api_key": venice_api_key,
            "referer": referer,
            "site_name": site_name,
        },
        {
            "type": "local",
            "name": "cognitivecomputations_dolphin-mistral-24b-venice-edition",
            "base_url": "http://REPLACEME:7045/v1", # Note: Local model using LLMSTUDIO uses /v1
            "api_key": None, # No API key needed for local
            "referer": None, # Not applicable for local
            "site_name": None, # Not applicable for local
        }
    ]

    tasks = [call_llm(tag, config) for config in model_configs]
    results_raw = await asyncio.gather(*tasks) # Run all requests concurrently

    results_processed = []
    model_results_map = {}
    invalid_counts = defaultdict(int)

    for i, (cat_num, status) in enumerate(results_raw):
        model_name = model_configs[i]["name"]
        if status != 200 or cat_num == "uncategorized":
            logging.warning(f"Model '{model_name}' returned status {status} or invalid result for '{tag}'. Skipping.")
            invalid_counts[model_name] += 1
        else:
            results_processed.append(cat_num)
            model_results_map[model_name] = cat_num

    if not results_processed:
        logging.warning(f"No valid results for '{tag}' after all attempts from any model.")
        return "uncategorized"

    final_category = get_majority_category(results_processed, model_results_map)
    logging.info(f"Final category for '{tag}': {final_category} based on votes {results_processed}")

    # Log invalid counts
    for model, count in invalid_counts.items():
        logging.warning(f"Model '{model}' returned invalid category {count} times for '{tag}'.")

    return final_category

async def main():
    # API keys for different services
    OPENROUTER_API_KEY = "REPLACEME"
    VENICE_API_KEY = "REPLACEME"

    REFERER = "sample.org"
    SITE_NAME = "sample"

    with open('all_tags.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        tags = [row[0] for row in reader]

    loop = asyncio.get_event_loop()

    with open('categorized_tags.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Tag', 'Category'])

        with tqdm(total=len(tags), desc="Categorizing tags") as pbar:
            for tag in tags:
                clean_tag = tag.strip()
                if not clean_tag:
                    pbar.update(1)
                    continue

                logging.info(f"Processing tag: {clean_tag}")
                # Call the async categorization function
                category = await categorize_tag(clean_tag, OPENROUTER_API_KEY, VENICE_API_KEY, REFERER, SITE_NAME)
                writer.writerow([clean_tag, category])
                f.flush()
                pbar.update(1)
                # No sleep here as requests are parallelized, only if sequential processing
                # of tags needs a pause.
                # time.sleep(3) # Removed for potential overall speedup


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
