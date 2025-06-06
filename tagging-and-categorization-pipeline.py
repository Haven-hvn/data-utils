#!/usr/bin/env python3
import argparse
import asyncio
import csv
import json
import logging
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.sql import text
from tqdm.asyncio import tqdm

# --- INITIAL SETUP ---
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='pipeline.log',
    filemode='w'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


# --- CONSTANTS AND CONFIGURATION ---

@dataclass
class Config:
    """Holds all configuration parameters for the pipeline."""
    # Scraping params
    scrape_base_url: str = "https://pornbay.org/tags.php"
    scrape_start_page: int = 1
    scrape_end_page: int = 1053
    scrape_workers: int = 15
    scrape_timeout: int = 20
    scraped_tags_file: str = 'all_tags.csv'

    # Categorization params
    categorize_workers: int = 10
    categorize_retries: int = 3
    categorize_timeout: int = 45
    categorized_tags_file: str = 'categorized_tags.csv'
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    venice_api_key: str = os.getenv("VENICE_API_KEY", "")
    local_api_base_url: str = os.getenv("LOCAL_API_BASE_URL", "http://localhost:11434/v1")
    referer: str = 'https://github.com/your-repo' # Be a good internet citizen
    site_name: str = 'Pipeline'

    # Database params
    db_connection_string: str = os.getenv("DB_CONNECTION_STRING", "")
    db_insert_batch_size: int = 100
    db_insert_batch_timeout: int = 5 # seconds

    # Control params
    steps: List[str] = field(default_factory=lambda: ["scrape", "categorize", "insert"])


PROMPT = """
You are a skilled adult content categorizer. Your task is to assign EACH TAG to a SPECIFIC CATEGORY from the list below. You MUST choose from these categories ONLY:

1. ACTION - Accessories, 2. ACTION - Acts, 3. ACTION - Finishers, 4. PEOPLE - Age Group, 5. PEOPLE - Ass, 6. PEOPLE - Body Type, 7. PEOPLE - Breasts, 8. PEOPLE - Clothing, 9. PEOPLE - Genitals, 10. PEOPLE - Hair Color, 11. PEOPLE - Hair Style, 12. PEOPLE - Height, 13. PEOPLE - Piercings, 14. PEOPLE - Race, 15. PEOPLE - Skin Tone, 16. PEOPLE - Tattoos, 17. PEOPLE - Actor, 18. SCENE - Audio, 19. SCENE - Genres, 20. SCENE - Group Makeup, 21. SCENE - Known File Quality, 22. SCENE - Locations, 23. SCENE - Misc, 24. SCENE - Moods, 25. SCENE - Orientation, 26. SCENE - Relations, 27. SCENE - Roles, 28. SCENE - Sources, 29. SCENE - Surfaces, 30. SCENE - Tease Mechanics, 31. SCENE - Teledildonics, 32. SCENE - Themes, 33. Other

This categorization system is designed to highlight SPECIFIC MOMENTS OR SECTIONS WITHIN ADULT VIDEOS. Respond ONLY with the NUMBER of the category.

Q: blowjob => 2
Q: cowgirl => 2
Q: kitchen sex => 22
Q: hardcore => 19
Q: 1080p => 21
Q: doggy style => 2
Q: Brunette => 10
Q: gay => 25

Categorize this tag: "{tag}"
""".strip()

CATEGORY_MAP = {
    "1": "ACTION - Accessories", "2": "ACTION - Acts", "3": "ACTION - Finishers", "4": "PEOPLE - Age Group", "5": "PEOPLE - Ass", "6": "PEOPLE - Body Type", "7": "PEOPLE - Breasts", "8": "PEOPLE - Clothing", "9": "PEOPLE - Genitals", "10": "PEOPLE - Hair Color", "11": "PEOPLE - Hair Style", "12": "PEOPLE - Height", "13": "PEOPLE - Piercings", "14": "PEOPLE - Race", "15": "PEOPLE - Skin Tone", "16": "PEOPLE - Tattoos", "17": "PEOPLE - Actor", "18": "SCENE - Audio", "19": "SCENE - Genres", "20": "SCENE - Group Makeup", "21": "SCENE - Known File Quality", "22": "SCENE - Locations", "23": "SCENE - Misc", "24": "SCENE - Moods", "25": "SCENE - Orientation", "26": "SCENE - Relations", "27": "SCENE - Roles", "28": "SCENE - Sources", "29": "SCENE - Surfaces", "30": "SCENE - Tease Mechanics", "31": "SCENE - Teledildonics", "32": "SCENE - Themes", "33": "Other"
}

OPENROUTER_MODELS = ["gryphe/mythomax-l2-13b", "mistralai/mistral-7b-instruct", "nousresearch/nous-hermes-2-mixtral-8x7b-dpo"]


# --- HELPER FUNCTIONS ---

def clean_tag_text(tag: str) -> str:
    """Cleans raw tag text by removing special chars and stripping whitespace."""
    return tag.replace('.', ' ').strip().rstrip('*')

def get_majority_category(categories: List[str], model_results: Dict[str, str]) -> str:
    """Determines the final category from a list of votes, with tie-breaking logic."""
    valid_categories = [cat for cat in categories if cat != "uncategorized"]
    if not valid_categories:
        return "uncategorized"

    category_counts = Counter(valid_categories)
    max_count = max(category_counts.values())
    majority_categories = [cat for cat, count in category_counts.items() if count == max_count]

    if len(majority_categories) == 1:
        return CATEGORY_MAP.get(majority_categories[0], "uncategorized")

    # Tie-breaker logic: Prioritize more reliable models
    tie_breaker_priority = ["qwen2:72b-instruct-q4_K_M", "cognitivecomputations/dolphin-2.9-llama-3-8b", *OPENROUTER_MODELS]
    for model_name in tie_breaker_priority:
        result = model_results.get(model_name)
        if result and result in majority_categories:
            logging.debug(f"Tie broken by {model_name}, selecting {result}")
            return CATEGORY_MAP.get(result, "uncategorized")

    logging.warning(f"Tie remains among {majority_categories}. Picking first: {majority_categories[0]}")
    return CATEGORY_MAP.get(majority_categories[0], "uncategorized")


# --- STAGE 1: SCRAPING ---

async def scrape_worker(
    worker_id: int,
    client: httpx.AsyncClient,
    config: Config,
    page_queue: asyncio.Queue,
    tags_queue: asyncio.Queue,
    scraped_tags_set: Set[str],
    pbar: tqdm,
):
    """A worker that fetches pages, parses tags, and puts them in a queue."""
    logging.info(f"[Scraper-{worker_id}] Started.")
    while True:
        try:
            page_num = await page_queue.get()
            url = f"{config.scrape_base_url}?page={page_num}"
            try:
                response = await client.get(url, timeout=config.scrape_timeout)
                response.raise_for_status()
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                logging.error(f"[Scraper-{worker_id}] Error fetching page {page_num}: {e}")
                page_queue.task_done()
                pbar.update(1)
                continue

            soup = BeautifulSoup(response.content, 'html.parser')
            new_tags_on_page = []
            for table in soup.find_all('table', class_='box shadow'):
                for tr in table.find_all('tr', class_=['rowa', 'rowb']):
                    a_tag = tr.find('td').find('a')
                    if not a_tag:
                        continue
                    
                    cleaned_tag = clean_tag_text(a_tag.text)
                    if cleaned_tag and cleaned_tag not in scraped_tags_set:
                        scraped_tags_set.add(cleaned_tag)
                        new_tags_on_page.append(cleaned_tag)
                        await tags_queue.put(cleaned_tag)

            if new_tags_on_page:
                with open(config.scraped_tags_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    for tag in new_tags_on_page:
                        writer.writerow([tag])
            
            pbar.update(1)
            page_queue.task_done()
        except asyncio.CancelledError:
            logging.info(f"[Scraper-{worker_id}] Cancelled.")
            break
        except Exception as e:
            logging.error(f"[Scraper-{worker_id}] Unexpected error: {e}", exc_info=True)
            page_queue.task_done()


# --- STAGE 2: CATEGORIZATION ---

async def llm_api_call(client: httpx.AsyncClient, tag: str, model_config: Dict[str, Any], config: Config) -> Tuple[str, int]:
    """Makes a single API call to an LLM provider."""
    model_name = model_config["name"]
    for attempt in range(config.categorize_retries):
        try:
            response = await client.post(
                url=model_config["url"],
                headers=model_config["headers"],
                json=model_config["payload_template"](tag),
                timeout=config.categorize_timeout,
            )

            if response.status_code == 429: # Rate limit
                retry_after = int(response.headers.get("Retry-After", 30))
                logging.warning(f"Rate limit for {model_name} on tag '{tag}'. Waiting {retry_after}s.")
                await asyncio.sleep(retry_after)
                continue

            response.raise_for_status()
            data = response.json()
            reply = data["choices"][0]["message"]["content"].strip()

            # Find the first number in the reply
            import re
            match = re.search(r'\d+', reply)
            if match and match.group(0) in CATEGORY_MAP:
                return match.group(0), 200
            
            logging.warning(f"Invalid category '{reply}' from {model_name} for tag '{tag}'")
            return "uncategorized", 200
        except (httpx.RequestError, httpx.HTTPStatusError, KeyError, IndexError, json.JSONDecodeError) as e:
            logging.error(f"API call failed for {model_name} on tag '{tag}' (attempt {attempt+1}): {e}")
            await asyncio.sleep(2 ** attempt) # Exponential backoff

    return "uncategorized", 500


async def categorize_worker(
    worker_id: int,
    client: httpx.AsyncClient,
    config: Config,
    tags_queue: asyncio.Queue,
    db_queue: asyncio.Queue,
    pbar: tqdm,
):
    """A worker that takes tags, gets categorizations from LLMs, and puts results in a DB queue."""
    logging.info(f"[Categorizer-{worker_id}] Started.")
    
    # Define model configurations
    model_configs = [
        {
            "type": "openrouter", "name": random.choice(OPENROUTER_MODELS),
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "headers": {"Authorization": f"Bearer {config.openrouter_api_key}", "HTTP-Referer": config.referer, "X-Title": config.site_name},
            "payload_template": lambda tag: {"model": model_configs[0]["name"], "messages": [{"role": "user", "content": PROMPT.format(tag=tag)}], "max_tokens": 10},
        },
        {
            "type": "venice", "name": "qwen2:72b-instruct-q4_K_M",
            "url": "https://api.venice.ai/v1/chat/completions",
            "headers": {"Authorization": f"Bearer {config.venice_api_key}"},
            "payload_template": lambda tag: {"model": "qwen2:72b-instruct-q4_K_M", "messages": [{"role": "user", "content": PROMPT.format(tag=tag)}], "max_tokens": 10},
        },
        {
            "type": "local", "name": "cognitivecomputations/dolphin-2.9-llama-3-8b",
            "url": config.local_api_base_url + "/chat/completions",
            "headers": {},
            "payload_template": lambda tag: {"model": "cognitivecomputations/dolphin-2.9-llama-3-8b", "messages": [{"role": "user", "content": PROMPT.format(tag=tag)}], "max_tokens": 10, "stream": False},
        },
    ]

    while True:
        try:
            tag = await tags_queue.get()
            if tag is None: # Sentinel value
                tags_queue.task_done()
                break

            tasks = [llm_api_call(client, tag, mc, config) for mc in model_configs if mc["headers"].get("Authorization") != "Bearer "]
            results_raw = await asyncio.gather(*tasks)

            categories = [cat for cat, status in results_raw if status == 200]
            model_results_map = {model_configs[i]["name"]: cat for i, (cat, status) in enumerate(results_raw) if status == 200}
            
            final_category = get_majority_category(categories, model_results_map)
            
            logging.debug(f"Tag '{tag}': Votes: {categories} -> Final: {final_category}")
            if final_category != "uncategorized":
                await db_queue.put((tag.strip().title(), final_category))
            else:
                logging.warning(f"Could not categorize tag '{tag}', skipping.")

            pbar.update(1)
            tags_queue.task_done()
        except asyncio.CancelledError:
            logging.info(f"[Categorizer-{worker_id}] Cancelled.")
            break
        except Exception as e:
            logging.error(f"[Categorizer-{worker_id}] Unexpected error: {e}", exc_info=True)
            tags_queue.task_done()


# --- STAGE 3: DATABASE INSERTION ---

async def db_insert_worker(
    config: Config,
    db_queue: asyncio.Queue,
    pbar: tqdm,
):
    """A worker that batches and inserts categorized tags into the database."""
    logging.info("[DB-Worker] Started.")
    if not config.db_connection_string:
        logging.error("[DB-Worker] Database connection string not set. Exiting.")
        return

    engine = create_async_engine(config.db_connection_string)
    batch = []
    last_insert_time = time.time()

    async def insert_batch(current_batch):
        if not current_batch:
            return 0
        
        # 1. Write to CSV for backup
        try:
            with open(config.categorized_tags_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(current_batch)
        except IOError as e:
            logging.error(f"[DB-Worker] Failed to write to {config.categorized_tags_file}: {e}")

        # 2. Insert into database
        insert_count = 0
        try:
            async with engine.connect() as conn:
                # Filter out categories to skip
                records_to_insert = [
                    {"action_name": tag, "description": cat}
                    for tag, cat in current_batch
                    if cat.lower() not in ['other', 'scene - known file quality', 'scene - sources']
                ]
                
                if not records_to_insert:
                    logging.info(f"[DB-Worker] Batch of {len(current_batch)} contained only skippable categories.")
                    return 0

                stmt = text("""
                    INSERT INTO "Action" (action_name, description)
                    VALUES (:action_name, :description)
                    ON CONFLICT (action_name) DO NOTHING;
                """)
                
                result = await conn.execute(stmt, records_to_insert)
                await conn.commit()
                insert_count = result.rowcount
                logging.info(f"[DB-Worker] Inserted {insert_count}/{len(records_to_insert)} new records.")
                pbar.update(len(current_batch))

        except SQLAlchemyError as e:
            logging.error(f"[DB-Worker] Database error during batch insert: {e}", exc_info=True)
        except Exception as e:
            logging.error(f"[DB-Worker] Unexpected error in DB worker: {e}", exc_info=True)
        
        return len(current_batch) # Return total processed items for progress

    while True:
        try:
            item = await asyncio.wait_for(db_queue.get(), timeout=config.db_insert_batch_timeout)
            if item is None: # Sentinel value
                if batch:
                    await insert_batch(batch)
                db_queue.task_done()
                break
            
            batch.append(item)
            if len(batch) >= config.db_insert_batch_size:
                await insert_batch(batch)
                batch = []
                last_insert_time = time.time()

        except asyncio.TimeoutError:
            if batch:
                logging.info("[DB-Worker] Batch timeout, inserting partial batch.")
                await insert_batch(batch)
                batch = []
                last_insert_time = time.time()
        except asyncio.CancelledError:
            if batch:
                await insert_batch(batch)
            logging.info("[DB-Worker] Cancelled.")
            break

    await engine.dispose()
    logging.info("[DB-Worker] Finished.")


# --- MAIN ORCHESTRATOR ---

async def run_pipeline(config: Config):
    """Sets up and runs the entire data processing pipeline."""
    
    # --- Queues for inter-stage communication ---
    page_queue = asyncio.Queue()
    tags_to_categorize_queue = asyncio.Queue(maxsize=config.scrape_workers * 2)
    tags_to_insert_queue = asyncio.Queue(maxsize=config.categorize_workers * config.db_insert_batch_size)
    
    # --- Shared state ---
    scraped_tags_set = set()
    
    # --- Progress Bars ---
    pbars = {
        'scrape': tqdm(total=0, desc="Scraping Pages", unit="page"),
        'categorize': tqdm(total=0, desc="Categorizing Tags", unit="tag"),
        'insert': tqdm(total=0, desc="Inserting to DB", unit="tag")
    }

    # --- Step 1: SCRAPE ---
    if 'scrape' in config.steps:
        # Prepare file
        with open(config.scraped_tags_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Tag Name'])
        
        num_pages = config.scrape_end_page - config.scrape_start_page + 1
        pbars['scrape'].total = num_pages
        for i in range(config.scrape_start_page, config.scrape_end_page + 1):
            await page_queue.put(i)

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'}
        async with httpx.AsyncClient(headers=headers, http2=True, follow_redirects=True) as client:
            scraper_tasks = [
                asyncio.create_task(scrape_worker(i, client, config, page_queue, tags_to_categorize_queue, scraped_tags_set, pbars['scrape']))
                for i in range(config.scrape_workers)
            ]
            await page_queue.join()
            for task in scraper_tasks:
                task.cancel()
            await asyncio.gather(*scraper_tasks, return_exceptions=True)
            pbars['scrape'].close()
            logging.info(f"Scraping finished. Found {len(scraped_tags_set)} unique tags.")
    else:
        # If not scraping, load tags from file for the next step
        if 'categorize' in config.steps:
            try:
                with open(config.scraped_tags_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader) # Skip header
                    for row in reader:
                        await tags_to_categorize_queue.put(row[0])
                logging.info(f"Loaded {tags_to_categorize_queue.qsize()} tags from {config.scraped_tags_file}")
            except FileNotFoundError:
                logging.error(f"Scraping step was skipped but '{config.scraped_tags_file}' not found.")
                return

    # --- Step 2: CATEGORIZE ---
    if 'categorize' in config.steps:
        pbars['categorize'].total = tags_to_categorize_queue.qsize()
        for _ in range(config.categorize_workers):
            await tags_to_categorize_queue.put(None) # Sentinels for workers
        
        # Prepare file
        with open(config.categorized_tags_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Tag', 'Category'])

        async with httpx.AsyncClient(http2=True) as client:
            categorizer_tasks = [
                asyncio.create_task(categorize_worker(i, client, config, tags_to_categorize_queue, tags_to_insert_queue, pbars['categorize']))
                for i in range(config.categorize_workers)
            ]
            await tags_to_categorize_queue.join()
            for task in categorizer_tasks:
                task.cancel()
            await asyncio.gather(*categorizer_tasks, return_exceptions=True)
            pbars['categorize'].close()
            logging.info("Categorization finished.")
    else:
        # If not categorizing, load from file for insert step
        if 'insert' in config.steps:
            try:
                with open(config.categorized_tags_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader) # Skip header
                    for row in reader:
                        # row[0] is tag (already title-cased), row[1] is category
                        await tags_to_insert_queue.put((row[0], row[1]))
                logging.info(f"Loaded {tags_to_insert_queue.qsize()} categorized tags from {config.categorized_tags_file}")
            except FileNotFoundError:
                logging.error(f"Categorization step was skipped but '{config.categorized_tags_file}' not found.")
                return


    # --- Step 3: INSERT ---
    if 'insert' in config.steps:
        pbars['insert'].total = tags_to_insert_queue.qsize()
        await tags_to_insert_queue.put(None) # Sentinel for the single DB worker

        db_task = asyncio.create_task(db_insert_worker(config, tags_to_insert_queue, pbars['insert']))
        await tags_to_insert_queue.join()
        await db_task
        pbars['insert'].close()
        logging.info("Database insertion finished.")
    
    for pbar in pbars.values():
        if not pbar.disable:
            pbar.close()

def main():
    parser = argparse.ArgumentParser(description="A multi-stage, parallel pipeline for scraping, categorizing, and storing tags.")
    parser.add_argument(
        '--steps', 
        nargs='+', 
        choices=['scrape', 'categorize', 'insert'], 
        default=['scrape', 'categorize', 'insert'],
        help="Which steps of the pipeline to run."
    )
    parser.add_argument('--scrape-pages', type=int, nargs=2, metavar=('START', 'END'), help="Range of pages to scrape (e.g., 1 50).")
    parser.add_argument('--scrape-workers', type=int, help="Number of concurrent scrapers.")
    parser.add_argument('--categorize-workers', type=int, help="Number of concurrent LLM categorizers.")
    parser.add_argument('--db-batch-size', type=int, help="Number of records to insert into the DB at once.")
    
    args = parser.parse_args()
    config = Config()

    # Override config with command-line arguments if provided
    if args.steps:
        config.steps = args.steps
    if args.scrape_pages:
        config.scrape_start_page, config.scrape_end_page = args.scrape_pages
    if args.scrape_workers:
        config.scrape_workers = args.scrape_workers
    if args.categorize_workers:
        config.categorize_workers = args.categorize_workers
    if args.db_batch_size:
        config.db_insert_batch_size = args.db_batch_size
    
    logging.info(f"Starting pipeline with steps: {config.steps}")
    logging.info(f"Configuration: {config}")

    # Validate required configs for selected steps
    if 'categorize' in config.steps and (not config.openrouter_api_key or not config.venice_api_key):
        logging.error("Categorize step requires OPENROUTER_API_KEY and VENICE_API_KEY to be set in .env")
        return
    if 'insert' in config.steps and not config.db_connection_string:
        logging.error("Insert step requires DB_CONNECTION_STRING to be set in .env")
        return

    try:
        asyncio.run(run_pipeline(config))
        logging.info("Pipeline completed successfully.")
    except KeyboardInterrupt:
        logging.warning("Pipeline interrupted by user.")
    except Exception as e:
        logging.critical(f"A critical error occurred in the pipeline: {e}", exc_info=True)


if __name__ == "__main__":
    main()
