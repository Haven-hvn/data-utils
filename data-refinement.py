import os
import asyncio
import asyncpg
import aiohttp
import json
from dotenv import load_dotenv
from typing import List, Tuple

# Load environment variables
load_dotenv()

class DataRefinement:
    def __init__(self):
        self.db_connection_string = os.getenv('DB_CONNECTION_STRING')
        self.local_llm_url = "http://localhost:7045/v1/chat/completions"
        self.model_name = "cognitivecomputations_dolphin-mistral-24b-venice-edition"
        
        if not self.db_connection_string:
            raise ValueError("DB_CONNECTION_STRING not found in environment variables")
    
    async def connect_to_db(self):
        """Connect to PostgreSQL database"""
        try:
            # Validate connection string format
            if 'example.com' in self.db_connection_string or 'xxxx' in self.db_connection_string:
                print("Error: Please update the DB_CONNECTION_STRING in .env file with actual database credentials")
                return False
            
            # Parse the connection string to extract components
            # Format: postgresql+asyncpg://user:password@host:port/dbname?ssl=require
            connection_string = self.db_connection_string.replace('postgresql+asyncpg://', 'postgresql://')
            self.connection = await asyncpg.connect(connection_string)
            print("Successfully connected to PostgreSQL database")
            return True
        except Exception as e:
            print(f"Error connecting to database: {e}")
            print("Please check your database connection string in the .env file")
            return False
    
    async def close_db_connection(self):
        """Close database connection"""
        if hasattr(self, 'connection'):
            await self.connection.close()
            print("Database connection closed")
    
    async def get_people_actions(self) -> List[Tuple[int, str]]:
        """Get action_id and action_name from Action table where description LIKE 'PEOPLE%'"""
        try:
            query = """
            SELECT action_id, action_name 
            FROM "Action" 
            WHERE description LIKE 'PEOPLE%'
            """
            rows = await self.connection.fetch(query)
            print(f"Found {len(rows)} rows with description LIKE 'PEOPLE%'")
            return [(row['action_id'], row['action_name']) for row in rows]
        except Exception as e:
            print(f"Error querying database: {e}")
            return []
    
    async def test_llm_connection(self) -> bool:
        """Test if the LLM server is accessible"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.local_llm_url.replace('/v1/chat/completions', '/v1/models'),
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        print(f"✓ LLM server is accessible at {self.local_llm_url}")
                        return True
                    else:
                        print(f"✗ LLM server returned status {response.status}")
                        return False
        except Exception as e:
            print(f"✗ Cannot connect to LLM server at {self.local_llm_url}: {e}")
            return False

    async def is_person_name_async(self, session: aiohttp.ClientSession, action_id: int, action_name: str) -> Tuple[int, str, bool]:
        """Use local LLM to determine if action_name is a person's name or stage name"""
        # print(f"Analyzing '{action_name}' (ID: {action_id}) with LLM...")
        try:
            prompt = f"""Analyze the following text and determine if it represents a person's name or stage name.

Text to analyze: "{action_name}"

Consider as person names if they meet ANY of these criteria:

1. **Standard Full Names**: Two or more words that follow typical name patterns:
   - First name + Last name (e.g., "John Smith", "Mary Johnson", "Alexis Crystal", "Lauren Phillips")
   - First + Middle + Last name (e.g., "John Michael Smith")
   - Names with common prefixes/suffixes (e.g., "Dr. Smith", "John Jr.", "Mary O'Connor")

2. **Well-known Stage Names**: 
   - Multi-word stage names (e.g., "Lady Gaga", "The Rock")
   - Single famous names (e.g., "Madonna", "Cher", "Prince", "Beyoncé")
   - Artist names with clear personal identity (e.g., "Dr. Dre", "50 Cent")

3. **Name Pattern Recognition**:
   - Capitalized words that follow human naming conventions
   - Common first names paired with surnames
   - Names that sound like real people (even if not famous)

Do NOT consider as person names:
- Single common words without clear name context (e.g., "Slim", "Fast", "Big", "Cool", "Quick")
- Generic action descriptions (e.g., "Create Account", "Delete File", "Update Profile")
- Company/brand names (e.g., "Microsoft", "Apple Inc.")
- Technical terms or system functions
- Common nouns or adjectives standing alone

Key principle: If it looks like a human name with proper capitalization and name structure, it's likely a person's name, even if the person isn't famous.

Respond with exactly one word: "Yes" if it represents a person's name or stage name, "No" if it doesn't."""
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a precise classifier. Always respond with exactly 'Yes' or 'No' based on the criteria provided."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.0,
                "top_p": 0.1,
                "max_tokens": 5,
                "seed": 42
            }
            
            async with session.post(
                self.local_llm_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)  # Increased timeout to 2 minutes
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    answer = result['choices'][0]['message']['content'].strip().lower()
                    is_person = answer.startswith('yes')
                    return (action_id, action_name, is_person)
                else:
                    try:
                        text = await response.text()
                        print(f"Error calling LLM API for '{action_name}':")
                        print(f"  - HTTP Status: {response.status}")
                        print(f"  - Response: {text}")
                        print(f"  - URL: {self.local_llm_url}")
                        print(f"  - Model: {self.model_name}")
                    except Exception as read_error:
                        print(f"Error calling LLM API for '{action_name}':")
                        print(f"  - HTTP Status: {response.status}")
                        print(f"  - Could not read response text: {read_error}")
                    return (action_id, action_name, False)
                    
        except asyncio.TimeoutError as e:
            print(f"Timeout error calling LLM for '{action_name}': {e}")
            print(f"  - Request timed out after 120 seconds")
            return (action_id, action_name, False)
        except aiohttp.ClientError as e:
            print(f"HTTP client error calling LLM for '{action_name}': {e}")
            print(f"  - Error type: {type(e).__name__}")
            print(f"  - LLM URL: {self.local_llm_url}")
            return (action_id, action_name, False)
        except json.JSONDecodeError as e:
            print(f"JSON decode error calling LLM for '{action_name}': {e}")
            print(f"  - Failed to parse LLM response as JSON")
            return (action_id, action_name, False)
        except KeyError as e:
            print(f"Key error calling LLM for '{action_name}': {e}")
            print(f"  - Missing expected key in LLM response")
            return (action_id, action_name, False)
        except Exception as e:
            print(f"Unexpected error calling LLM for '{action_name}': {e}")
            print(f"  - Error type: {type(e).__name__}")
            print(f"  - Error details: {str(e)}")
            import traceback
            print(f"  - Traceback: {traceback.format_exc()}")
            return (action_id, action_name, False)
    
    async def delete_action_row(self, action_id: int, action_name: str) -> bool:
        """Delete a row from the Action table by action_id"""
        try:
            query = 'DELETE FROM "Action" WHERE action_id = $1'
            result = await self.connection.execute(query, action_id)
            print(f"Deleted row with action_id={action_id}, action_name='{action_name}'")
            return True
        except Exception as e:
            print(f"Error deleting row with action_id={action_id}: {e}")
            return False
    
    async def process_people_actions(self, concurrent_requests: int = 3):
        """Main processing function with concurrent LLM requests"""
        print("Starting data refinement process...")
        
        # Test LLM connection first
        print("Testing LLM server connection...")
        if not await self.test_llm_connection():
            print("Cannot proceed without LLM server. Please ensure your local LLM is running.")
            return
        
        # Connect to database
        if not await self.connect_to_db():
            return
        
        try:
            # Get all rows with PEOPLE% description
            people_actions = await self.get_people_actions()
            
            if not people_actions:
                print("No rows found with description LIKE 'PEOPLE%'")
                return
            
            print(f"Processing {len(people_actions)} rows with {concurrent_requests} concurrent LLM requests...")
            
            deleted_count = 0
            
            # Create HTTP session for concurrent requests
            async with aiohttp.ClientSession() as session:
                # Process in batches to avoid overwhelming the LLM server
                for i in range(0, len(people_actions), concurrent_requests):
                    batch = people_actions[i:i + concurrent_requests]
                    
                    # Create concurrent tasks for LLM requests
                    tasks = [
                        self.is_person_name_async(session, action_id, action_name)
                        for action_id, action_name in batch
                    ]
                    
                    # Wait for all LLM responses in this batch
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Process results and delete rows immediately
                    for result in results:
                        if isinstance(result, Exception):
                            print(f"Error in batch processing: {result}")
                            continue
                            
                        action_id, action_name, is_person = result
                        
                        if is_person:
                            print(f"\nProcessing batch {i//concurrent_requests + 1}/{(len(people_actions) + concurrent_requests - 1)//concurrent_requests} ({len(batch)} items)...")
                            print(f"'{action_name}' identified as person name -> Deleting")
                            if await self.delete_action_row(action_id, action_name):
                                deleted_count += 1
            
            print(f"\nProcessing complete!")
            print(f"Total rows processed: {len(people_actions)}")
            print(f"Total rows deleted: {deleted_count}")
            print(f"Total rows kept: {len(people_actions) - deleted_count}")
            
        finally:
            await self.close_db_connection()

async def main():
    """Main function"""
    try:
        refiner = DataRefinement()
        await refiner.process_people_actions()
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    asyncio.run(main())
