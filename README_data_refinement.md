# Data Refinement Script

This script connects to a PostgreSQL database, identifies rows in the "Action" table where the description starts with "PEOPLE", and uses a local LLM to determine if the action_name represents a person's name or stage name. If it does, the row is immediately deleted from the database.

## Prerequisites

1. **Database Setup**: Update the `DB_CONNECTION_STRING` in your `.env` file with actual PostgreSQL credentials
2. **Local LLM Server**: Ensure LM Studio is running on `localhost:7045` with the model `cognitivecomputations_dolphin-mistral-24b-venice-edition`
3. **Python Dependencies**: Already installed (`asyncpg`, `python-dotenv`, `aiohttp`)

## How it Works

1. **Database Connection**: Connects to PostgreSQL using the connection string from `.env`
2. **Query**: Selects `action_id` and `action_name` from the "Action" table where `description LIKE 'PEOPLE%'`
3. **Concurrent LLM Assessment**: Processes multiple rows simultaneously using async HTTP requests to determine if action_name represents a person's name or stage name
4. **Immediate Deletion**: If the LLM responds "Yes", the row is immediately deleted from the database after each batch
5. **Progress Tracking**: Shows detailed progress and final statistics

## Usage

```bash
python data-refinement.py
```

## Configuration

### Database Connection (.env file)
Update the `DB_CONNECTION_STRING` in your `.env` file:
```
DB_CONNECTION_STRING="postgresql+asyncpg://username:password@host:port/database?ssl=require"
```

### Local LLM Server
The script expects LM Studio to be running on:
- URL: `http://localhost:7045/v1/chat/completions`
- Model: `cognitivecomputations_dolphin-mistral-24b-venice-edition`

## Output Example

```
Starting data refinement process...
Successfully connected to PostgreSQL database
Found 50 rows with description LIKE 'PEOPLE%'
Processing 50 rows with 10 concurrent LLM requests...

Processing batch 1/5 (10 items)...
'John Smith' identified as person name -> Deleting
Deleted row with action_id=123, action_name='John Smith'
'Lady Gaga' identified as person name -> Deleting
Deleted row with action_id=125, action_name='Lady Gaga'

Processing batch 2/5 (10 items)...
...

Processing complete!
Total rows processed: 50
Total rows deleted: 15
Total rows kept: 35
Database connection closed
```

## Performance Optimization

The script uses concurrent processing for maximum performance:
- **Concurrent LLM Requests**: Processes 10 LLM requests simultaneously by default
- **Batch Processing**: Groups requests into batches to avoid overwhelming the server
- **Async HTTP**: Uses aiohttp for non-blocking HTTP requests
- **Immediate Deletion**: Deletes rows immediately after each batch completes

## LLM Configuration for Maximum Accuracy

The script uses optimized parameters for deterministic and accurate responses:
- **Temperature**: 0.0 (completely deterministic)
- **Top-p**: 0.1 (focused sampling)
- **Seed**: 42 (reproducible results)
- **Max tokens**: 5 (forces concise responses)
- **System prompt**: Provides clear classification instructions
- **Detailed criteria**: Explicit examples of what constitutes person names vs other text

## Safety Features

- **Validation**: Checks for placeholder values in database connection string
- **Error Handling**: Comprehensive error handling for database and API calls
- **Batch Processing**: Processes in controlled batches to avoid overwhelming the LLM server
- **Detailed Logging**: Shows LLM responses and decisions for identified person names
- **Connection Management**: Properly closes database connections

## Notes

- The script processes multiple rows concurrently (10 by default) for significantly improved performance
- Each LLM call has a 30-second timeout
- The script uses temperature 0.0 and seed 42 for maximum consistency and reproducibility
- Database queries use parameterized statements to prevent SQL injection
- Enhanced prompt engineering with specific criteria reduces classification errors
- Concurrent processing can be adjusted by modifying the `concurrent_requests` parameter
- Performance improvement: ~10x faster than sequential processing for large datasets
