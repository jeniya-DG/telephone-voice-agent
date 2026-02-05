## Project Structure

```
├── common/
│   ├── agent_functions.py    # Function definitions and routing
│   ├── business_logic.py     # Core function implementations
│   ├── config.py             # Configuration settings
│   ├── log_formatter.py      # Logger setup
├── client.py             # WebSocket client and message handling
```


## Sample .evn
```
DEEPGRAM_API_KEY=xxx

# Kapa AI
KAPA_PROJECT_ID=xx
KAPA_API_KEY=xxx
```

## Application Usage

1. Run the client:
   ```bash
   python client.py
   ```

> The application will be available at http://localhost:8000



