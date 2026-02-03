# LLM Council 🤖

A web application that queries Claude, Gemini, GPT, and Grok simultaneously, then uses AI to analyze their agreements/disagreements and generate cross-commentary.

## Features

- **Parallel Querying**: Sends your prompt to all four LLMs simultaneously
- **AI Analysis**: Claude analyzes the responses to identify agreements and disagreements
- **Cross-Commentary**: Each AI comments on the others' responses
- **Beautiful Web UI**: Clean, modern interface with color-coded responses

## Setup Instructions

### 1. Install Python Dependencies

```bash
cd llm-council
pip install -r requirements.txt
```

Or create a virtual environment (recommended):

```bash
cd llm-council
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get API Keys

You'll need API keys for all four services:

#### Anthropic (Claude)
1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Sign up or log in
3. Navigate to API Keys
4. Create a new API key
5. Add credits to your account (pay-as-you-go)

#### Google (Gemini)
1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Sign in with your Google account
3. Click "Get API Key"
4. Create a new API key
5. **Free tier available!** (60 requests/minute)

#### OpenAI (GPT)
1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up or log in
3. Navigate to API Keys
4. Create a new API key
5. Add credits to your account (separate from ChatGPT Plus)

#### xAI (Grok)
1. Go to [x.ai/api](https://x.ai/api)
2. Sign up or log in with your X/Twitter account
3. Navigate to API Keys
4. Create a new API key
5. Add credits to your account

### 3. Configure Environment Variables

Create a `.env` file in the `llm-council` directory:

```bash
cp .env.example .env
```

Then edit `.env` and add your API keys:

```
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
OPENAI_API_KEY=sk-proj-...
XAI_API_KEY=xai-...
```

### 4. Run the Application

```bash
python app.py
```

Then open your browser to: **http://localhost:5000**

## Usage

1. Enter your prompt in the text area
2. Click "Ask the Council" or press Enter
3. Wait 30-60 seconds while all models are queried
4. View the results:
   - **Initial Responses**: See what each AI said
   - **Analysis**: Claude's analysis of agreements/disagreements
   - **Cross-Commentary**: Each AI comments on the others

## Cost Estimates

Approximate costs per query (varies by response length):

- **Claude (Sonnet 4.5)**: ~$0.02-0.05 per query
- **Gemini (2.0 Flash)**: **FREE** (up to 60 requests/min)
- **GPT (GPT-4o)**: ~$0.01-0.03 per query
- **Grok (Beta)**: ~$0.02-0.04 per query

**Total per query**: ~$0.05-0.12 (with Gemini free tier)

For personal use, this is very affordable. 100 queries = ~$5-12.

## Architecture

```
llm-council/
├── app.py              # Flask web server
├── council.py          # Core LLM orchestration logic
├── requirements.txt    # Python dependencies
├── .env               # API keys (you create this)
├── templates/
│   └── index.html     # Web interface
└── README.md          # This file
```

## Troubleshooting

### "Error querying [Model]"
- Check that your API key is correct in `.env`
- Verify you have credits/quota available
- Check your internet connection

### "ModuleNotFoundError"
- Make sure you ran `pip install -r requirements.txt`
- If using a virtual environment, make sure it's activated

### Slow responses
- Normal! Each query makes 4 initial calls + 5 follow-up calls = 9 LLM requests
- Total time typically 40-90 seconds

## Advanced Usage

### Command Line Testing

You can test the council logic directly:

```bash
python council.py
```

This will run a test query and print results to the console.

### Customize Models

Edit `council.py` to change which models are used:

```python
# Line 21: Change Claude model
model="claude-sonnet-4-5-20250929"

# Line 18: Change Gemini model
self.gemini_model = genai.GenerativeModel("gemini-2.0-flash-exp")

# Line 56: Change GPT model
model="gpt-4o"
```

## License

MIT - Feel free to modify and use as you wish!
