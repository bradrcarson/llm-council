"""
Flask web application for LLM Council
"""
from flask import Flask, render_template, request, jsonify, Response
from functools import wraps
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()
from werkzeug.utils import secure_filename
from council import LLMCouncil
import PyPDF2
import docx
from PIL import Image
import io

app = Flask(__name__)

# Simple password protection - set COUNCIL_PASSWORD in .env (optional)
COUNCIL_PASSWORD = os.getenv('COUNCIL_PASSWORD')

def check_auth(password):
    return password == COUNCIL_PASSWORD

def authenticate():
    return Response(
        'Access denied. Please enter the correct password.', 401,
        {'WWW-Authenticate': 'Basic realm="Council of LLMs"'})

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not COUNCIL_PASSWORD:  # No password set, allow access
            return f(*args, **kwargs)
        auth = request.authorization
        if not auth or not check_auth(auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = '/tmp/council_uploads'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Lazy initialization of council to avoid crashes on import
council = None

def get_council():
    global council
    if council is None:
        council = LLMCouncil()
    return council

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(filepath):
    """Extract text content from various file types"""
    ext = filepath.rsplit('.', 1)[1].lower()

    if ext == 'txt':
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

    elif ext == 'pdf':
        text = []
        with open(filepath, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text.append(page.extract_text())
        return '\n\n'.join(text)

    elif ext == 'docx':
        doc = docx.Document(filepath)
        return '\n\n'.join([paragraph.text for paragraph in doc.paragraphs])

    elif ext in ['png', 'jpg', 'jpeg', 'gif']:
        # For images, return a placeholder - we'll handle image analysis separately
        return f"[Image file: {os.path.basename(filepath)}]"

    return ""


@app.route('/debug-env')
@requires_auth
def debug_env():
    """Debug endpoint to check environment variables"""
    keys = ['ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'OPENAI_API_KEY', 'XAI_API_KEY',
            'DEEPSEEK_API_KEY', 'OPENROUTER_API_KEY', 'COUNCIL_PASSWORD']
    status = {}
    for key in keys:
        val = os.getenv(key)
        if val:
            status[key] = f"SET ({len(val)} chars, starts with {val[:8]}...)"
        else:
            status[key] = "NOT SET"
    return jsonify(status)

@app.route('/')
@requires_auth
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/query', methods=['POST'])
@requires_auth
def query():
    """Handle council query requests"""
    conversation_history = None
    original_prompt = ''

    # Check if this is a multipart request (with files)
    if request.content_type and 'multipart/form-data' in request.content_type:
        prompt = request.form.get('prompt', '')
        original_prompt = prompt  # Store the original user prompt
        include_analysis = request.form.get('includeAnalysis', 'true').lower() == 'true'
        include_commentary = request.form.get('includeCrossCommentary', 'true').lower() == 'true'
        conversation_history_json = request.form.get('conversationHistory')
        if conversation_history_json:
            import json
            conversation_history = json.loads(conversation_history_json)

        files = request.files.getlist('documents')

        # Extract text from uploaded files
        document_texts = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                text = extract_text_from_file(filepath)
                if text:
                    document_texts.append(f"=== Document: {filename} ===\n{text}")

                # Clean up
                os.remove(filepath)

        # Combine prompt with document content
        if document_texts:
            full_prompt = f"{prompt}\n\n{'=' * 60}\nATTACHED DOCUMENTS:\n{'=' * 60}\n\n" + "\n\n".join(document_texts)
        else:
            full_prompt = prompt
    else:
        # Regular JSON request
        data = request.get_json()
        full_prompt = data.get('prompt', '')
        original_prompt = full_prompt  # For non-file requests, they're the same
        include_analysis = data.get('includeAnalysis', True)
        include_commentary = data.get('includeCrossCommentary', True)
        conversation_history = data.get('conversationHistory')

    if not full_prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    try:
        # Run the full council session with optional components and conversation history
        result = asyncio.run(get_council().full_council_session(full_prompt, include_analysis, include_commentary, conversation_history))
        # Replace the full prompt with the original user prompt for display
        result['original_prompt'] = original_prompt
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("LLM COUNCIL - Starting Web Interface")
    print("="*60)
    print("\nMake sure you have created a .env file with your API keys:")
    print("  - ANTHROPIC_API_KEY")
    print("  - GOOGLE_API_KEY")
    print("  - OPENAI_API_KEY")
    print("  - XAI_API_KEY")
    print("  - DEEPSEEK_API_KEY")
    print("  - KIMI_API_KEY")
    print("  - QWEN_API_KEY")
    print("  - OPENROUTER_API_KEY")
    print("\nOpen your browser to: http://localhost:8000")
    print("="*60 + "\n")

    port = int(os.getenv('PORT', 8000))
    app.run(debug=True, port=port, host='0.0.0.0')
