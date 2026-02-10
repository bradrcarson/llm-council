"""
LLM Council - Orchestrates queries across Claude, Gemini, GPT, and Grok models
"""
import os
import asyncio
from typing import Dict, List
from anthropic import Anthropic
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMCouncil:
    """Manages the council of LLMs and orchestrates their interactions"""

    def __init__(self):
        # Initialize API clients
        self.anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # Llama via OpenRouter (OpenAI-compatible)
        self.llama_client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # Grok uses OpenAI-compatible API
        self.grok_client = OpenAI(
            api_key=os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1"
        )
        # DeepSeek uses OpenAI-compatible API
        self.deepseek_client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        # Kimi via OpenRouter (OpenAI-compatible)
        self.kimi_client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        # Mistral via OpenRouter (OpenAI-compatible)
        self.mistral_client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        # Qwen via OpenRouter (OpenAI-compatible)
        self.qwen_client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )

    def query_claude(self, prompt: str, conversation_history: List[Dict] = None) -> str:
        """Query Claude API with optional conversation history"""
        try:
            messages = []
            if conversation_history:
                messages.extend(conversation_history)
            messages.append({"role": "user", "content": prompt})

            message = self.anthropic_client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=4096,
                messages=messages
            )
            return message.content[0].text
        except Exception as e:
            return f"Error querying Claude: {str(e)}"

    def query_llama(self, prompt: str, conversation_history: List[Dict] = None) -> str:
        """Query Llama API via OpenRouter with optional conversation history"""
        try:
            messages = []
            if conversation_history:
                messages.extend(conversation_history)
            messages.append({"role": "user", "content": prompt})

            response = self.llama_client.chat.completions.create(
                model="meta-llama/llama-3.1-70b-instruct",
                messages=messages,
                max_tokens=4096
            )
            return response.choices[0].message.content or "(No response from Llama)"
        except Exception as e:
            return f"Error querying Llama: {str(e)}"

    def query_gpt(self, prompt: str, conversation_history: List[Dict] = None) -> str:
        """Query GPT API with optional conversation history"""
        try:
            messages = []
            if conversation_history:
                messages.extend(conversation_history)
            messages.append({"role": "user", "content": prompt})

            response = self.openai_client.chat.completions.create(
                model="gpt-5.2",
                messages=messages,
                max_completion_tokens=4096
            )
            return response.choices[0].message.content or "(No response from GPT)"
        except Exception as e:
            return f"Error querying GPT: {str(e)}"

    def query_grok(self, prompt: str, conversation_history: List[Dict] = None) -> str:
        """Query Grok API with optional conversation history"""
        try:
            messages = []
            if conversation_history:
                messages.extend(conversation_history)
            messages.append({"role": "user", "content": prompt})

            response = self.grok_client.chat.completions.create(
                model="grok-4-1-fast-reasoning",
                messages=messages,
                max_tokens=4096
            )
            return response.choices[0].message.content or "(No response from Grok)"
        except Exception as e:
            return f"Error querying Grok: {str(e)}"

    def query_deepseek(self, prompt: str, conversation_history: List[Dict] = None) -> str:
        """Query DeepSeek API with optional conversation history"""
        try:
            messages = []
            if conversation_history:
                messages.extend(conversation_history)
            messages.append({"role": "user", "content": prompt})

            response = self.deepseek_client.chat.completions.create(
                model="deepseek-reasoner",
                messages=messages,
                max_tokens=4096
            )
            return response.choices[0].message.content or "(No response from DeepSeek)"
        except Exception as e:
            return f"Error querying DeepSeek: {str(e)}"

    def query_kimi(self, prompt: str, conversation_history: List[Dict] = None) -> str:
        """Query Kimi API via OpenRouter with optional conversation history"""
        try:
            messages = []
            if conversation_history:
                messages.extend(conversation_history)
            messages.append({"role": "user", "content": prompt})

            response = self.kimi_client.chat.completions.create(
                model="moonshotai/kimi-k2",
                messages=messages,
                max_tokens=4096
            )
            return response.choices[0].message.content or "(No response from Kimi)"
        except Exception as e:
            return f"Error querying Kimi: {str(e)}"

    def query_mistral(self, prompt: str, conversation_history: List[Dict] = None) -> str:
        """Query Mistral API via OpenRouter with optional conversation history"""
        try:
            messages = []
            if conversation_history:
                messages.extend(conversation_history)
            messages.append({"role": "user", "content": prompt})

            response = self.mistral_client.chat.completions.create(
                model="mistralai/mistral-large-2512",
                messages=messages,
                max_tokens=4096
            )
            return response.choices[0].message.content or "(No response from Mistral)"
        except Exception as e:
            return f"Error querying Mistral: {str(e)}"

    def query_qwen(self, prompt: str, conversation_history: List[Dict] = None) -> str:
        """Query Qwen API via OpenRouter with optional conversation history"""
        try:
            messages = []
            if conversation_history:
                messages.extend(conversation_history)
            messages.append({"role": "user", "content": prompt})

            response = self.qwen_client.chat.completions.create(
                model="qwen/qwen-2.5-72b-instruct",
                messages=messages,
                max_tokens=4096
            )
            if response and response.choices and len(response.choices) > 0:
                msg = response.choices[0].message
                if msg and msg.content:
                    return msg.content
            return "(No response from Qwen)"
        except Exception as e:
            return f"Error querying Qwen: {str(e)}"

    async def query_all_async(self, prompt: str, conversation_history: List[Dict] = None) -> Dict[str, str]:
        """Query all LLMs in parallel with optional conversation history"""
        loop = asyncio.get_event_loop()

        # Run all queries concurrently
        tasks = [
            loop.run_in_executor(None, self.query_claude, prompt, conversation_history),
            loop.run_in_executor(None, self.query_llama, prompt, conversation_history),
            loop.run_in_executor(None, self.query_gpt, prompt, conversation_history),
            loop.run_in_executor(None, self.query_grok, prompt, conversation_history),
            loop.run_in_executor(None, self.query_deepseek, prompt, conversation_history),
            loop.run_in_executor(None, self.query_kimi, prompt, conversation_history),
            loop.run_in_executor(None, self.query_mistral, prompt, conversation_history),
            loop.run_in_executor(None, self.query_qwen, prompt, conversation_history)
        ]

        results = await asyncio.gather(*tasks)

        return {
            "claude": results[0],
            "llama": results[1],
            "gpt": results[2],
            "grok": results[3],
            "deepseek": results[4],
            "kimi": results[5],
            "mistral": results[6],
            "qwen": results[7]
        }

    def analyze_responses(self, responses: Dict[str, str]) -> str:
        """Use Claude to analyze agreements and disagreements"""
        analysis_prompt = f"""You are analyzing responses from eight different AI models (Claude, Llama, GPT, Grok, DeepSeek, Kimi, Mistral, and Qwen) to the same question.

CLAUDE's response:
{responses['claude']}

LLAMA's response:
{responses['llama']}

GPT's response:
{responses['gpt']}

GROK's response:
{responses['grok']}

DEEPSEEK's response:
{responses['deepseek']}

KIMI's response:
{responses['kimi']}

MISTRAL's response:
{responses['mistral']}

QWEN's response:
{responses['qwen']}

Please provide a detailed analysis covering:
1. Key agreements - What do all eight models agree on?
2. Key disagreements - Where do they differ?
3. Unique perspectives - What unique insights does each model provide?
4. Overall assessment - Which response seems most comprehensive/accurate?

Be objective and thorough in your analysis."""

        return self.query_claude(analysis_prompt)

    def generate_cross_commentary(self, responses: Dict[str, str]) -> Dict[str, str]:
        """Generate prompts for each AI to comment on the others"""
        commentaries = {}

        # Build responses text for each model to review (excluding itself)
        def get_others_responses(exclude: str) -> str:
            models = ['claude', 'llama', 'gpt', 'grok', 'deepseek', 'kimi', 'mistral', 'qwen']
            others = [m for m in models if m != exclude]
            return '\n\n'.join([f"{m.upper()}'s response:\n{responses[m]}" for m in others])

        # Claude comments on the others
        claude_prompt = f"""You are Claude. Review these responses from the other AI models to the same question:

{get_others_responses('claude')}

Please provide your thoughts on their responses. What do they get right? Where might they be mistaken or incomplete? What would you add or clarify?"""
        commentaries['claude_commentary'] = self.query_claude(claude_prompt)

        # Llama comments on the others
        llama_prompt = f"""You are Llama. Review these responses from the other AI models to the same question:

{get_others_responses('llama')}

Please provide your thoughts on their responses. What do they get right? Where might they be mistaken or incomplete? What would you add or clarify?"""
        commentaries['llama_commentary'] = self.query_llama(llama_prompt)

        # GPT comments on the others
        gpt_prompt = f"""You are GPT. Review these responses from the other AI models to the same question:

{get_others_responses('gpt')}

Please provide your thoughts on their responses. What do they get right? Where might they be mistaken or incomplete? What would you add or clarify?"""
        commentaries['gpt_commentary'] = self.query_gpt(gpt_prompt)

        # Grok comments on the others
        grok_prompt = f"""You are Grok. Review these responses from the other AI models to the same question:

{get_others_responses('grok')}

Please provide your thoughts on their responses. What do they get right? Where might they be mistaken or incomplete? What would you add or clarify?"""
        commentaries['grok_commentary'] = self.query_grok(grok_prompt)

        # DeepSeek comments on the others
        deepseek_prompt = f"""You are DeepSeek. Review these responses from the other AI models to the same question:

{get_others_responses('deepseek')}

Please provide your thoughts on their responses. What do they get right? Where might they be mistaken or incomplete? What would you add or clarify?"""
        commentaries['deepseek_commentary'] = self.query_deepseek(deepseek_prompt)

        # Kimi comments on the others
        kimi_prompt = f"""You are Kimi. Review these responses from the other AI models to the same question:

{get_others_responses('kimi')}

Please provide your thoughts on their responses. What do they get right? Where might they be mistaken or incomplete? What would you add or clarify?"""
        commentaries['kimi_commentary'] = self.query_kimi(kimi_prompt)

        # Mistral comments on the others
        mistral_prompt = f"""You are Mistral. Review these responses from the other AI models to the same question:

{get_others_responses('mistral')}

Please provide your thoughts on their responses. What do they get right? Where might they be mistaken or incomplete? What would you add or clarify?"""
        commentaries['mistral_commentary'] = self.query_mistral(mistral_prompt)

        # Qwen comments on the others
        qwen_prompt = f"""You are Qwen. Review these responses from the other AI models to the same question:

{get_others_responses('qwen')}

Please provide your thoughts on their responses. What do they get right? Where might they be mistaken or incomplete? What would you add or clarify?"""
        commentaries['qwen_commentary'] = self.query_qwen(qwen_prompt)

        return commentaries

    async def full_council_session(self, prompt: str, include_analysis: bool = True, include_commentary: bool = True, conversation_history: List[Dict] = None) -> Dict:
        """Run a complete council session with optional analysis, cross-commentary, and conversation history"""
        # Step 1: Get initial responses from all models
        print("Querying all models...")
        responses = await self.query_all_async(prompt, conversation_history)

        result = {
            "original_prompt": prompt,
            "responses": responses
        }

        # Step 2: Analyze agreements and disagreements (optional)
        if include_analysis:
            print("Analyzing responses...")
            result["analysis"] = self.analyze_responses(responses)

        # Step 3: Generate cross-commentary (optional)
        if include_commentary:
            print("Generating cross-commentary...")
            result["commentaries"] = self.generate_cross_commentary(responses)

        return result


if __name__ == "__main__":
    # Test the council
    async def test():
        council = LLMCouncil()
        result = await council.full_council_session(
            "What are the most important considerations when building AI systems?"
        )

        print("\n=== ORIGINAL RESPONSES ===")
        for model, response in result['responses'].items():
            print(f"\n{model.upper()}:\n{response}\n")

        print("\n=== ANALYSIS ===")
        print(result['analysis'])

        print("\n=== CROSS-COMMENTARY ===")
        for commentary_type, commentary in result['commentaries'].items():
            print(f"\n{commentary_type.upper()}:\n{commentary}\n")

    asyncio.run(test())
