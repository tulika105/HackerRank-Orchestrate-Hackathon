import os
import json
from groq import Groq
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

class SupportAgent:
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = model_name

    def process_ticket(self, issue: str, subject: str, company: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Processes a single ticket using retrieved context and Groq LLM.
        """
        context_str = "\n---\n".join([f"Source: {c['metadata']['filename']}\nContent: {c['content']}" for c in context])
        
        system_prompt = f"""
You are a high-performance support triage agent for HackerRank, Claude, and Visa.
Your goal is to accurately classify tickets, decide whether to reply or escalate, and provide grounded responses.

RULES:
1. Use ONLY the provided context to answer. 
2. If the answer is NOT found in the provided context, you MUST state that the request is out of scope. Do NOT use your own knowledge to answer questions about actors, movies, or general knowledge not in the corpus.
3. Status MUST be 'replied' or 'escalated'.
4. Request Type MUST be 'product_issue', 'feature_request', 'bug', or 'invalid'.
5. Escalate if:
   - High risk (security, fraud, legal, stolen items)
   - Site/service is confirmed down
   - Sensitive account access issues not covered by FAQ
   - Explicit request for human intervention
6. Response should be polite, professional, and grounded.
7. Justification should explain why you decided to reply or escalate, and how you used the context.

OUTPUT FORMAT:
Return a JSON object with exactly these keys:
- status: "replied" | "escalated"
- product_area: string (best fit category)
- response: string (user-facing answer or escalation notice)
- justification: string (internal reasoning)
- request_type: "product_issue" | "feature_request" | "bug" | "invalid"
"""

        user_prompt = f"""
COMPANY: {company}
SUBJECT: {subject}
ISSUE: {issue}

RELEVANT SUPPORT DOCUMENTATION:
{context_str}

Please analyze this ticket and provide the required JSON output.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            print(f"Error processing ticket: {e}")
            return {
                "status": "escalated",
                "product_area": "unknown",
                "response": "I am having trouble processing your request. Escalating to a human.",
                "justification": f"Error: {str(e)}",
                "request_type": "product_issue"
            }

if __name__ == "__main__":
    # Test with mock data
    agent = SupportAgent()
    res = agent.process_ticket("How to delete my account?", "", "HackerRank", [{"content": "To delete account, go to settings...", "metadata": {"filename": "delete.md"}}])
    print(json.dumps(res, indent=2))
