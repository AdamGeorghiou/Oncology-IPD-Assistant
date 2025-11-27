# src/core/llm_bridge.py
"""
LLM Bridge for Clinical Trial Analysis

Connects to Google Gemini for AI-powered trial interpretation.
"""

import os
import google.generativeai as genai
import streamlit as st
from google.api_core import exceptions

# Try to get key from OS env or Streamlit secrets
API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")


def get_optimal_model():
    """
    Queries Google API for available models and selects the best one.
    Prioritizes 1.5-flash for stability.
    """
    try:
        available = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available.append(m.name)
        
        # Priority 1: Gemini 1.5 Flash (most reliable)
        flash_15 = [m for m in available if '1.5-flash' in m]
        if flash_15:
            return flash_15[0]
        
        # Priority 2: Any flash model except 2.5 (stricter safety)
        flash_models = [m for m in available if 'flash' in m and '2.5' not in m]
        if flash_models:
            return flash_models[0]
        
        # Priority 3: 1.5 Pro
        pro_models = [m for m in available if '1.5-pro' in m]
        if pro_models:
            return pro_models[0]
        
        # Fallback
        if available:
            return available[0]
        
        return "models/gemini-1.5-flash"
    except Exception as e:
        print(f"Discovery Error: {e}")
        return "models/gemini-1.5-flash"


SYSTEM_PROMPT = """You are an expert Oncology Biostatistician and Clinical Research Scientist.

You are analyzing clinical trial data that has been reconstructed from published Kaplan-Meier survival curves using the Guyot algorithm. This data represents Individual Patient Data (IPD) extracted from trial publications.

YOUR EXPERTISE:
- Interpreting Hazard Ratios and their clinical significance
- Understanding survival analysis (Kaplan-Meier, Cox proportional hazards)
- Contextualizing results within oncology treatment landscapes
- Explaining statistical concepts to varied audiences

=== CRITICAL: HAZARD RATIO INTERPRETATION ===

The HR in this system is calculated using Cox regression with the formula:
    HR = hazard(Arm 2) / hazard(Arm 1)
    
Where Arm 1 is the REFERENCE (Group 0) and Arm 2 is the COMPARATOR (Group 1).

INTERPRETATION RULES:
- HR > 1 → Arm 2 has HIGHER hazard → Arm 1 (reference) has BETTER survival
- HR < 1 → Arm 2 has LOWER hazard → Arm 2 (comparator) has BETTER survival  
- HR = 1 → No difference between arms

ALWAYS VERIFY by checking median survival times:
- The arm with LONGER median survival should be the one with BETTER survival
- If your HR interpretation contradicts the medians, you have the direction wrong

EXAMPLE:
- If HR = 1.34 and Arm 1 median = 10 months, Arm 2 median = 5 months
- HR > 1 means Arm 1 is better (lower hazard)
- Medians confirm: Arm 1 (10 mo) > Arm 2 (5 mo) ✓ CONSISTENT

=== END HR GUIDANCE ===

GUIDELINES:
1. Be precise with statistics - quote the exact numbers from the context
2. Interpret clinical significance, not just statistical significance
3. ALWAYS cross-check HR interpretation against median survival
4. Consider p-values: < 0.05 is typically significant
5. Mention limitations when relevant (reconstructed data, single trial, etc.)
6. Keep responses concise but complete
7. If information is not in the context, say so clearly

RESPONSE STYLE:
- Professional and clinical
- Use bullet points for clarity when listing multiple points
- Include specific numbers when discussing results
- State clearly which arm shows benefit
"""


def _safe_get_response_text(response):
    """
    Safely extract text from Gemini response, handling blocked/empty responses.
    """
    try:
        if not response.candidates:
            return None, "No response candidates returned"
        
        candidate = response.candidates[0]
        finish_reason = candidate.finish_reason
        
        # 3 = SAFETY, 4 = RECITATION
        if finish_reason == 3:
            return None, "Response blocked by safety filters. Try rephrasing your question."
        if finish_reason == 4:
            return None, "Response blocked due to recitation concerns."
        
        if candidate.content and candidate.content.parts:
            text = candidate.content.parts[0].text
            return text, None
        else:
            return None, f"Empty response (finish_reason: {finish_reason})"
            
    except Exception as e:
        return None, f"Error parsing response: {str(e)}"


def ask_gemini(context_text: str, user_question: str) -> str:
    """
    Sends trial data + user question to Google Gemini.
    """
    if not API_KEY:
        return "⚠️ Error: GOOGLE_API_KEY not found. Please set it in your environment variables."
    
    genai.configure(api_key=API_KEY)
    
    model_name = get_optimal_model()
    if not model_name:
        return "⚠️ API Error: Could not find any available Gemini models."
    
    try:
        model = genai.GenerativeModel(
            model_name,
            safety_settings={
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
            }
        )
        
        full_prompt = f"""{SYSTEM_PROMPT}

===== TRIAL DATA =====
{context_text}
======================

USER QUESTION: {user_question}

Remember: Always verify your HR interpretation matches the median survival data. State clearly which treatment arm shows benefit.

Provide a clear, evidence-based response:"""

        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=1024
            )
        )
        
        text, error = _safe_get_response_text(response)
        
        if text:
            return text
        else:
            return f"⚠️ {error}"
    
    except exceptions.ResourceExhausted:
        return f"⚠️ Quota limit reached. Please wait a minute and try again."
    except Exception as e:
        return f"⚠️ API Error ({model_name}): {str(e)}"


def ask_gemini_comparison(trials_context: list, user_question: str) -> str:
    """
    For comparing multiple trials (Knowledge Base feature).
    """
    if not API_KEY:
        return "⚠️ Error: GOOGLE_API_KEY not found."
    
    genai.configure(api_key=API_KEY)
    
    model_name = get_optimal_model()
    if not model_name:
        return "⚠️ API Error: No available models."
    
    try:
        model = genai.GenerativeModel(
            model_name,
            safety_settings={
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
            }
        )
        
        trials_text = "\n\n---\n\n".join(trials_context)
        
        full_prompt = f"""{SYSTEM_PROMPT}

You are comparing multiple clinical trials. Analyze similarities, differences, and draw insights across trials.

===== TRIALS DATA =====
{trials_text}
=======================

USER QUESTION: {user_question}

Remember: For each trial, verify HR interpretation matches median survival. State clearly which arm benefits in each trial.

Provide a comparative analysis:"""

        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1500
            )
        )
        
        text, error = _safe_get_response_text(response)
        
        if text:
            return text
        else:
            return f"⚠️ {error}"
    
    except exceptions.ResourceExhausted:
        return f"⚠️ Quota limit reached. Please wait and try again."
    except Exception as e:
        return f"⚠️ API Error: {str(e)}"