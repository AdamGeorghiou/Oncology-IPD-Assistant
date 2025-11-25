# src/core/llm_bridge.py
import os
import google.generativeai as genai
import streamlit as st
from google.api_core import exceptions

# Try to get key from OS env or Streamlit secrets
API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

def get_optimal_model():
    """
    Queries Google API for available models and selects the best one 
    for the Free Tier (prioritizing 'flash').
    """
    try:
        available = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available.append(m.name)
        
        # Priority 1: Look for any "flash" model (Best for free tier limits)
        # e.g., models/gemini-1.5-flash-latest or models/gemini-1.5-flash-001
        flash_models = [m for m in available if 'flash' in m]
        if flash_models:
            # Pick the first one found
            return flash_models[0]
            
        # Priority 2: Look for 1.5 Pro (Standard)
        pro_models = [m for m in available if '1.5-pro' in m]
        if pro_models:
            return pro_models[0]
            
        # Fallback: Take the first available generative model
        if available:
            return available[0]
            
        return None
        
    except Exception as e:
        print(f"Discovery Error: {e}")
        return None

def ask_gemini(context_text: str, user_question: str) -> str:
    """
    Sends trial data + user question to Google Gemini.
    """
    if not API_KEY:
        return "⚠️ Error: GOOGLE_API_KEY not found. Please set it in your environment variables."

    genai.configure(api_key=API_KEY)

    # 1. Dynamically find the correct model ID
    model_name = get_optimal_model()
    
    if not model_name:
        return "⚠️ API Error: Could not list any available Gemini models for this API Key."

    try:
        # 2. Initialize
        model = genai.GenerativeModel(model_name)
        
        full_prompt = (
            "You are an expert Oncology Research Assistant. "
            "You are analyzing clinical trial data reconstructed from Kaplan-Meier curves.\n\n"
            "--- CONTEXT (TRIAL DATA) ---\n"
            f"{context_text}\n"
            "----------------------------\n\n"
            "INSTRUCTIONS:\n"
            "1. Answer the user's question based ONLY on the context provided above.\n"
            "2. Be precise with numbers (Hazard Ratios, P-values, Medians).\n"
            "3. If the data is missing, say so.\n"
            "4. Keep answers professional, concise, and clinical.\n\n"
            f"USER QUESTION: {user_question}"
        )

        # 3. Generate
        # We catch quota errors specifically to give a helpful message
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.2)
        )
        return response.text

    except exceptions.ResourceExhausted:
        return f"⚠️ Quota Limit Reached on {model_name}. Please wait a minute and try again."
        
    except Exception as e:
        return f"⚠️ API Error ({model_name}): {str(e)}"