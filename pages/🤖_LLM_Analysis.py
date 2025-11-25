import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from src.core.project_io import load_project_names, load_project_data

# Page Config
st.set_page_config(page_title="LLM Trial Analysis", page_icon="🤖", layout="wide")
st.title("🤖 Multi-Trial LLM Analysis")

st.markdown("""
Select saved trials from the Knowledge Base. The system will extract clinical metadata and 
survival statistics to generate a **context-aware prompt** for the AI Agent.
""")

# 1. LOAD LIBRARY
saved_trials = load_project_names()

if not saved_trials:
    st.warning("Knowledge Base is empty. Go to the main app to digitize and save trials.")
    st.stop()

# Helper: Load meta for table
table_data = []
for t_id in saved_trials:
    d = load_project_data(t_id)
    if d and 'meta' in d:
        m = d['meta']
        table_data.append({
            "Trial": m.get('name'),
            "Disease": m.get('disease'),
            "Setting": m.get('setting'),
            "Phase": m.get('phase'),
            "Endpoint": m.get('endpoint'),
            "HR": m.get('hr'),
            "P-Value": m.get('p_value')
        })

# 2. TRIAL SELECTION
df_overview = pd.DataFrame(table_data)
st.dataframe(
    df_overview, 
    use_container_width=True, 
    column_config={
        "HR": st.column_config.NumberColumn(format="%.2f"),
        "P-Value": st.column_config.NumberColumn(format="%.4f")
    },
    hide_index=True
)

selected_names = st.multiselect("Select trials to analyze:", options=df_overview['Trial'].tolist())

if selected_names:
    st.divider()
    
    # 3. CONTEXT GENERATION & PLOTTING (Do this FIRST so context exists)
    col_viz, col_context = st.columns([1, 1])
    
    llm_context_lines = []
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for trial_name in selected_names:
        # Find filename via simple search
        clean_name = [t for t in saved_trials if trial_name in t][0] 
        data = load_project_data(clean_name)
        
        if not data: continue
        
        meta = data['meta']
        results = data['reconstruction_results']
        
        # A. Plotting
        for arm_name, res in results.items():
            color = res.get('color', None)
            label = f"{meta['name']} - {arm_name}"
            ax.plot(res['recon_curve']['time'], res['recon_curve']['survival'], 
                   linewidth=2, label=label, color=color)
        
        # B. Text Context Construction
        medians_str = ", ".join([f"{k}: {v:.1f}m" for k,v in meta.get('medians', {}).items()])
        n_counts = meta.get('n_per_arm', {})
        n_str = ", ".join([f"{k} (n={v})" for k,v in n_counts.items()])
        
        line = (
            f"TRIAL: {meta.get('name')}\n"
            f" - Meta: {meta.get('disease')}, {meta.get('setting', 'N/A')}, {meta.get('phase', 'N/A')}, {meta.get('endpoint')}\n"
            f" - Comparison: {meta.get('comparison')}\n"
            f" - Stats: HR {meta.get('hr', 0):.2f} (95% CI {meta.get('ci_lower',0):.2f}-{meta.get('ci_upper',0):.2f}), p={meta.get('p_value',0):.4f}\n"
            f" - Survival: Medians [{medians_str}]\n"
            f" - Population: {n_str}\n"
        )
        llm_context_lines.append(line)

    # Finalize Plot
    with col_viz:
        st.subheader("📈 Survival Overlay")
        ax.set_ylim(0, 105)
        ax.set_xlabel("Time (Months)")
        ax.set_ylabel("Survival (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize='small')
        st.pyplot(fig)

    # Finalize Context String
    full_prompt_context = "You are an expert Oncology Assistant. Here is the data for the selected trials:\n\n" + "\n".join(llm_context_lines)
    
    with col_context:
        st.subheader("🧠 LLM Context Preview")
        with st.expander("View generated prompt context", expanded=False):
            st.text(full_prompt_context)
        st.info("The Agent uses this structured text to answer your questions.")

    # 4. CHAT INTERFACE (Now safe to run)
    st.divider()
    st.subheader("💬 Comparative Agent")
    
    if "llm_chat_history" not in st.session_state:
        st.session_state.llm_chat_history = [{"role": "assistant", "content": "I have reviewed the selected trials. How can I help you compare them?"}]

    for msg in st.session_state.llm_chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask about these trials (e.g. 'Compare the hazard ratios')..."):
        # 1. Display User Message
        st.chat_message("user").write(prompt)
        st.session_state.llm_chat_history.append({"role": "user", "content": prompt})
        
        # 2. Call Real API
        from src.core.llm_bridge import ask_gemini
        
        with st.spinner("🤖 Comparing trials..."):
            # Now full_prompt_context is defined and populated!
            ai_response = ask_gemini(full_prompt_context, prompt)
        
        # 3. Display AI Response
        st.chat_message("assistant").write(ai_response)
        st.session_state.llm_chat_history.append({"role": "assistant", "content": ai_response})

else:
    st.info("👈 Select trials from the table above to generate the analysis.")