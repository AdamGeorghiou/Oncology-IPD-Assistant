# pages/2_🤖_LLM_Analysis.py
"""
Multi-Trial LLM Analysis Page

Compare multiple trials from the Knowledge Base using AI.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from src.core.project_io import load_project_names, load_project_data

# Page Config
st.set_page_config(page_title="Multi-Trial Analysis", page_icon="🤖", layout="wide")

st.title("🤖 Multi-Trial Analysis")
st.caption("Compare trials from your Knowledge Base using AI-powered analysis")

# 1. LOAD LIBRARY
saved_trials = load_project_names()

if not saved_trials:
    st.warning("Knowledge Base is empty. Save trials from the main app first.")
    st.stop()

# Build overview table
table_data = []
trial_lookup = {}  # Map display name to file ID

for t_id in saved_trials:
    d = load_project_data(t_id)
    if d and 'meta' in d:
        m = d['meta']
        display_name = m.get('name', t_id)
        trial_lookup[display_name] = t_id
        table_data.append({
            "Trial": display_name,
            "Disease": m.get('disease', ''),
            "Phase": m.get('phase', ''),
            "Endpoint": m.get('endpoint', ''),
            "HR": m.get('hr'),
            "p-value": m.get('p_value'),
            "Arms": len(m.get('medians', {}))
        })

df_overview = pd.DataFrame(table_data)

# 2. TRIAL SELECTION
st.markdown("##### Select Trials to Compare")

col_table, col_select = st.columns([3, 1])

with col_table:
    st.dataframe(
        df_overview,
        use_container_width=True,
        column_config={
            "HR": st.column_config.NumberColumn(format="%.2f"),
            "p-value": st.column_config.NumberColumn(format="%.4f"),
            "Arms": st.column_config.NumberColumn(format="%d")
        },
        hide_index=True
    )

with col_select:
    selected_names = st.multiselect(
        "Choose trials:",
        options=df_overview['Trial'].tolist(),
        default=[],
        label_visibility="collapsed"
    )
    
    if len(selected_names) == 0:
        st.info("Select 1+ trials")
    elif len(selected_names) == 1:
        st.success("1 trial selected")
    else:
        st.success(f"{len(selected_names)} trials selected")

# Reset chat if selection changes
if 'last_selected_trials' not in st.session_state:
    st.session_state.last_selected_trials = []

if set(selected_names) != set(st.session_state.last_selected_trials):
    st.session_state.llm_chat_history = []
    st.session_state.last_selected_trials = list(selected_names)

# 3. ANALYSIS DASHBOARD
if selected_names:
    st.divider()
    
    # Collect data and build context
    llm_context_lines = []
    plot_data = []
    
    for trial_name in selected_names:
        t_id = trial_lookup.get(trial_name)
        if not t_id:
            continue
            
        data = load_project_data(t_id)
        if not data:
            continue
        
        meta = data['meta']
        results = data.get('reconstruction_results', {})
        
        # Collect plot data
        for arm_name, res in results.items():
            plot_data.append({
                'trial': meta.get('name', trial_name),
                'arm': arm_name,
                'time': res['recon_curve']['time'],
                'survival': res['recon_curve']['survival'],
                'color': res.get('color')
            })
        
        # Build context string for this trial
        medians = meta.get('medians', {})
        medians_str = ", ".join([f"{k}: {v:.1f}m" for k, v in medians.items()]) if medians else "N/A"
        
        n_per_arm = meta.get('n_per_arm', {})
        n_str = ", ".join([f"{k} (n={v})" for k, v in n_per_arm.items()]) if n_per_arm else "N/A"
        
        context_line = (
            f"TRIAL: {meta.get('name', 'Unknown')}\n"
            f"  Disease: {meta.get('disease', 'Unknown')}\n"
            f"  Setting: {meta.get('setting', 'N/A')}\n"
            f"  Phase: {meta.get('phase', 'N/A')}\n"
            f"  Endpoint: {meta.get('endpoint', 'N/A')}\n"
            f"  Comparison: {meta.get('comparison', 'N/A')}\n"
            f"  Hazard Ratio: {meta.get('hr', 0):.2f} (95% CI: {meta.get('ci_lower', 0):.2f}–{meta.get('ci_upper', 0):.2f})\n"
            f"  P-value: {meta.get('p_value', 0):.4f}\n"
            f"  Median Survival: {medians_str}\n"
            f"  Population: {n_str}\n"
        )
        llm_context_lines.append(context_line)
    
    # Layout: Plot + Stats | Chat
    col_viz, col_chat = st.columns([1, 1])
    
    # --- LEFT: Visualization ---
    with col_viz:
        st.markdown("##### 📈 Survival Curves")
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        for item in plot_data:
            label = f"{item['trial']} - {item['arm']}"
            ax.plot(
                item['time'],
                item['survival'],
                linewidth=2,
                label=label,
                color=item['color']
            )
        
        ax.set_ylim(0, 105)
        ax.set_xlabel("Time (months)")
        ax.set_ylabel("Survival (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left', fontsize='small')
        st.pyplot(fig)
        plt.close(fig)
        
        # Quick stats table
        st.markdown("##### 📊 Summary")
        
        summary_data = []
        for trial_name in selected_names:
            t_id = trial_lookup.get(trial_name)
            if t_id:
                data = load_project_data(t_id)
                if data and 'meta' in data:
                    m = data['meta']
                    summary_data.append({
                        "Trial": m.get('name', ''),
                        "HR": f"{m.get('hr', 0):.2f}",
                        "95% CI": f"{m.get('ci_lower', 0):.2f}–{m.get('ci_upper', 0):.2f}",
                        "p-value": f"{m.get('p_value', 0):.4f}",
                    })
        
        if summary_data:
            st.dataframe(
                pd.DataFrame(summary_data),
                use_container_width=True,
                hide_index=True
            )
        
        # Context preview
        with st.expander("🔍 View LLM Context"):
            st.text("\n---\n".join(llm_context_lines))
    
    # --- RIGHT: Chat ---
    with col_chat:
        st.markdown("##### 💬 AI Analysis")
        
        # Initialize chat
        if not st.session_state.llm_chat_history:
            if len(selected_names) == 1:
                intro = f"I've loaded **{selected_names[0]}**. What would you like to know?"
            else:
                intro = f"I've loaded **{len(selected_names)} trials**. Ask me to compare them or explain the findings."
            st.session_state.llm_chat_history = [
                {"role": "assistant", "content": intro}
            ]
        
        # Chat container
        chat_container = st.container(height=400)
        with chat_container:
            for msg in st.session_state.llm_chat_history:
                st.chat_message(msg["role"]).write(msg["content"])
        
        # Input
        if prompt := st.chat_input("Ask about these trials..."):
            st.session_state.llm_chat_history.append({"role": "user", "content": prompt})
            
            # Call LLM
            from src.core.llm_bridge import ask_gemini_comparison
            
            with st.spinner("Analyzing..."):
                ai_response = ask_gemini_comparison(llm_context_lines, prompt)
            
            st.session_state.llm_chat_history.append({"role": "assistant", "content": ai_response})
            st.rerun()
        
        # Quick prompts
        st.caption("Try asking:")
        col_q1, col_q2 = st.columns(2)
        with col_q1:
            if st.button("Compare HRs", use_container_width=True, type="secondary"):
                st.session_state.llm_chat_history.append({"role": "user", "content": "Compare the hazard ratios across these trials"})
                from src.core.llm_bridge import ask_gemini_comparison
                with st.spinner("Analyzing..."):
                    ai_response = ask_gemini_comparison(llm_context_lines, "Compare the hazard ratios across these trials")
                st.session_state.llm_chat_history.append({"role": "assistant", "content": ai_response})
                st.rerun()
        with col_q2:
            if st.button("Summarize all", use_container_width=True, type="secondary"):
                st.session_state.llm_chat_history.append({"role": "user", "content": "Summarize the key findings from all trials"})
                from src.core.llm_bridge import ask_gemini_comparison
                with st.spinner("Analyzing..."):
                    ai_response = ask_gemini_comparison(llm_context_lines, "Summarize the key findings from all trials")
                st.session_state.llm_chat_history.append({"role": "assistant", "content": ai_response})
                st.rerun()

else:
    st.info("👆 Select trials from the table above to begin analysis.")