# app.py
import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.core.vision import detect_axes_and_crop
from src.core.extractor import auto_extract_curves
from PIL import Image
from src.core.project_io import save_project, load_project_names, load_project_data

# --- CONFIG ---
st.set_page_config(page_title="Oncology IPD Assistant", page_icon="🧬", layout="wide")

# INITIALIZE STATE VARIABLES
if 'step' not in st.session_state: st.session_state.step = 1
if 'extracted_arms' not in st.session_state: st.session_state.extracted_arms = []
if 'clinical_summary' not in st.session_state: st.session_state.clinical_summary = {} # <--- Added
if 'loaded_from_disk' not in st.session_state: st.session_state.loaded_from_disk = False # <--- Added
if 'reconstruction_results' not in st.session_state: st.session_state.reconstruction_results = {}
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = {}

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("🧬 IPD Assistant")

    # 1. Navigation for Current Session
    options = ["1. Ingest"]
    if st.session_state.get("cropped_img") is not None:
        options.append("2. Curves")
    if st.session_state.get("extracted_arms"):
        options.append("3. At-Risk")
    if st.session_state.get("risk_df") is not None:
        options.append("4. Validate")
    if st.session_state.get("reconstruction_results"):
        options.append("5. Analysis")

    

    step_map = {
        "1. Ingest": 1,
        "2. Curves": 2,
        "3. At-Risk": 3,
        "4. Validate": 4,
        "5. Analysis": 5,
    }

    
    # Determine current index safely
    current_step = st.session_state.get("step", 1)
    # If loaded from disk, force display of Step 5
    if st.session_state.get("loaded_from_disk"):
        current_step = 5
        
    # Find label for current step
    current_label = next((k for k, v in step_map.items() if v == current_step), "1. Ingest")
    if current_label not in options: current_label = options[-1]

    selected_option = st.radio("Current Workflow:", options, index=options.index(current_label))
    st.session_state.step = step_map[selected_option]

    st.markdown("---")
    
    # 2. KNOWLEDGE BASE (Load Logic)
    from src.core.project_io import load_project_names, load_project_data
    st.subheader("📚 Knowledge Base")
    
    saved_trials = load_project_names()
    selected_trial = st.selectbox("Load Trial", [""] + saved_trials)
    
    if st.button("📂 Open Trial"):
        if selected_trial:
            data = load_project_data(selected_trial)
            if data:
                # Hydrate Session State
                st.session_state.reconstruction_results = data['reconstruction_results']
                st.session_state.clinical_summary = data['meta'] # The Agent Context
                st.session_state.loaded_from_disk = True # Flag to prevent R re-runs
                st.session_state.step = 5 # Jump to end
                st.rerun()
    
    if st.button("🆕 Start New Extraction"):
        st.session_state.clear()
        st.rerun()



# --- STEP 1: INGEST ---
if st.session_state.step == 1:
    st.header("Step 1: Ingest Trial Data")
    uploaded_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        save_path = os.path.join("data/uploads", uploaded_file.name)
        os.makedirs("data/uploads", exist_ok=True)
        with open(save_path, "wb") as f: f.write(uploaded_file.getbuffer())
        
        st.session_state.current_image_path = save_path
        
        # Load image once
        original_img = cv2.imread(save_path)
        height, width = original_img.shape[:2]
        
        # Tabs for Auto vs Manual Crop
        tab_auto, tab_manual = st.tabs(["🤖 Auto-Detect Axes", "✂️ Manual Crop"])
        
        # --- AUTO DETECT ---
        with tab_auto:
            if st.button("Run Auto-Detection", type="primary"):
                with st.spinner("Analyzing geometry..."):
                    cropped_img, metadata = detect_axes_and_crop(save_path)
                    st.session_state.crop_metadata = metadata
                    st.session_state.cropped_img = cropped_img
            
            if 'crop_metadata' in st.session_state:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Detection")
                    if st.session_state.crop_metadata['status'] == 'success':
                        st.image(cv2.cvtColor(st.session_state.crop_metadata['debug_image'], cv2.COLOR_BGR2RGB), use_column_width=True)
                    else:
                        st.warning("Auto-detection failed.")
                        st.image(original_img, caption="Original", use_column_width=True)
                
                with col2:
                    st.subheader("Result")
                    if 'cropped_img' in st.session_state:
                        st.image(cv2.cvtColor(st.session_state.cropped_img, cv2.COLOR_BGR2RGB), use_column_width=True)
                        if st.button("Confirm Auto Crop ➡️"):
                            st.session_state.step = 2
                            st.rerun()

        # --- MANUAL CROP ---
        with tab_manual:
            st.info("Adjust sliders to define the graph area manually.")
            
            # Sliders for cropping (percent based)
            col_y1, col_y2 = st.columns(2)
            with col_y1: top_pct = st.slider("Top Edge (%)", 0, 100, 10)
            with col_y2: bottom_pct = st.slider("Bottom Edge (%)", 0, 100, 90)
            
            col_x1, col_x2 = st.columns(2)
            with col_x1: left_pct = st.slider("Left Edge (%)", 0, 100, 10)
            with col_x2: right_pct = st.slider("Right Edge (%)", 0, 100, 90)
            
            # Calculate pixel coords
            c_y1, c_y2 = int(height * top_pct / 100), int(height * bottom_pct / 100)
            c_x1, c_x2 = int(width * left_pct / 100), int(width * right_pct / 100)
            
            # Validation
            if c_y2 > c_y1 and c_x2 > c_x1:
                manual_crop = original_img[c_y1:c_y2, c_x1:c_x2]
                st.image(cv2.cvtColor(manual_crop, cv2.COLOR_BGR2RGB), caption="Manual Preview", width=500)
                
                if st.button("Use Manual Crop ➡️"):
                    st.session_state.cropped_img = manual_crop
                    # Create dummy metadata for compatibility
                    st.session_state.crop_metadata = {
                        "status": "manual",
                        "origin": (c_x1, c_y2), # Bottom-left of crop roughly corresponds to origin
                        "x_len_px": c_x2 - c_x1,
                        "y_len_px": c_y2 - c_y1
                    }
                    st.session_state.step = 2
                    st.rerun()
            else:
                st.error("Invalid crop dimensions. Top must be above Bottom, Left must be before Right.")

# --- STEP 2: CURVE EXTRACTION ---
elif st.session_state.step == 2:
    st.header("Step 2: Curve Extraction")
    
    col_cfg, col_view = st.columns([1, 2])
    with col_cfg:
        st.subheader("Axis Scaling")
        max_x = st.number_input("Max Time (X-Axis)", value=30.0, step=1.0, key="max_x")
        max_y = st.number_input("Max Survival (Y-Axis)", value=100.0, step=10.0, key="max_y")
    
    tab_auto, tab_seeded, tab_manual = st.tabs(["🤖 Auto", "🎯 Seeded", "✍️ Manual"])
    
    # Helper to add arm with metadata
    def add_arm(df, source, color_hex="#000000"):
        if 'extracted_arms' not in st.session_state: st.session_state.extracted_arms = []
        idx = len(st.session_state.extracted_arms) + 1
        
        # NEW DATA STRUCTURE
        arm_data = {
            "id": idx,
            "name": f"Arm {idx}",
            "data": df,          # The DataFrame
            "source": source,    # 'Auto', 'Seeded', 'Manual'
            "color": color_hex
        }
        st.session_state.extracted_arms.append(arm_data)
    
    # === AUTO ===
    with tab_auto:
        if st.button("Run Auto-Extraction"):
            with st.spinner("Processing..."):
                arms = auto_extract_curves(st.session_state.cropped_img, max_x, max_y)
                if arms:
                    for arm_df in arms:
                        add_arm(arm_df, "Auto", "#1f77b4") # Default blue
                    st.success(f"Found {len(arms)} arms!")
                    st.rerun()
                else:
                    st.warning("No curves found.")

    # === SEEDED ===
    with tab_seeded:
        st.info("Click a curve to extract it.")
        from streamlit_image_coordinates import streamlit_image_coordinates
        from src.core.seeded_extractor import get_hsv_at_point, extract_single_color_arm
        
        img_rgb = cv2.cvtColor(st.session_state.cropped_img, cv2.COLOR_BGR2RGB)
        coords = streamlit_image_coordinates(np.array(Image.fromarray(img_rgb)), key="seed")
        
        if coords:
            hsv = get_hsv_at_point(st.session_state.cropped_img, coords['x'], coords['y'])
            df = extract_single_color_arm(st.session_state.cropped_img, hsv, max_x, max_y)
            if df is not None:
                if st.button(f"Add Curve?"):
                    add_arm(df, "Seeded", "#ff7f0e") # Default orange
                    st.rerun()

    # === MANUAL ===
    with tab_manual:
        st.info("Draw the curve.")
        from streamlit_drawable_canvas import st_canvas
        from src.core.tracer import process_manual_drawing
        
        h, w = st.session_state.cropped_img.shape[:2]
        canvas = st_canvas(
            fill_color="rgba(0,0,0,0)", stroke_width=2, stroke_color="#FF0000",
            background_image=Image.fromarray(cv2.cvtColor(st.session_state.cropped_img, cv2.COLOR_BGR2RGB)),
            height=h, width=w, drawing_mode="freedraw", key="canvas"
        )
        if st.button("Process Drawing"):
            df = process_manual_drawing(canvas, w, h)
            if df is not None:
                # Convert to Data Units
                df['time'] = df['rel_time'] * max_x
                df['survival'] = df['rel_surv'] * max_y
                add_arm(df[['time', 'survival']], "Manual", "#d62728") # Default red
                st.rerun()

    # === MANAGEMENT & PREVIEW ===
    with col_view:
        st.subheader("Current Arms")
        if st.session_state.get('extracted_arms'):
            arms = st.session_state.extracted_arms
            
            # 1. Table View with Renaming
            for i, arm in enumerate(arms):
                c1, c2, c3 = st.columns([3, 2, 1])
                with c1: 
                    # RENAMING FEATURE:
                    # We use a text_input, defaulting to the current name.
                    # We update the dictionary immediately when changed.
                    new_name = st.text_input(
                        f"Name (Arm {i+1})", 
                        value=arm['name'], 
                        key=f"rename_{i}",
                        label_visibility="collapsed"
                    )
                    arm['name'] = new_name # Save back to state
                    
                with c2: 
                    new_color = st.color_picker(
                        "Color", 
                        arm['color'], 
                        key=f"c_{i}", 
                        label_visibility="collapsed"
                    )
                    arm['color'] = new_color
                    
                with c3: 
                    if st.button("🗑️", key=f"del_{i}"):
                        st.session_state.extracted_arms.pop(i)
                        st.rerun()
            
            # 2. Plot (Updates instantly when color/name changes)
            fig, ax = plt.subplots(figsize=(8, 4))
            for arm in arms:
                ax.plot(arm['data']['time'], arm['data']['survival'], 
                      linewidth=2, label=arm['name'], color=arm['color'])
            ax.set_xlim(0, max_x); ax.set_ylim(0, 105); ax.grid(True, alpha=0.3); ax.legend()
            st.pyplot(fig)
            
            if st.button("✅ Lock & Proceed to At-Risk", type="primary"):
                if len(arms) != 2:
                    st.warning("Analysis best supports exactly 2 arms.")
                st.session_state.step = 3
                st.rerun()

# --- STEP 3: AT-RISK TABLE ---
elif st.session_state.step == 3:
    st.header("Step 3: At-Risk Data")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.image(st.session_state.current_image_path, use_column_width=True)
        t_start = st.number_input("Start", 0.0); t_end = st.number_input("End", 30.0); t_step = st.number_input("Step", 3.0)
        
        if st.button("Initialize Grid"):
            times = [round(t, 1) for t in np.arange(t_start, t_end + 0.1, t_step) if t <= t_end]
            st.session_state.risk_times = times
            
            from src.core.table_manager import create_empty_risk_table
            # Use names from Step 2
            names = [a['name'] for a in st.session_state.extracted_arms]
            st.session_state.risk_df = create_empty_risk_table(names, times)
            st.rerun()
            
    with c2:
        if 'risk_df' in st.session_state:
            st.subheader("Data Entry")
            
            # OCR BUTTON
            if st.button("✨ Auto-Read Table (Best Effort)"):
                from src.core.ocr_table import auto_read_risk_table
                with st.spinner("Scanning bottom of image..."):
                    n_arms = len(st.session_state.extracted_arms)
                    n_cols = len(st.session_state.risk_times)
                    
                    ocr_data = auto_read_risk_table(st.session_state.current_image_path, n_arms, n_cols)
                    
                    if ocr_data:
                        # Update DataFrame
                        st.session_state.risk_df = pd.DataFrame(
                            ocr_data, 
                            index=st.session_state.risk_df.index, 
                            columns=st.session_state.risk_df.columns
                        )
                        st.success("OCR Complete! Please verify numbers.")
                        st.rerun()
                    else:
                        st.error("OCR failed to find a structured table. Please type manually.")

            # EDITOR
            edited = st.data_editor(st.session_state.risk_df, use_container_width=True, key="editor")
            
            if st.button("Validate & Finish ➡️", type="primary"):
                st.session_state.risk_df = edited
                # (Validation logic same as before...)
                st.session_state.step = 4
                st.rerun()

# --- STEP 4: VALIDATION ---
elif st.session_state.step == 4:
    st.header("Step 4: Validation & Quality Check")
    st.markdown("The system is now reconstructing individual patients (IPD) and comparing them to the digitised curves.")

    # 1. THE CONTROLLER
    col_run, col_status = st.columns([1, 3])
    
    with col_run:
        if st.button("🔄 Run Reconstruction Engine", type="primary"):
            from src.core.reconstructor import reconstruct_survival, calculate_km_from_ipd
            from sklearn.metrics import mean_absolute_error
            from scipy.interpolate import interp1d

            # Reset State
            st.session_state.reconstruction_results = {}
            st.session_state.validation_status = "Unknown"
            st.session_state.validation_color = "gray"
            st.session_state.reconstruction_has_run = True # NEW FLAG

            arms_meta = st.session_state.extracted_arms
            n_arms = len(arms_meta)
            
            if n_arms == 0:
                st.error("No arms found. Go back to Step 2.")
            else:
                progress_bar = st.progress(0.0)
                max_mae_across_arms = 0.0
                
                # Check for At-Risk Mismatch BEFORE starting
                if len(st.session_state.risk_df) < n_arms:
                    st.error(f"Mismatch: You have {n_arms} arms but only {len(st.session_state.risk_df)} rows in the At-Risk table. Go back to Step 3.")
                else:
                    for i, arm_meta in enumerate(arms_meta):
                        # Unpack metadata
                        arm_name = arm_meta["name"]
                        arm_df = arm_meta["data"]
                        arm_color = arm_meta.get("color", "blue")

                        try:
                            risk_counts = st.session_state.risk_df.iloc[i].values.astype(int)
                            risk_times = st.session_state.risk_times

                            # Reconstruct IPD + KM
                            ipd_df = reconstruct_survival(arm_df, risk_times, risk_counts)
                            recon_km = calculate_km_from_ipd(ipd_df)

                            # Validation Metric
                            f_recon = interp1d(
                                recon_km["time"],
                                recon_km["survival"],
                                bounds_error=False,
                                fill_value="extrapolate",
                            )
                            recon_vals_at_orig_times = f_recon(arm_df["time"].values)

                            mae = mean_absolute_error(
                                arm_df["survival"].values,
                                recon_vals_at_orig_times,
                            )

                            if mae > max_mae_across_arms:
                                max_mae_across_arms = mae

                            # Store result
                            st.session_state.reconstruction_results[arm_name] = {
                                "ipd": ipd_df,
                                "recon_curve": recon_km,
                                "mae": mae,
                                "original_curve": arm_df,
                                "color": arm_color,
                            }

                        except Exception as e:
                            st.error(f"Error reconstructing {arm_name}: {e}")

                        progress_bar.progress((i + 1) / n_arms)

                    # Overall quality based on worst arm
                    if st.session_state.reconstruction_results:
                        if max_mae_across_arms < 2.0:
                            st.session_state.validation_status = "High Confidence"
                            st.session_state.validation_color = "green"
                        elif max_mae_across_arms < 5.0:
                            st.session_state.validation_status = "Moderate - Use Caution"
                            st.session_state.validation_color = "orange"
                        else:
                            st.session_state.validation_status = "Poor - Do Not Trust"
                            st.session_state.validation_color = "red"
            
            st.rerun()

    # 2. THE DASHBOARD
    # We now check specifically if results exist
    results = st.session_state.get("reconstruction_results", {})
    has_run = st.session_state.get("reconstruction_has_run", False)

    if results:
        # --- SUCCESS STATE ---
        tabs = st.tabs(list(results.keys()))

        for idx, (arm_name, res) in enumerate(results.items()):
            with tabs[idx]:
                col1, col2 = st.columns([3, 1])

                with col1:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(
                        res["original_curve"]["time"],
                        res["original_curve"]["survival"],
                        "o",
                        markersize=3,
                        alpha=0.5,
                        color="black",
                        label="Digitised",
                    )
                    ax.plot(
                        res["recon_curve"]["time"],
                        res["recon_curve"]["survival"],
                        "-",
                        linewidth=2,
                        color="blue",
                        label="Reconstructed",
                    )
                    ax.set_title(f"{arm_name} – Overlay")
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Survival (%)")
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    st.pyplot(fig)

                with col2:
                    mae = res["mae"]
                    st.metric("MAE Error", f"{mae:.2f}%")
                    if mae < 2.0:
                        st.success("✅ High Accuracy")
                    elif mae < 5.0:
                        st.warning("⚠️ Moderate Deviation")
                    else:
                        st.error("❌ Poor Fit")
                        
                    st.caption(f"Reconstructed {len(res['ipd'])} patients.")

        # Global status banner
        st.markdown("---")
        status = st.session_state.validation_status
        color = st.session_state.validation_color
        
        if color == "green": st.success(f"Overall Status: {status}")
        elif color == "orange": st.warning(f"Overall Status: {status}")
        elif color == "red": st.error(f"Overall Status: {status}")

        if st.button("Proceed to Clinical Analysis ➡️", type="primary"):
            st.session_state.step = 5
            st.rerun()

    elif has_run and not results:
        # --- FAILURE STATE (Ran but empty) ---
        st.error("Reconstruction failed for all arms. Check your inputs in Step 2 and Step 3.")
        
    else:
        # --- IDLE STATE (Hasn't run yet) ---
        st.info("Click 'Run Reconstruction Engine' to generate patients and validate accuracy.")


# --- STEP 5: ANALYSIS & KNOWLEDGE BASE ---
elif st.session_state.step == 5:
    st.header("Step 5: Clinical Analysis & Knowledge Base")
    
    # A. PERFORM ANALYSIS (Only if not loaded from disk)
    if not st.session_state.get("loaded_from_disk"):
        if 'reconstruction_results' in st.session_state:
            from src.core.analysis import run_clinical_analysis
            results = run_clinical_analysis(st.session_state.reconstruction_results)
            
            if results['status'] == 'success':
                st.session_state.analysis_results = results
            else:
                st.error(results['msg'])
    
    # B. PREPARE DISPLAY DATA
    summary = {}
    
    # Case 1: Loaded from Knowledge Base (meta already saved)
    if st.session_state.get("loaded_from_disk"):
        summary = st.session_state.get("clinical_summary", {})

    # Case 2: Fresh analysis in this session
    elif st.session_state.get("analysis_results"):
        res = st.session_state.analysis_results
        summary = {
            "comparison": res['comparison'],
            "hr": res['stats']['hr'],
            "ci_lower": res['stats']['ci_lower'],
            "ci_upper": res['stats']['ci_upper'],
            "p_value": res['stats']['p_value'],
            "medians": res['medians'],
        }

    # C. RENDER DASHBOARD
    if summary:
        # 1. Metrics Header
        st.subheader(f"📊 Results: {summary.get('comparison', 'Analysis')}")
        if summary.get('name'):
            st.caption(
                f"{summary['name']} | "
                f"{summary.get('disease','')} | "
                f"{summary.get('phase','')}"
            )
            
        c1, c2, c3 = st.columns(3)
        c1.metric("Hazard Ratio", f"{summary.get('hr', 0):.2f}")
        c2.metric(
            "95% CI",
            f"{summary.get('ci_lower', 0):.2f} - {summary.get('ci_upper', 0):.2f}",
        )
        c3.metric("P-Value", f"{summary.get('p_value', 0):.4f}")
        
        # 2. Survival Curves (from reconstructed IPD)
        st.subheader("Survival Curves")
        if st.session_state.get('reconstruction_results'):
            fig, ax = plt.subplots(figsize=(10, 5))
            for name, res in st.session_state.reconstruction_results.items():
                color = res.get('color', 'blue')
                ax.plot(
                    res['recon_curve']['time'],
                    res['recon_curve']['survival'],
                    linewidth=2,
                    label=name,
                    color=color,
                )
            ax.set_ylim(0, 105)
            ax.set_xlabel("Time")
            ax.set_ylabel("Survival (%)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)

        st.markdown("---")
        
        # Build LLM context string from summary
        medians = summary.get("medians", {}) or {}
        med_parts = []
        for arm, val in medians.items():
            try:
                med_parts.append(f"{arm}: {float(val):.1f}m")
            except Exception:
                med_parts.append(f"{arm}: {val}")
        medians_str = ", ".join(med_parts) if med_parts else "Not available"

        n_per_arm = summary.get("n_per_arm", {}) or {}
        n_parts = [f"{arm} (n={n})" for arm, n in n_per_arm.items()]
        n_str = ", ".join(n_parts) if n_parts else "Not available"

        context = (
            f"TRIAL: {summary.get('name', 'Unknown')}\n"
            f"Disease: {summary.get('disease', 'Unknown')}\n"
            f"Setting: {summary.get('setting', 'Unknown')}\n"
            f"Phase: {summary.get('phase', 'Unknown')}\n"
            f"Endpoint: {summary.get('endpoint', 'Unknown')}\n"
            f"Comparison: {summary.get('comparison', 'Not specified')}\n"
            f"Stats: HR {summary.get('hr', 0):.2f} "
            f"(95% CI {summary.get('ci_lower', 0):.2f}–{summary.get('ci_upper', 0):.2f}), "
            f"p = {summary.get('p_value', 0):.4f}\n"
            f"Median survival per arm: {medians_str}\n"
            f"Population per arm: {n_str}\n"
        )
        
        # 3. AGENT + SAVE BLOCKS
        col_chat, col_save = st.columns([2, 1])
        
        # --- CHAT WITH LLM (Single-trial) ---
        with col_chat:
            st.subheader("💬 AI Assistant")

            # Build richer context using everything we know
            # summary already has HR, CI, p, medians, comparison
            # clinical_summary (if present) may add disease, setting, phase, endpoint, n_per_arm
            meta = st.session_state.get("clinical_summary", {})
            disease = meta.get("disease", "")
            setting = meta.get("setting", "N/A")
            phase = meta.get("phase", "N/A")
            endpoint = meta.get("endpoint", "N/A")
            n_per_arm = meta.get("n_per_arm", {})
            n_str = ", ".join([f"{arm} (n={n})" for arm, n in n_per_arm.items()]) if n_per_arm else "N/A"

            medians = summary.get("medians", {})
            medians_str = ", ".join([f"{arm}: {val:.1f}m" for arm, val in medians.items()]) if medians else "N/A"

            context = (
                f"TRIAL: {meta.get('name', 'Unknown')}\n"
                f" - Disease: {disease}\n"
                f" - Setting: {setting}\n"
                f" - Phase: {phase}\n"
                f" - Endpoint: {endpoint}\n"
                f" - Comparison: {summary.get('comparison')}\n"
                f" - Stats: HR {summary.get('hr', 0):.2f} "
                f"(95% CI {summary.get('ci_lower',0):.2f}-{summary.get('ci_upper',0):.2f}), "
                f"p={summary.get('p_value',0):.4f}\n"
                f" - Survival: Medians [{medians_str}]\n"
                f" - Population: {n_str}\n"
            )

            if "messages" not in st.session_state:
                st.session_state.messages = [
                    {"role": "assistant",
                    "content": "I have analyzed this trial. Ask me about the clinical significance."}
                ]

            for msg in st.session_state.messages:
                st.chat_message(msg["role"]).write(msg["content"])

            if prompt := st.chat_input("Ask a question..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.chat_message("user").write(prompt)

                # REAL LLM CALL
                from src.core.llm_bridge import ask_gemini
                with st.spinner("🤖 Analyzing clinical data..."):
                    ai_response = ask_gemini(context, prompt)

                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                st.chat_message("assistant").write(ai_response)


        # --- SAVE / KNOWLEDGE BASE FORM ---
        with col_save:
            st.subheader("💾 Knowledge Base")
            
            if st.session_state.get("loaded_from_disk"):
                st.success("✅ Trial already saved in Library")
                st.json(summary, expanded=False)
                if st.button("🆕 Start New Extraction", key="new_ext_saved"):
                    st.session_state.clear()
                    st.rerun()
            else:
                with st.form("save_form"):
                    st.write("Save this analysis to the Knowledge Base.")
                    
                    c_a, c_b = st.columns(2)
                    with c_a:
                        t_name = st.text_input(
                            "Trial Name", placeholder="e.g. RELATIVITY-047"
                        )
                        t_disease = st.text_input(
                            "Disease", placeholder="e.g. Melanoma"
                        )
                        t_endpoint = st.selectbox(
                            "Endpoint", ["OS", "PFS", "DFS", "EFS"]
                        )
                    with c_b:
                        t_setting = st.text_input(
                            "Setting", placeholder="e.g. First-line metastatic"
                        )
                        t_phase = st.selectbox(
                            "Phase",
                            ["Phase III", "Phase II", "Phase I/II", "Retrospective"],
                        )
                    
                    submitted = st.form_submit_button("Finalize & Save")
                    
                    if submitted:
                        if t_name and t_disease:
                            # Construct Rich Metadata to persist
                            meta_info = summary.copy()
                            meta_info.update(
                                {
                                    "name": t_name,
                                    "disease": t_disease,
                                    "endpoint": t_endpoint,
                                    "setting": t_setting,
                                    "phase": t_phase,
                                }
                            )
                            
                            from src.core.project_io import save_project
                            success, msg = save_project(st.session_state, meta_info)
                            
                            if success:
                                st.success("Saved to Library.")
                                st.session_state.clinical_summary = meta_info
                                st.session_state.loaded_from_disk = True
                                st.rerun()
                            else:
                                st.error(msg)
                        else:
                            st.warning("Trial Name and Disease are required.")
    else:
        st.info("Analysis pending. Please run reconstruction first in Step 4.")
