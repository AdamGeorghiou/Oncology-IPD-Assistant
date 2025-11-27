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
        tab_auto, tab_manual = st.tabs(["🤖 Auto-Detect", "✂️ Manual Crop"])
        
        # --- AUTO DETECT ---
        with tab_auto:
            col_btn, col_info = st.columns([1, 2])
            with col_btn:
                run_auto = st.button("Run Auto-Detection", type="primary", use_container_width=True)
            with col_info:
                st.caption("Attempts to find X and Y axes automatically using edge detection")
            
            if run_auto:
                with st.spinner("Analyzing..."):
                    cropped_img, metadata = detect_axes_and_crop(save_path)
                    st.session_state.crop_metadata = metadata
                    st.session_state.cropped_img = cropped_img
            
            if 'crop_metadata' in st.session_state:
                st.divider()
                
                if st.session_state.crop_metadata['status'] == 'success':
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption("Detection (green=X-axis, red=Y-axis)")
                        st.image(cv2.cvtColor(st.session_state.crop_metadata['debug_image'], cv2.COLOR_BGR2RGB), use_column_width=True)
                    
                    with col2:
                        st.caption("Cropped Result")
                        st.image(cv2.cvtColor(st.session_state.cropped_img, cv2.COLOR_BGR2RGB), use_column_width=True)
                    
                    # Confirm button centered
                    col_l, col_c, col_r = st.columns([1, 2, 1])
                    with col_c:
                        if st.button("✅ Confirm & Continue", type="primary", use_container_width=True):
                            st.session_state.step = 2
                            st.rerun()
                else:
                    st.warning("⚠️ Auto-detection failed. Try Manual Crop instead.")
                    st.image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), caption="Original Image", width=400)

        # --- MANUAL CROP ---
        with tab_manual:
            st.caption("Drag the sliders to select the plot area (excluding axes labels and at-risk table)")
            
            # Show original image first for reference
            col_img, col_sliders = st.columns([2, 1])
            
            with col_sliders:
                st.markdown("**Vertical Bounds**")
                top_pct = st.slider("Top", 0, 100, 5, format="%d%%", key="top")
                bottom_pct = st.slider("Bottom", 0, 100, 85, format="%d%%", key="bottom")
                
                st.markdown("**Horizontal Bounds**")
                left_pct = st.slider("Left", 0, 100, 10, format="%d%%", key="left")
                right_pct = st.slider("Right", 0, 100, 95, format="%d%%", key="right")
            
            # Calculate pixel coords
            c_y1, c_y2 = int(height * top_pct / 100), int(height * bottom_pct / 100)
            c_x1, c_x2 = int(width * left_pct / 100), int(width * right_pct / 100)
            
            with col_img:
                # Draw crop rectangle on original
                preview_img = original_img.copy()
                cv2.rectangle(preview_img, (c_x1, c_y1), (c_x2, c_y2), (0, 255, 0), 3)
                st.image(cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            st.divider()
            
            # Validation and result
            if c_y2 > c_y1 and c_x2 > c_x1:
                manual_crop = original_img[c_y1:c_y2, c_x1:c_x2]
                
                col_preview, col_action = st.columns([2, 1])
                
                with col_preview:
                    st.caption("Cropped Preview")
                    st.image(cv2.cvtColor(manual_crop, cv2.COLOR_BGR2RGB), use_column_width=True)
                
                with col_action:
                    st.markdown(f"**Size:** {c_x2-c_x1} × {c_y2-c_y1} px")
                    st.markdown("")  # Spacing
                    if st.button("✅ Use This Crop", type="primary", use_container_width=True):
                        st.session_state.cropped_img = manual_crop
                        st.session_state.crop_metadata = {
                            "status": "manual",
                            "origin": (c_x1, c_y2),
                            "x_len_px": c_x2 - c_x1,
                            "y_len_px": c_y2 - c_y1
                        }
                        st.session_state.step = 2
                        st.rerun()
            else:
                st.error("Invalid selection. Ensure Top < Bottom and Left < Right.")
# =============================================================================
# STEP 2: CURVE EXTRACTION - ENHANCED VERSION
# =============================================================================
# Replace your existing Step 2 code block in app.py with this version
#
# Features:
# - Pre-flight quality assessment
# - Enhanced auto extraction with adaptive tolerances
# - Improved seeded extraction with multi-pixel sampling
# - Manual tracer with zoom preview
# - NEW: Vision LLM extraction (most robust)
# =============================================================================

elif st.session_state.step == 2:
    st.header("Step 2: Curve Extraction")
    
    # --- CONFIGURATION PANEL ---
    col_cfg, col_view = st.columns([1, 2])
    
    with col_cfg:
        st.subheader("Axis Scaling")
        max_x = st.number_input("Max Time (X-Axis)", value=30.0, step=1.0, key="max_x")
        max_y = st.number_input("Max Survival (Y-Axis)", value=100.0, step=10.0, key="max_y")
        
        # --- PRE-FLIGHT QUALITY CHECK ---
        st.markdown("---")
        st.subheader("📊 Image Quality")
        
        if st.button("Analyze Image"):
            from src.core.extractor import assess_extraction_difficulty
            
            assessment = assess_extraction_difficulty(st.session_state.cropped_img)
            st.session_state.quality_assessment = assessment
        
        if 'quality_assessment' in st.session_state:
            assessment = st.session_state.quality_assessment
            score = assessment['score']
            
            # Color-coded score
            if score >= 65:
                st.success(f"Quality Score: {score}/100 ✅")
            elif score >= 40:
                st.warning(f"Quality Score: {score}/100 ⚠️")
            else:
                st.error(f"Quality Score: {score}/100 ❌")
            
            st.caption(f"Recommended method: **{assessment['recommendation'].upper()}**")
            
            with st.expander("Details"):
                for reason in assessment['reasons']:
                    st.write(f"• {reason}")
    
    # --- EXTRACTION METHODS (TABS) ---
    tab_auto, tab_seeded, tab_llm, tab_manual = st.tabs([
        "🤖 Auto", 
        "🎯 Seeded", 
        "🧠 Vision LLM",
        "✍️ Manual"
    ])
    
    # Helper function to add extracted arm
    def add_arm(df, source, color_hex="#000000", name=None):
        if 'extracted_arms' not in st.session_state:
            st.session_state.extracted_arms = []
        idx = len(st.session_state.extracted_arms) + 1
        
        arm_data = {
            "id": idx,
            "name": name or f"Arm {idx}",
            "data": df,
            "source": source,
            "color": color_hex
        }
        st.session_state.extracted_arms.append(arm_data)
    
    # Default colors for arms
    DEFAULT_COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e", "#9467bd", "#8c564b"]
    
    # =================================================================
    # TAB 1: AUTO EXTRACTION (Enhanced)
    # =================================================================
    with tab_auto:
        st.markdown("**Automatic color-based extraction** - Best for high-contrast images")
        
        col_btn, col_status = st.columns([1, 2])
        
        with col_btn:
            if st.button("🔍 Run Auto-Extraction", type="primary", key="auto_btn"):
                from src.core.extractor import auto_extract_curves, get_extraction_summary
                
                with st.spinner("Detecting curves..."):
                    arms = auto_extract_curves(st.session_state.cropped_img, max_x, max_y)
                    
                    if arms:
                        # Clear existing auto-extracted arms
                        st.session_state.extracted_arms = [
                            a for a in st.session_state.get('extracted_arms', [])
                            if a['source'] != 'Auto'
                        ]
                        
                        for i, arm_df in enumerate(arms):
                            color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
                            add_arm(arm_df, "Auto", color)
                        
                        summary = get_extraction_summary(arms)
                        st.session_state.auto_summary = summary
                        st.success(f"Found {len(arms)} curves!")
                        st.rerun()
                    else:
                        st.error("No curves detected. Try Seeded or Vision LLM method.")
        
        with col_status:
            if 'auto_summary' in st.session_state:
                summary = st.session_state.auto_summary
                st.caption(f"Detected {summary['count']} arms")
                for arm_info in summary['arms']:
                    st.caption(f"  • Arm {arm_info['index']+1}: {arm_info['points']} points, "
                              f"survival {arm_info['start_survival']:.0f}% → {arm_info['end_survival']:.0f}%")
    
    # =================================================================
    # TAB 2: SEEDED EXTRACTION (Enhanced)
    # =================================================================
    with tab_seeded:
        st.markdown("**Click on a curve** to extract it - More reliable for tricky images")
        
        from streamlit_image_coordinates import streamlit_image_coordinates
        from src.core.seeded_extractor import (
            get_hsv_at_point, 
            extract_single_color_arm,
            get_color_info,
            visualize_color_match
        )
        
        # Display image and get click coordinates
        img_rgb = cv2.cvtColor(st.session_state.cropped_img, cv2.COLOR_BGR2RGB)
        
        col_img, col_info = st.columns([3, 1])
        
        with col_img:
            coords = streamlit_image_coordinates(
                Image.fromarray(img_rgb), 
                key="seed_click"
            )
        
        with col_info:
            if coords:
                x, y = coords['x'], coords['y']
                st.write(f"📍 Click: ({x}, {y})")
                
                # Sample color (5x5 region for robustness)
                hsv = get_hsv_at_point(st.session_state.cropped_img, x, y, sample_size=5)
                
                if hsv is not None:
                    color_info = get_color_info(hsv)
                    
                    # Show color preview
                    st.write(f"**Color:** {color_info['color_name']}")
                    st.write(f"HSV: {color_info['hsv']}")
                    st.write(f"Saturation: {color_info['saturation_level']}")
                    
                    # Preview matched pixels
                    with st.expander("Preview color match"):
                        preview = visualize_color_match(st.session_state.cropped_img, hsv)
                        st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), 
                                caption="Green = matched pixels", width=300)
                    
                    # Extract button
                    if st.button("✅ Extract This Curve", key="seeded_extract"):
                        with st.spinner("Extracting..."):
                            df = extract_single_color_arm(
                                st.session_state.cropped_img, 
                                hsv, max_x, max_y,
                                use_delta_e=True  # Use improved color matching
                            )
                            
                            if df is not None and len(df) > 10:
                                # Auto-assign color based on detected hue
                                h = hsv[0]
                                if h < 15 or h > 165:
                                    color = "#d62728"  # Red
                                elif h < 45:
                                    color = "#ff7f0e"  # Orange
                                elif h < 75:
                                    color = "#2ca02c"  # Green
                                elif h < 135:
                                    color = "#1f77b4"  # Blue
                                else:
                                    color = "#9467bd"  # Purple
                                
                                add_arm(df, "Seeded", color, name=f"{color_info['color_name']} Curve")
                                st.success("Curve extracted!")
                                st.rerun()
                            else:
                                st.error("Could not extract curve. Try clicking directly on the line.")
    
    # =================================================================
    # TAB 3: VISION LLM EXTRACTION (New!)
    # =================================================================
    with tab_llm:
        st.markdown("**AI-powered extraction** - Most robust for difficult images")
        st.caption("Uses Gemini Vision (~$0.001 per extraction)")
        
        col_llm_cfg, col_llm_btn = st.columns([2, 1])
        
        with col_llm_cfg:
            num_arms_hint = st.number_input(
                "Expected number of arms (0 = auto-detect)", 
                min_value=0, max_value=10, value=2, key="llm_num_arms"
            )
            
            arm_names_input = st.text_input(
                "Arm names (comma-separated, optional)",
                placeholder="e.g., Treatment, Control",
                key="llm_arm_names"
            )
        
        with col_llm_btn:
            if st.button("🧠 Extract with Vision LLM", type="primary", key="llm_extract_btn"):
                from src.core.llm_extractor import extract_curves_with_vision_llm
                
                # Parse arm names
                arm_names = None
                if arm_names_input.strip():
                    arm_names = [n.strip() for n in arm_names_input.split(',')]
                
                with st.spinner("AI is analyzing the curves..."):
                    result = extract_curves_with_vision_llm(
                        st.session_state.cropped_img,
                        max_time_x=max_x,
                        max_surv_y=max_y,
                        num_arms=num_arms_hint if num_arms_hint > 0 else None,
                        arm_names=arm_names
                    )
                    
                    if result['status'] == 'success':
                        # Clear existing LLM-extracted arms
                        st.session_state.extracted_arms = [
                            a for a in st.session_state.get('extracted_arms', [])
                            if a['source'] != 'Vision LLM'
                        ]
                        
                        for i, arm_df in enumerate(result['arms']):
                            color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
                            # Use name from LLM if available
                            name = arm_df.attrs.get('name', f'LLM Arm {i+1}')
                            add_arm(arm_df, "Vision LLM", color, name=name)
                        
                        st.session_state.llm_metadata = result.get('metadata', {})
                        st.success(f"Extracted {len(result['arms'])} curves!")
                        st.rerun()
                    else:
                        st.error(f"Extraction failed: {result.get('error', 'Unknown error')}")
        
        # Show metadata if available
        if 'llm_metadata' in st.session_state and st.session_state.llm_metadata:
            with st.expander("Extraction Details"):
                st.json(st.session_state.llm_metadata)
    
# =============================================================================
# MANUAL TAB - SIMPLE VERSION
# =============================================================================
# Replace your "with tab_manual:" block with this

    with tab_manual:
        from streamlit_drawable_canvas import st_canvas
        from src.core.tracer import process_manual_drawing
        
        img_rgb = cv2.cvtColor(st.session_state.cropped_img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        st.markdown("**Draw the curve manually**")
        
        # --- CONTROLS ---
        c1, c2 = st.columns([1, 3])
        with c1:
            stroke_width = st.slider("Stroke Width", 1, 5, 2, key="stroke_w")
        with c2:
            stroke_color = st.color_picker("Stroke Color", "#FF0000", key="stroke_c")
        
        # --- SCALE UP IMAGE (fixed 2x for better precision) ---
        scale = 1.0
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_scaled = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # --- DRAWING CANVAS ---
        canvas = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_image=Image.fromarray(img_scaled),
            height=new_h,
            width=new_w,
            drawing_mode="freedraw",
            key="manual_canvas"
        )
        
        # --- BUTTONS ---
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📈 Process Drawing", type="primary", use_container_width=True):
                df = process_manual_drawing(canvas, new_w, new_h)
                
                if df is not None and len(df) > 5:
                    # Convert to data coordinates
                    df['time'] = df['rel_time'] * max_x
                    df['survival'] = df['rel_surv'] * max_y
                    
                    add_arm(df[['time', 'survival']], "Manual", stroke_color)
                    st.success(f"✅ Added curve ({len(df)} points)")
                    st.rerun()
                else:
                    st.error("Draw a longer line along the curve")
        
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.rerun()
    
    # =================================================================
    # CURVE MANAGEMENT & PREVIEW (Right Column)
    # =================================================================
    with col_view:
        st.subheader("Extracted Curves")
        
        if st.session_state.get('extracted_arms'):
            arms = st.session_state.extracted_arms
            
            # --- ARM LIST WITH EDITING ---
            for i, arm in enumerate(arms):
                with st.container():
                    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
                    
                    with c1:
                        new_name = st.text_input(
                            f"Name", 
                            value=arm['name'], 
                            key=f"name_{i}",
                            label_visibility="collapsed"
                        )
                        arm['name'] = new_name
                    
                    with c2:
                        new_color = st.color_picker(
                            "Color", 
                            arm['color'], 
                            key=f"color_{i}",
                            label_visibility="collapsed"
                        )
                        arm['color'] = new_color
                    
                    with c3:
                        st.caption(f"📊 {arm['source']}")
                    
                    with c4:
                        if st.button("🗑️", key=f"del_{i}"):
                            st.session_state.extracted_arms.pop(i)
                            st.rerun()
            
            st.markdown("---")
            
            # --- PREVIEW PLOT ---
            fig, ax = plt.subplots(figsize=(8, 5))
            
            for arm in arms:
                ax.plot(
                    arm['data']['time'], 
                    arm['data']['survival'],
                    linewidth=2,
                    label=f"{arm['name']} ({arm['source']})",
                    color=arm['color']
                )
            
            ax.set_xlim(0, max_x)
            ax.set_ylim(0, 105)
            ax.set_xlabel("Time")
            ax.set_ylabel("Survival (%)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower left')
            st.pyplot(fig)
            
            # --- STATISTICS ---
            with st.expander("📊 Curve Statistics"):
                for arm in arms:
                    df = arm['data']
                    st.write(f"**{arm['name']}**")
                    st.write(f"  • Points: {len(df)}")
                    st.write(f"  • Time range: {df['time'].min():.1f} - {df['time'].max():.1f}")
                    st.write(f"  • Survival: {df['survival'].iloc[0]:.1f}% → {df['survival'].iloc[-1]:.1f}%")
            
            # --- PROCEED BUTTON ---
            st.markdown("---")
            
            if len(arms) < 2:
                st.warning("Tip: Most analyses need at least 2 arms (treatment vs control)")
            
            if st.button("✅ Lock Curves & Proceed to At-Risk →", type="primary", use_container_width=True):
                st.session_state.step = 3
                st.rerun()
        
        else:
            st.info("👈 Use one of the extraction methods to add curves")
            
            # Quick tips
            with st.expander("💡 Which method should I use?"):
                st.markdown("""
                **🤖 Auto** - Try this first. Works well on:
                - High contrast images
                - Distinct colored curves (blue vs green)
                - Clean backgrounds
                
                **🎯 Seeded** - Use when Auto fails:
                - Click directly on each curve
                - Better for similar colors
                - Good for 2-3 curves
                
                **🧠 Vision LLM** - Most reliable:
                - Handles difficult images
                - Works on low contrast
                - Small cost (~$0.001)
                
                **✍️ Manual** - Last resort:
                - Draw the curve yourself
                - Use zoom preview for accuracy
                - Good for partial extractions
                """)

# =============================================================================
# STEP 3: AT-RISK TABLE - UPDATED VERSION
# =============================================================================
# Replace your existing Step 3 code block in app.py with this version
# This adds multiple extraction methods: Enhanced OCR and Gemini Vision LLM

elif st.session_state.step == 3:
    st.header("Step 3: At-Risk Data")
    
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.subheader("1. Configure")
        
        # Load Image
        full_img = cv2.imread(st.session_state.current_image_path)
        h, w = full_img.shape[:2]
        
        # Time Intervals
        t_start = st.number_input("Start", value=0.0, step=1.0)
        t_end = st.number_input("End", value=30.0, step=1.0)
        t_step = st.number_input("Step", value=3.0, min_value=0.1, step=1.0)
        
        if st.button("Initialize Grid"):
            times = [round(t, 1) for t in np.arange(t_start, t_end + 0.1, t_step) if t <= t_end]
            st.session_state.risk_times = times
            from src.core.table_manager import create_empty_risk_table
            names = [a['name'] for a in st.session_state.extracted_arms]
            st.session_state.risk_df = create_empty_risk_table(names, times)
            st.rerun()

        st.markdown("---")
        st.markdown("**OCR Crop Region:**")
        st.caption("Crop ONLY the numbers. Exclude headers and arm names.")
        
        # 4-Way Sliders
        c_v1, c_v2 = st.columns(2)
        with c_v1: top_pct = st.slider("Top %", 0, 100, 80)
        with c_v2: bottom_pct = st.slider("Bottom %", 0, 100, 100)
        
        c_h1, c_h2 = st.columns(2)
        with c_h1: left_pct = st.slider("Left %", 0, 100, 15)
        with c_h2: right_pct = st.slider("Right %", 0, 100, 100)
        
        # Calculate Crop
        c_y1 = int(h * top_pct / 100)
        c_y2 = int(h * bottom_pct / 100)
        c_x1 = int(w * left_pct / 100)
        c_x2 = int(w * right_pct / 100)
        
        if c_y2 > c_y1 and c_x2 > c_x1:
            table_crop = full_img[c_y1:c_y2, c_x1:c_x2]
            st.image(cv2.cvtColor(table_crop, cv2.COLOR_BGR2RGB), caption="OCR Target", use_column_width=True)
            
            # =========================================================
            # EXTRACTION METHODS - NEW MULTI-METHOD INTERFACE
            # =========================================================
            if 'risk_df' in st.session_state:
                st.markdown("---")
                st.markdown("**📊 Extraction Methods:**")
                
                n_arms = len(st.session_state.extracted_arms)
                n_cols = len(st.session_state.risk_times)
                arm_names = [a['name'] for a in st.session_state.extracted_arms]
                
                # Store last extraction results for comparison
                if 'last_ocr_result' not in st.session_state:
                    st.session_state.last_ocr_result = None
                if 'last_llm_result' not in st.session_state:
                    st.session_state.last_llm_result = None
                
                # --- METHOD 1: Enhanced OCR ---
                col_ocr, col_llm = st.columns(2)
                
                with col_ocr:
                    if st.button("🔍 Enhanced OCR", use_container_width=True, help="Fast, free, works well on clear images"):
                        from src.core.ocr_table import extract_table_enhanced_ocr
                        
                        with st.spinner("Processing with Enhanced OCR..."):
                            ocr_data = extract_table_enhanced_ocr(
                                table_crop, 
                                n_arms, 
                                n_cols,
                                use_column_segmentation=True
                            )
                            
                            if ocr_data:
                                st.session_state.last_ocr_result = ocr_data
                                st.session_state.risk_df = pd.DataFrame(
                                    ocr_data, 
                                    index=st.session_state.risk_df.index, 
                                    columns=st.session_state.risk_df.columns
                                )
                                st.success("✅ OCR Complete!")
                                st.rerun()
                            else:
                                st.error("❌ OCR failed - try Vision LLM")
                
                # --- METHOD 2: Gemini Vision LLM ---
                with col_llm:
                    if st.button("🤖 Vision LLM", use_container_width=True, help="More accurate, handles difficult cases (~$0.0003/use)"):
                        from src.core.ocr_table import extract_table_with_vision_llm
                        
                        with st.spinner("Analyzing with Gemini Vision..."):
                            llm_data = extract_table_with_vision_llm(
                                table_crop,
                                n_arms,
                                n_cols,
                                arm_names=arm_names,
                                time_points=st.session_state.risk_times
                            )
                            
                            if llm_data:
                                st.session_state.last_llm_result = llm_data
                                st.session_state.risk_df = pd.DataFrame(
                                    llm_data, 
                                    index=st.session_state.risk_df.index, 
                                    columns=st.session_state.risk_df.columns
                                )
                                st.success("✅ Vision LLM Complete!")
                                st.rerun()
                            else:
                                st.error("❌ Vision LLM failed - check API key")
                
                # --- VALIDATION FEEDBACK ---
                if st.session_state.last_ocr_result or st.session_state.last_llm_result:
                    from src.core.ocr_table import validate_at_risk_data
                    
                    current_data = st.session_state.risk_df.values.tolist()
                    validation = validate_at_risk_data(current_data)
                    
                    if validation['warnings']:
                        with st.expander("⚠️ Data Warnings", expanded=False):
                            for warn in validation['warnings']:
                                st.warning(warn)
                    
                    if validation['errors']:
                        for err in validation['errors']:
                            st.error(err)
                
        else:
            st.error("Invalid crop dimensions.")

    # =========================================================
    # RIGHT COLUMN: DATA EDITOR
    # =========================================================
    with c2:
        if 'risk_df' in st.session_state:
            st.subheader("2. Data Entry")
            
            # Status indicator
            extraction_method = "Manual"
            if st.session_state.get('last_llm_result'):
                extraction_method = "Vision LLM"
            elif st.session_state.get('last_ocr_result'):
                extraction_method = "Enhanced OCR"
            
            st.info(f"📝 Verify numbers below. Last extraction: **{extraction_method}**")
            
            # Editable data grid
            edited = st.data_editor(
                st.session_state.risk_df, 
                use_container_width=True, 
                key="editor",
                num_rows="fixed"
            )
            
            # Quick actions
            action_col1, action_col2 = st.columns(2)
            
            with action_col1:
                if st.button("🔄 Reset to Zeros"):
                    st.session_state.risk_df = pd.DataFrame(
                        [[0] * len(st.session_state.risk_times) for _ in range(len(st.session_state.extracted_arms))],
                        index=st.session_state.risk_df.index,
                        columns=st.session_state.risk_df.columns
                    )
                    st.session_state.last_ocr_result = None
                    st.session_state.last_llm_result = None
                    st.rerun()
            
            with action_col2:
                # Reference values from the source image
                if st.button("📋 Show Expected Pattern"):
                    st.info("""
                    **Typical At-Risk Pattern:**
                    - Values should **decrease** over time
                    - First column = total patients enrolled
                    - Last columns may have single digits or zeros
                    """)
            
            st.markdown("---")
            
            if st.button("✅ Validate & Proceed to Reconstruction ➡️", type="primary", use_container_width=True):
                st.session_state.risk_df = edited
                
                # Final validation check
                from src.core.ocr_table import validate_at_risk_data
                final_validation = validate_at_risk_data(edited.values.tolist())
                
                if final_validation['errors']:
                    for err in final_validation['errors']:
                        st.error(err)
                else:
                    if final_validation['warnings']:
                        st.warning(f"Proceeding with {len(final_validation['warnings'])} warnings. Review data if results seem off.")
                    st.session_state.step = 4
                    st.rerun()
        else:
            st.info("👈 Click 'Initialize Grid' to start.")
            
            # Helper: Show example of what the at-risk table looks like
            with st.expander("ℹ️ What is an At-Risk Table?"):
                st.markdown("""
                The **Number at Risk** table appears below survival curves and shows 
                how many patients remain in the study at each time point.
                
                **Example:**
                | Arm | 0 | 6 | 12 | 18 | 24 |
                |-----|---|---|----|----|-----|
                | Treatment | 373 | 350 | 280 | 180 | 90 |
                | Control | 184 | 165 | 120 | 70 | 30 |
                
                These numbers are essential for accurate IPD reconstruction.
                """)

# --- STEP 4: VALIDATION ---
elif st.session_state.step == 4:
    st.header("Step 4: Validation & Quality Check")

    # Reference arm selector (before running reconstruction)
    arms_meta = st.session_state.get('extracted_arms', [])
    arm_names = [a['name'] for a in arms_meta]
    
    if len(arm_names) >= 2:
        st.markdown("##### Select Reference Arm")
        st.caption("The reference arm is typically the control/standard-of-care. HR will be calculated as: Comparator / Reference")
        
        reference_arm = st.selectbox(
            "Reference (control) arm:",
            options=arm_names,
            index=len(arm_names) - 1,  # Default to last arm (often control)
            key="reference_arm_select"
        )
        st.session_state.reference_arm = reference_arm
        
        # Show what comparison will be
        comparator = [a for a in arm_names if a != reference_arm][0]
        st.info(f"📊 Comparison: **{comparator}** vs **{reference_arm}** (HR < 1 favors {comparator})")
        
        st.divider()

    # 1. THE CONTROLLER
    col_run, col_status = st.columns([1, 3])
    
    with col_run:
        if st.button("🔄 Run Reconstruction", type="primary", use_container_width=True):
            from src.core.reconstructor import reconstruct_survival, calculate_km_from_ipd
            from sklearn.metrics import mean_absolute_error
            from scipy.interpolate import interp1d

            # Reset State
            st.session_state.reconstruction_results = {}
            st.session_state.validation_status = "Unknown"
            st.session_state.validation_color = "gray"
            st.session_state.reconstruction_has_run = True

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
                            import traceback
                            print(traceback.format_exc())

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

    with col_status:
        st.caption("Reconstructs individual patient data (IPD) from digitised curves and validates accuracy against original data.")

    # 2. THE DASHBOARD
    results = st.session_state.get("reconstruction_results", {})
    has_run = st.session_state.get("reconstruction_has_run", False)

    if results:
        # --- SUCCESS STATE ---
        
        # Summary line
        total_patients = sum(len(res['ipd']) for res in results.values())
        st.success(f"✅ Reconstructed **{total_patients} patients** across **{len(results)} arms**")
        
        # Tabs for each arm
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
                    ax.set_title(f"{arm_name}")
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Survival (%)")
                    ax.set_ylim(0, 105)
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    st.pyplot(fig)
                    plt.close(fig)

                with col2:
                    # Metrics
                    mae = res["mae"]
                    st.metric("MAE", f"{mae:.2f}%")
                    
                    if mae < 2.0:
                        st.success("High Accuracy")
                    elif mae < 5.0:
                        st.warning("Moderate")
                    else:
                        st.error("Poor Fit")
                    
                    st.caption(f"{len(res['ipd'])} patients")
                    
                    st.divider()
                    
                    # Download button for this arm
                    csv = res['ipd'].to_csv(index=False)
                    st.download_button(
                        "📥 Download IPD",
                        csv,
                        file_name=f"{arm_name.replace(' ', '_')}_ipd.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key=f"dl_{arm_name}"
                    )
                
                # Expandable data preview
                with st.expander("Preview IPD Data"):
                    st.dataframe(res['ipd'].head(15), use_container_width=True)

        # Global status banner
        st.divider()
        
        status = st.session_state.validation_status
        color = st.session_state.validation_color
        
        col_status, col_download, col_next = st.columns([2, 1, 1])
        
        with col_status:
            if color == "green": 
                st.success(f"**Overall: {status}**")
            elif color == "orange": 
                st.warning(f"**Overall: {status}**")
            elif color == "red": 
                st.error(f"**Overall: {status}**")
        
        with col_download:
            # Combined download
            all_ipd = []
            for arm_name, res in results.items():
                arm_ipd = res['ipd'].copy()
                arm_ipd['arm'] = arm_name
                all_ipd.append(arm_ipd)
            combined = pd.concat(all_ipd, ignore_index=True)
            
            st.download_button(
                "📥 Download All IPD",
                combined.to_csv(index=False),
                file_name="combined_ipd.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_next:
            if st.button("Next Step ➡️", type="primary", use_container_width=True):
                st.session_state.step = 5
                st.rerun()

    elif has_run and not results:
        # --- FAILURE STATE ---
        st.error("Reconstruction failed for all arms. Check your inputs in Step 2 and Step 3.")
        
    else:
        # --- IDLE STATE ---
        st.info("Click **Run Reconstruction** to generate individual patient data and validate accuracy.")

# --- STEP 5: ANALYSIS & KNOWLEDGE BASE ---
elif st.session_state.step == 5:
    st.header("Step 5: Clinical Analysis")
    
    # A. PERFORM ANALYSIS (Only if not loaded from disk)
    if not st.session_state.get("loaded_from_disk"):
        if 'reconstruction_results' in st.session_state:
            from src.core.analysis import run_clinical_analysis
            
            # Get reference arm from Step 4 selection
            reference_arm = st.session_state.get('reference_arm', None)
            
            results = run_clinical_analysis(
                st.session_state.reconstruction_results,
                reference_arm=reference_arm
            )
            
            if results['status'] == 'success':
                st.session_state.analysis_results = results
            else:
                st.error(results['msg'])
    
    # B. PREPARE DISPLAY DATA
    summary = {}
    
    # Case 1: Loaded from Knowledge Base
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
            "logrank_p": res['stats'].get('logrank_p'),
            "medians": res['medians'],
            "n_per_arm": res.get('n_per_arm', {}),
        }

    # C. RENDER DASHBOARD
    if summary:
        # --- HEADER ---
        st.subheader(f"📊 {summary.get('comparison', 'Analysis')}")
        if summary.get('name'):
            st.caption(f"**{summary['name']}** • {summary.get('disease', '')} • {summary.get('phase', '')}")
        
        # --- STATS ROW ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Hazard Ratio", f"{summary.get('hr', 0):.2f}")
        c2.metric("95% CI", f"{summary.get('ci_lower', 0):.2f} – {summary.get('ci_upper', 0):.2f}")
        c3.metric("Cox p-value", f"{summary.get('p_value', 0):.4f}")
        logrank_p = summary.get('logrank_p')
        c4.metric("Log-rank p", f"{logrank_p:.4f}" if logrank_p else "N/A")
        
        # --- MEDIAN SURVIVAL ROW ---
        st.markdown("##### Median Survival")
        medians = summary.get("medians", {})
        n_per_arm = summary.get("n_per_arm", {})
        
        if medians:
            med_cols = st.columns(len(medians))
            for idx, (arm, median_val) in enumerate(medians.items()):
                n = n_per_arm.get(arm, "?")
                try:
                    med_cols[idx].metric(
                        f"{arm} (n={n})",
                        f"{float(median_val):.1f} months"
                    )
                except:
                    med_cols[idx].metric(f"{arm} (n={n})", f"{median_val}")
        
        st.divider()
        
        # --- PLOTS ROW ---
        col_km, col_forest = st.columns([2, 1])
        
        # Survival Curves
        with col_km:
            st.markdown("##### Kaplan-Meier Curves")
            if st.session_state.get('reconstruction_results'):
                fig, ax = plt.subplots(figsize=(8, 4))
                for name, res in st.session_state.reconstruction_results.items():
                    color = res.get('color', None)
                    ax.plot(
                        res['recon_curve']['time'],
                        res['recon_curve']['survival'],
                        linewidth=2,
                        label=name,
                        color=color,
                    )
                ax.set_ylim(0, 105)
                ax.set_xlabel("Time (months)")
                ax.set_ylabel("Survival (%)")
                ax.grid(True, alpha=0.3)
                ax.legend(loc='lower left')
                st.pyplot(fig)
                plt.close(fig)
        
        # Forest Plot
        with col_forest:
            st.markdown("##### Hazard Ratio")
            hr = summary.get('hr', 1)
            ci_low = summary.get('ci_lower', hr)
            ci_high = summary.get('ci_upper', hr)
            
            fig, ax = plt.subplots(figsize=(4, 2))
            
            # Plot HR point and CI line
            ax.errorbar(hr, 0, xerr=[[hr - ci_low], [ci_high - hr]], 
                       fmt='o', color='darkblue', markersize=10, capsize=5, linewidth=2)
            
            # Reference line at HR=1
            ax.axvline(x=1, color='gray', linestyle='--', linewidth=1)
            
            # Formatting
            ax.set_xlim(0, max(2, ci_high + 0.3))
            ax.set_ylim(-0.5, 0.5)
            ax.set_xlabel("Hazard Ratio")
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
            # Add text annotation
            ax.text(hr, 0.3, f"HR={hr:.2f}", ha='center', fontsize=10)
            ax.text(hr, -0.35, f"({ci_low:.2f}–{ci_high:.2f})", ha='center', fontsize=8, color='gray')
            
            st.pyplot(fig)
            plt.close(fig)
            
            # Interpretation
            if hr < 1 and summary.get('p_value', 1) < 0.05:
                st.success("Favors treatment arm")
            elif hr > 1 and summary.get('p_value', 1) < 0.05:
                st.warning("Favors control arm")
            else:
                st.info("No significant difference")
        
        st.divider()
        
        # --- CHAT + SAVE ROW ---
        col_chat, col_save = st.columns([2, 1])
        
        # --- AI CHAT ---
        with col_chat:
            st.markdown("##### 💬 AI Assistant")
            
            # Build context
            meta = st.session_state.get("clinical_summary", {})
            medians_str = ", ".join([f"{arm}: {val:.1f}m" for arm, val in medians.items()]) if medians else "N/A"
            n_str = ", ".join([f"{arm} (n={n})" for arm, n in n_per_arm.items()]) if n_per_arm else "N/A"
            
            logrank_str = f"{logrank_p:.4f}" if logrank_p else "N/A"
            
            context = (
                f"TRIAL: {meta.get('name', summary.get('comparison', 'Unknown'))}\n"
                f"Disease: {meta.get('disease', 'Unknown')}\n"
                f"Setting: {meta.get('setting', 'N/A')}\n"
                f"Phase: {meta.get('phase', 'N/A')}\n"
                f"Endpoint: {meta.get('endpoint', 'N/A')}\n"
                f"Comparison: {summary.get('comparison')}\n"
                f"Hazard Ratio: {summary.get('hr', 0):.2f} "
                f"(95% CI {summary.get('ci_lower', 0):.2f}–{summary.get('ci_upper', 0):.2f})\n"
                f"Cox p-value: {summary.get('p_value', 0):.4f}\n"
                f"Log-rank p-value: {logrank_str}\n"
                f"Median survival: {medians_str}\n"
                f"Population: {n_str}\n"
            )

            if "messages" not in st.session_state:
                st.session_state.messages = [
                    {"role": "assistant",
                     "content": "I've analyzed this trial data. Ask me about clinical significance, interpretation, or comparisons."}
                ]

            # Chat container with fixed height
            chat_container = st.container(height=300)
            with chat_container:
                for msg in st.session_state.messages:
                    st.chat_message(msg["role"]).write(msg["content"])

            if prompt := st.chat_input("Ask about this trial..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                from src.core.llm_bridge import ask_gemini
                with st.spinner("Thinking..."):
                    ai_response = ask_gemini(context, prompt)
                
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                st.rerun()

        # --- SAVE TO KNOWLEDGE BASE ---
        with col_save:
            st.markdown("##### 💾 Knowledge Base")
            
            if st.session_state.get("loaded_from_disk"):
                st.success("✅ Saved to Library")
                with st.expander("View metadata"):
                    st.json(summary)
                if st.button("🆕 New Extraction", use_container_width=True):
                    st.session_state.clear()
                    st.rerun()
            else:
                with st.form("save_form"):
                    t_name = st.text_input("Trial Name*", placeholder="e.g. RELATIVITY-047")
                    t_disease = st.text_input("Disease*", placeholder="e.g. Melanoma")
                    
                    c_a, c_b = st.columns(2)
                    with c_a:
                        t_endpoint = st.selectbox("Endpoint", ["OS", "PFS", "DFS", "EFS"])
                    with c_b:
                        t_phase = st.selectbox("Phase", ["Phase III", "Phase II", "Phase I/II", "Retrospective"])
                    
                    t_setting = st.text_input("Setting", placeholder="e.g. First-line metastatic")
                    
                    submitted = st.form_submit_button("💾 Save to Library", use_container_width=True)
                    
                    if submitted:
                        if t_name and t_disease:
                            meta_info = summary.copy()
                            meta_info.update({
                                "name": t_name,
                                "disease": t_disease,
                                "endpoint": t_endpoint,
                                "setting": t_setting,
                                "phase": t_phase,
                            })
                            
                            from src.core.project_io import save_project
                            success, msg = save_project(st.session_state, meta_info)
                            
                            if success:
                                st.success("Saved!")
                                st.session_state.clinical_summary = meta_info
                                st.session_state.loaded_from_disk = True
                                st.rerun()
                            else:
                                st.error(msg)
                        else:
                            st.warning("Trial Name and Disease required.")
                
                # Download report button
                st.divider()
                
                # Build report text
                logrank_report = f"{logrank_p:.5f}" if logrank_p else "N/A"
                
                report = f"""CLINICAL ANALYSIS REPORT
========================

Comparison: {summary.get('comparison', 'N/A')}

STATISTICAL RESULTS
-------------------
Hazard Ratio: {summary.get('hr', 0):.3f}
95% CI: {summary.get('ci_lower', 0):.3f} – {summary.get('ci_upper', 0):.3f}
Cox p-value: {summary.get('p_value', 0):.5f}
Log-rank p-value: {logrank_report}

MEDIAN SURVIVAL
---------------
{chr(10).join([f'{arm}: {val:.1f} months (n={n_per_arm.get(arm, "?")})' for arm, val in medians.items()])}

Generated by IPD Reconstruction Tool
"""
                st.download_button(
                    "📄 Download Report",
                    report,
                    file_name="analysis_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )
    else:
        st.info("Run reconstruction in Step 4 first.")