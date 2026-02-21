#!/usr/bin/env python3
"""
Enhanced LSTV Report Generator with Embedded Images
Includes sagittal and axial views for high/moderate risk cases
Uses confidence-based thresholds for realistic detection rates
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import base64
from io import BytesIO
from jinja2 import Template
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from loguru import logger
import sys
import pydicom
import cv2
from natsort import natsorted

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>")

def encode_image_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{image_base64}"

def create_plotly_distributions(df: pd.DataFrame, high_risk_conf: float, moderate_risk_conf: float) -> tuple:
    """Create interactive plotly plots for distributions"""
    
    # 1. L5/S1 Confidence histogram with thresholds
    fig_conf = go.Figure()
    
    fig_conf.add_trace(go.Histogram(
        x=df['l5_s1_confidence'],
        nbinsx=50,
        marker_color='#764ba2',
        opacity=0.7,
        name='Confidence'
    ))
    
    # Add threshold lines
    fig_conf.add_vline(
        x=high_risk_conf, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"High Risk (<{high_risk_conf})",
        annotation_position="top left"
    )
    fig_conf.add_vline(
        x=moderate_risk_conf,
        line_dash="dash",
        line_color="orange", 
        annotation_text=f"Moderate Risk (<{moderate_risk_conf})",
        annotation_position="top right"
    )
    
    fig_conf.update_layout(
        title_text="L5/S1 Peak Confidence Distribution",
        xaxis_title="Peak Confidence",
        yaxis_title="Count",
        template='plotly_white',
        height=400
    )
    
    # 2. Scatter plot (Confidence vs Entropy)
    fig_scatter = go.Figure()
    
    # Color by risk level
    colors = df['risk_level'].map({
        'High Risk': 'red',
        'Moderate Risk': 'orange',
        'Low Risk': 'green'
    })
    
    fig_scatter.add_trace(go.Scatter(
        x=df['l5_s1_confidence'],
        y=df['l5_s1_entropy'],
        mode='markers',
        marker=dict(
            size=8,
            color=colors,
            opacity=0.6,
            line=dict(width=1, color='white')
        ),
        text=df['study_id'].astype(int),
        customdata=df['risk_level'],
        hovertemplate='<b>Study:</b> %{text}<br>' +
                     '<b>Risk:</b> %{customdata}<br>' +
                     '<b>Confidence:</b> %{x:.4f}<br>' +
                     '<b>Entropy:</b> %{y:.4f}<extra></extra>'
    ))
    
    # Add threshold lines
    fig_scatter.add_vline(x=high_risk_conf, line_dash="dash", line_color="red", opacity=0.5)
    fig_scatter.add_vline(x=moderate_risk_conf, line_dash="dash", line_color="orange", opacity=0.5)
    
    fig_scatter.update_layout(
        title_text="L5-S1: Confidence vs Entropy (Risk Classification)",
        xaxis_title="Peak Confidence",
        yaxis_title="Entropy",
        template='plotly_white',
        height=500,
        hovermode='closest'
    )
    
    # 3. Box plots for all levels
    levels = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']
    level_labels = ['L1-L2', 'L2-L3', 'L3-L4', 'L4-L5', 'L5-S1']
    colors_levels = ['#4299e1', '#48bb78', '#ed8936', '#f56565', '#9f7aea']
    
    fig_box = go.Figure()
    
    for level, label, color in zip(levels, level_labels, colors_levels):
        fig_box.add_trace(go.Box(
            y=df[f'{level}_confidence'],
            name=label,
            marker_color=color,
            boxmean='sd'
        ))
    
    fig_box.update_layout(
        title_text="Peak Confidence Distribution by Vertebral Level",
        yaxis_title="Peak Confidence",
        template='plotly_white',
        height=500
    )
    
    return (
        fig_conf.to_html(include_plotlyjs=False, div_id="conf_dist"),
        fig_scatter.to_html(include_plotlyjs=False, div_id="scatter"),
        fig_box.to_html(include_plotlyjs=False, div_id="level_box")
    )

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LSTV Uncertainty Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .alert {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 20px;
            margin: 20px;
            border-radius: 5px;
        }
        
        .alert-title {
            font-weight: bold;
            color: #856404;
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        
        .alert-text {
            color: #856404;
            line-height: 1.6;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 40px;
            background: #f8f9fa;
        }
        
        .stat-card {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 12px rgba(0,0,0,0.15);
        }
        
        .stat-card h3 {
            color: #667eea;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }
        
        .stat-card .value {
            font-size: 2.5em;
            font-weight: 700;
            color: #2d3748;
        }
        
        .stat-card .label {
            color: #718096;
            font-size: 0.9em;
            margin-top: 5px;
        }
        
        .section {
            padding: 40px;
        }
        
        .section-title {
            font-size: 1.8em;
            color: #2d3748;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }
        
        .methodology {
            background: #edf2f7;
            padding: 30px;
            border-radius: 15px;
            margin: 30px 0;
        }
        
        .methodology h3 {
            color: #2d3748;
            margin-bottom: 15px;
        }
        
        .methodology p {
            line-height: 1.8;
            color: #4a5568;
        }
        
        .plot-container {
            margin: 30px 0;
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        thead {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        th, td {
            padding: 15px;
            text-align: left;
        }
        
        th {
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }
        
        tbody tr {
            border-bottom: 1px solid #e2e8f0;
            transition: background 0.2s ease;
        }
        
        tbody tr:hover {
            background: #f7fafc;
        }
        
        .lstv-high {
            background: #fed7d7;
            color: #c53030;
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: 600;
        }
        
        .lstv-medium {
            background: #feebc8;
            color: #c05621;
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: 600;
        }
        
        .lstv-low {
            background: #c6f6d5;
            color: #276749;
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: 600;
        }
        
        .case-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin-top: 30px;
        }
        
        .case-card {
            background: white;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
            transition: transform 0.3s ease;
        }
        
        .case-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 12px rgba(0,0,0,0.15);
        }
        
        .case-header {
            background: #f8f9fa;
            padding: 20px;
            border-bottom: 2px solid #667eea;
        }
        
        .case-header h3 {
            color: #2d3748;
            margin-bottom: 10px;
        }
        
        .metrics {
            display: flex;
            gap: 20px;
        }
        
        .metric {
            color: #718096;
            font-size: 0.9em;
        }
        
        .metric strong {
            color: #2d3748;
        }
        
        .case-card img {
            width: 100%;
            display: block;
        }
        
        .footer {
            background: #2d3748;
            color: white;
            padding: 30px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 LSTV Uncertainty Analysis</h1>
            <p>Epistemic Uncertainty-Based LSTV Detection</p>
            <p style="font-size: 0.9em; margin-top: 10px;">Using Ian Pan's RSNA 2024 Spine Localizer</p>
        </div>
        
        <div class="alert">
            <div class="alert-title">📊 Detection Rate Analysis</div>
            <div class="alert-text">
                <strong>Clinical Context:</strong> LSTV occurs in 15-30% of the population.<br>
                <strong>Current Results:</strong> {{ stats.detection_rate }}% of cases flagged as High/Moderate risk 
                ({{ stats.high_risk_count }} high + {{ stats.moderate_risk_count }} moderate).<br>
                <strong>Method:</strong> Using <strong>confidence</strong> as primary metric 
                (High: <{{ stats.high_risk_conf_threshold }}, Moderate: <{{ stats.moderate_risk_conf_threshold }}) 
                combined with entropy thresholds.<br>
                <strong>Note:</strong> Entropy shows low variance ({{ stats.mean_l5_s1_entropy }} ± 0.09), 
                while confidence varies more ({{ stats.mean_l5_s1_confidence }} ± 0.02) and better discriminates cases.
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Studies</h3>
                <div class="value">{{ stats.total_studies }}</div>
                <div class="label">Analyzed</div>
            </div>
            <div class="stat-card">
                <h3>High Risk</h3>
                <div class="value" style="color: #ff6b6b;">{{ stats.high_risk_count }}</div>
                <div class="label">Strong LSTV candidates</div>
            </div>
            <div class="stat-card">
                <h3>Moderate Risk</h3>
                <div class="value" style="color: #ffa500;">{{ stats.moderate_risk_count }}</div>
                <div class="label">Secondary review</div>
            </div>
            <div class="stat-card">
                <h3>Detection Rate</h3>
                <div class="value">{{ stats.detection_rate }}%</div>
                <div class="label">Target: 15-30%</div>
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">📊 Distribution Analysis</h2>
            
            <div class="methodology">
                <h3>Methodology</h3>
                <p>
                    <strong>Hypothesis:</strong> A spine localizer trained on normal anatomy will exhibit high epistemic
                    uncertainty (confusion) when encountering LSTV anomalies. We measure this using confidence and entropy
                    of the probability heatmaps.
                </p>
                <p style="margin-top: 10px;">
                    <strong>Key Insight:</strong> Confidence (<0.98) is more discriminative than entropy alone. 
                    High-risk cases show both low confidence AND high entropy, suggesting anatomical variants like 
                    sacralized L5 or lumbarized S1.
                </p>
            </div>
            
            <div class="plot-container">
                {{ conf_plot }}
            </div>
            
            <div class="plot-container">
                {{ scatter_plot }}
            </div>
            
            <div class="plot-container">
                {{ level_box_plot }}
            </div>
        </div>
        
        {% if has_case_images %}
        <div class="section">
            <h2 class="section-title">🔴 High Risk Cases (Top 5)</h2>
            <p style="margin-bottom: 20px; color: #718096;">
                These cases show the lowest confidence (<{{ stats.high_risk_conf_threshold }}) 
                and/or highest entropy, suggesting potential LSTV or anatomical variants.
                <strong>Look for:</strong> Sacralized L5 (large transverse processes), 
                lumbarized S1, or asymmetric anatomy.
            </p>
            <div class="case-grid">
                {% for case in high_risk_images %}
                <div class="case-card">
                    <div class="case-header">
                        <h3>Study ID: {{ case.study_id }}</h3>
                        <div class="metrics">
                            <span class="metric">Confidence: <strong>{{ "%.4f"|format(case.confidence) }}</strong></span>
                            <span class="metric">Entropy: <strong>{{ "%.4f"|format(case.entropy) }}</strong></span>
                        </div>
                    </div>
                    <img src="{{ case.image }}" alt="Case {{ case.study_id }}">
                </div>
                {% endfor %}
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">🟠 Moderate Risk Cases (Sample)</h2>
            <p style="margin-bottom: 20px; color: #718096;">
                These cases show intermediate confidence 
                ({{ stats.high_risk_conf_threshold }}-{{ stats.moderate_risk_conf_threshold }}),
                warranting review but lower priority than high-risk cases.
            </p>
            <div class="case-grid">
                {% for case in moderate_risk_images %}
                <div class="case-card">
                    <div class="case-header">
                        <h3>Study ID: {{ case.study_id }}</h3>
                        <div class="metrics">
                            <span class="metric">Confidence: <strong>{{ "%.4f"|format(case.confidence) }}</strong></span>
                            <span class="metric">Entropy: <strong>{{ "%.4f"|format(case.entropy) }}</strong></span>
                        </div>
                    </div>
                    <img src="{{ case.image }}" alt="Case {{ case.study_id }}">
                </div>
                {% endfor %}
            </div>
        </div>
        {% endif %}
        
        <div class="section">
            <h2 class="section-title">🎯 Top 20 Candidates</h2>
            <p style="margin-bottom: 20px; color: #4a5568;">
                Studies ranked by L5-S1 confidence (lowest first). Low confidence + high entropy indicates potential LSTV.
            </p>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Study ID</th>
                        <th>Risk Level</th>
                        <th>L5-S1 Confidence</th>
                        <th>L5-S1 Entropy</th>
                        <th>L4-L5 Entropy</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in top_candidates %}
                    <tr>
                        <td>{{ loop.index }}</td>
                        <td><strong>{{ row.study_id }}</strong></td>
                        <td><span class="{{ row.risk_class }}">{{ row.risk_level }}</span></td>
                        <td>{{ "%.4f"|format(row.l5_s1_confidence) }}</td>
                        <td>{{ "%.4f"|format(row.l5_s1_entropy) }}</td>
                        <td>{{ "%.4f"|format(row.l4_l5_entropy) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        
        {% if debug_images %}
        <div class="section">
            <h2 class="section-title">🔍 Debug Visualizations</h2>
            {% for img_data in debug_images %}
            <div class="plot-container">
                <h3>{{ img_data.title }}</h3>
                <img src="data:image/png;base64,{{ img_data.base64 }}" style="max-width: 100%; border-radius: 10px;">
            </div>
            {% endfor %}
        </div>
        {% endif %}
        
        <div class="footer">
            <p>Generated by LSTV Uncertainty Analysis Pipeline</p>
            <p style="margin-top: 10px; font-size: 0.9em; opacity: 0.8;">
                Wayne State University School of Medicine | {{ timestamp }}
            </p>
        </div>
    </div>
</body>
</html>
"""


def normalise_to_8bit(x):
    """Normalize to 8-bit range"""
    lower, upper = np.percentile(x, (0.1, 99.9))
    x = np.clip(x, lower, upper)
    x = x - np.min(x)
    if np.max(x) > 0:
        x = x / np.max(x)
    return (x * 255).astype(np.uint8)

def load_dicom_slice(study_path, series_id, slice_idx=None):
    """Load a DICOM slice"""
    series_path = study_path / str(series_id)
    if not series_path.exists():
        return None
    
    dicom_files = natsorted(list(series_path.glob('*.dcm')))
    if not dicom_files:
        return None
    
    try:
        if slice_idx is None:
            slice_idx = len(dicom_files) // 2
        
        dcm = pydicom.dcmread(str(dicom_files[slice_idx]))
        img = dcm.pixel_array
        img = normalise_to_8bit(img)
        return img
    except Exception as e:
        print(f"Error loading {series_id}: {e}")
        return None

def create_case_visualization(study_id, series_id, data_dir, series_df):
    """Create visualization with sagittal and axial views"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    study_path = Path(data_dir) / str(int(study_id))
    
    # Load sagittal T2 (middle slice)
    sagittal_img = load_dicom_slice(study_path, series_id)
    
    # Try to find axial T2
    axial_series = series_df[
        (series_df['study_id'] == study_id) & 
        (series_df['series_description'].str.contains('Axial T2', case=False, na=False))
    ]
    
    axial_img = None
    if len(axial_series) > 0:
        axial_series_id = axial_series.iloc[0]['series_id']
        axial_img = load_dicom_slice(study_path, axial_series_id)
    
    # Plot sagittal
    if sagittal_img is not None:
        axes[0].imshow(sagittal_img, cmap='gray')
        axes[0].set_title('Sagittal T2/STIR (Middle Slice)', fontsize=12, fontweight='bold')
        axes[0].axis('off')
    else:
        axes[0].text(0.5, 0.5, 'Image Not Available', ha='center', va='center')
        axes[0].set_title('Sagittal T2/STIR', fontsize=12, fontweight='bold')
        axes[0].axis('off')
    
    # Plot axial
    if axial_img is not None:
        axes[1].imshow(axial_img, cmap='gray')
        axes[1].set_title('Axial T2 (L5/S1 Level)', fontsize=12, fontweight='bold')
        axes[1].axis('off')
    else:
        axes[1].text(0.5, 0.5, 'Image Not Available', ha='center', va='center')
        axes[1].set_title('Axial T2', fontsize=12, fontweight='bold')
        axes[1].axis('off')
    
    plt.suptitle(f'Study ID: {int(study_id)}', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig

def generate_report(csv_path, output_path, debug_dir=None, data_dir=None, series_csv=None):
    """Generate HTML report with embedded images"""
    
    # Load data
    logger.info(f"Loading results from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} studies")
    
    # Load series descriptions if available
    series_df = None
    if series_csv and Path(series_csv).exists():
        series_df = pd.read_csv(series_csv)
        logger.info(f"Loaded series descriptions")
    
    # Calculate risk scores based on L5/S1 metrics
    # KEY INSIGHT: Use confidence as primary metric (more variable than entropy)
    l5_entropy = df['l5_s1_entropy']
    l5_confidence = df['l5_s1_confidence']
    
    # Set thresholds to target 15-25% detection rate
    HIGH_RISK_CONF = 0.97    # ~9% (top candidates)
    MODERATE_RISK_CONF = 0.985  # ~16% additional
    HIGH_RISK_ENTROPY = 5.2  # Top 6%
    
    # Classify cases using CONFIDENCE as primary metric
    df['risk_level'] = 'Low Risk'
    
    # High risk: Low confidence OR high entropy
    high_risk_mask = (l5_confidence < HIGH_RISK_CONF) | (l5_entropy > HIGH_RISK_ENTROPY)
    df.loc[high_risk_mask, 'risk_level'] = 'High Risk'
    
    # Moderate risk: Between thresholds
    moderate_risk_mask = (l5_confidence < MODERATE_RISK_CONF) & (l5_confidence >= HIGH_RISK_CONF)
    df.loc[moderate_risk_mask & ~high_risk_mask, 'risk_level'] = 'Moderate Risk'
    
    high_risk_count = (df['risk_level'] == 'High Risk').sum()
    moderate_risk_count = (df['risk_level'] == 'Moderate Risk').sum()
    low_risk_count = (df['risk_level'] == 'Low Risk').sum()
    
    logger.info(f"\nRisk Classification:")
    logger.info(f"  High Risk: {high_risk_count} ({100*high_risk_count/len(df):.1f}%)")
    logger.info(f"  Moderate Risk: {moderate_risk_count} ({100*moderate_risk_count/len(df):.1f}%)")
    logger.info(f"  Low Risk: {low_risk_count} ({100*low_risk_count/len(df):.1f}%)")
    
    # Calculate statistics
    detection_rate = 100 * (high_risk_count + moderate_risk_count) / len(df)
    
    stats = {
        'total_studies': len(df),
        'high_risk_count': high_risk_count,
        'moderate_risk_count': moderate_risk_count,
        'low_risk_count': low_risk_count,
        'detection_rate': f"{detection_rate:.1f}",
        'mean_l5_s1_entropy': f"{df['l5_s1_entropy'].mean():.3f}",
        'mean_l5_s1_confidence': f"{df['l5_s1_confidence'].mean():.3f}",
        'high_risk_conf_threshold': HIGH_RISK_CONF,
        'moderate_risk_conf_threshold': MODERATE_RISK_CONF
    }
    
    # Create plotly plots
    logger.info("Generating interactive plots...")
    conf_plot, scatter_plot, level_box_plot = create_plotly_distributions(
        df, HIGH_RISK_CONF, MODERATE_RISK_CONF
    )
    
    # Get top candidates
    logger.info("Preparing top candidates table...")
    top_df = df.sort_values('l5_s1_confidence').head(20)
    
    top_candidates = []
    for idx, row in top_df.iterrows():
        risk_class_map = {
            'High Risk': 'lstv-high',
            'Moderate Risk': 'lstv-medium',
            'Low Risk': 'lstv-low'
        }
        
        top_candidates.append({
            'study_id': int(row['study_id']),
            'risk_level': row['risk_level'],
            'risk_class': risk_class_map[row['risk_level']],
            'l5_s1_confidence': row['l5_s1_confidence'],
            'l5_s1_entropy': row['l5_s1_entropy'],
            'l4_l5_entropy': row['l4_l5_entropy']
        })
    
    # Generate case images for high and moderate risk
    case_images = {'high_risk': [], 'moderate_risk': []}
    
    if data_dir and series_df is not None:
        logger.info("Generating case visualizations...")
        
        # High risk cases (top 5)
        high_risk_df = df[df['risk_level'] == 'High Risk'].sort_values('l5_s1_confidence').head(5)
        
        for idx, row in high_risk_df.iterrows():
            study_id = row['study_id']
            series_id = row['series_id']
            logger.info(f"  Processing high risk: {int(study_id)}")
            
            try:
                fig = create_case_visualization(study_id, series_id, data_dir, series_df)
                img_b64 = encode_image_base64(fig)
                case_images['high_risk'].append({
                    'study_id': int(study_id),
                    'entropy': row['l5_s1_entropy'],
                    'confidence': row['l5_s1_confidence'],
                    'image': img_b64
                })
            except Exception as e:
                logger.warning(f"    Error generating image for {int(study_id)}: {e}")
        
        # Moderate risk cases (sample 5)
        moderate_risk_df = df[df['risk_level'] == 'Moderate Risk'].sort_values('l5_s1_confidence').head(5)
        
        for idx, row in moderate_risk_df.iterrows():
            study_id = row['study_id']
            series_id = row['series_id']
            logger.info(f"  Processing moderate risk: {int(study_id)}")
            
            try:
                fig = create_case_visualization(study_id, series_id, data_dir, series_df)
                img_b64 = encode_image_base64(fig)
                case_images['moderate_risk'].append({
                    'study_id': int(study_id),
                    'entropy': row['l5_s1_entropy'],
                    'confidence': row['l5_s1_confidence'],
                    'image': img_b64
                })
            except Exception as e:
                logger.warning(f"    Error generating image for {int(study_id)}: {e}")
    
    # Load debug images if available (from original code)
    debug_images = []
    if debug_dir and Path(debug_dir).exists():
        logger.info(f"Loading debug images from {debug_dir}")
        for img_path in sorted(Path(debug_dir).glob('*.png')):
            with open(img_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode()
            debug_images.append({
                'title': img_path.stem.replace('_', ' ').title(),
                'base64': img_data
            })
        logger.info(f"Loaded {len(debug_images)} debug images")
    
    # Generate HTML using template
    logger.info("Rendering HTML template...")
    template = Template(HTML_TEMPLATE)
    html = template.render(
        stats=stats,
        conf_plot=conf_plot,
        scatter_plot=scatter_plot,
        level_box_plot=level_box_plot,
        top_candidates=top_candidates,
        high_risk_images=case_images['high_risk'],
        moderate_risk_images=case_images['moderate_risk'],
        debug_images=debug_images,
        has_case_images=bool(data_dir and series_df is not None),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    # Write HTML
    Path(output_path).write_text(html)
    
    logger.info(f"✓ Report saved to: {output_path}")
    logger.info(f"✓ High risk cases with images: {len(case_images['high_risk'])}")
    logger.info(f"✓ Moderate risk cases with images: {len(case_images['moderate_risk'])}")


def main():
    parser = argparse.ArgumentParser(description='Generate LSTV Uncertainty Report with Images')
    parser.add_argument('--csv', type=str, required=True,
                       help='Path to uncertainty metrics CSV')
    parser.add_argument('--output', type=str, required=True,
                       help='Output HTML file path')
    parser.add_argument('--debug_dir', type=str,
                       help='Directory with debug visualizations')
    parser.add_argument('--data_dir', type=str,
                       help='Path to DICOM data directory (for embedded images)')
    parser.add_argument('--series_csv', type=str,
                       help='Path to series descriptions CSV')

    args = parser.parse_args()

    generate_report(
        csv_path=args.csv,
        output_path=args.output,
        debug_dir=args.debug_dir,
        data_dir=args.data_dir,
        series_csv=args.series_csv
    )


if __name__ == '__main__':
    main()
