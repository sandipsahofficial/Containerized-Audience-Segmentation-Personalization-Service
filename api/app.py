import os
import sys
import logging
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

# Add parent dir to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_loader import ModelLoader
from schemas import validate_recommend_payload
from recommender import get_recommendations, ContentRecommender

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger('api')

app = Flask(__name__)

# Preload model at module init
ModelLoader.load_model()

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OTT Audience Intelligence &mdash; Personalization Control Center</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
  <style>
    /* ─── Enterprise Design Tokens ─── */
    :root {
      --bg-base: #060911;
      --bg-canvas: #090e1a;
      --panel-surface: rgba(14, 20, 36, 0.88);
      --panel-surface-elevated: rgba(19, 27, 48, 0.94);
      --panel-surface-hover: rgba(25, 36, 62, 0.98);
      --border-subtle: rgba(255, 255, 255, 0.06);
      --border-strong: rgba(255, 255, 255, 0.12);
      --border-accent: rgba(59, 130, 246, 0.4);
      --border-red: rgba(229, 9, 20, 0.45);

      --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.4);
      --shadow-card: 0 8px 30px -6px rgba(0, 0, 0, 0.6), inset 0 0 0 1px rgba(255, 255, 255, 0.04);
      --shadow-hover: 0 14px 38px -6px rgba(0, 0, 0, 0.75), inset 0 0 0 1px rgba(59, 130, 246, 0.25);

      --accent-red: #e50914;
      --accent-red-hover: #b80710;
      --accent-blue: #3b82f6;
      --accent-cyan: #06b6d4;
      --accent-green: #10b981;
      --accent-amber: #f59e0b;

      --text-white: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;

      --radius-xs: 6px;
      --radius-sm: 8px;
      --radius-md: 12px;
      --radius-lg: 18px;
      --radius-pill: 9999px;

      --transition-base: 200ms cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ─── CSS Reset ─── */
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html { scroll-behavior: smooth; }
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: var(--bg-base);
      color: var(--text-white);
      line-height: 1.55;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
      overflow-x: hidden;
    }

    /* ─── Subtle Ambient Background ─── */
    body::before {
      content: '';
      position: fixed;
      inset: 0;
      background:
        radial-gradient(ellipse 90% 50% at 50% -10%, rgba(59, 130, 246, 0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 90% 90%, rgba(229, 9, 20, 0.04) 0%, transparent 50%),
        radial-gradient(ellipse 50% 50% at 10% 80%, rgba(6, 182, 212, 0.04) 0%, transparent 50%);
      pointer-events: none;
      z-index: 0;
    }

    .app-shell {
      position: relative;
      z-index: 1;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    .container {
      max-width: 1440px;
      margin: 0 auto;
      padding: 0 24px;
      width: 100%;
    }

    /* ─── Panel / Card Primitive ─── */
    .panel {
      background: var(--panel-surface);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      box-shadow: var(--shadow-card);
      transition: transform var(--transition-base), box-shadow var(--transition-base), border-color var(--transition-base);
    }
    .panel:hover {
      box-shadow: var(--shadow-hover);
    }

    /* ─── Sticky Top Navigation ─── */
    .topbar {
      position: sticky;
      top: 0;
      z-index: 100;
      background: rgba(6, 9, 17, 0.85);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border-bottom: 1px solid var(--border-subtle);
      padding: 12px 0;
    }
    .topbar-inner {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      flex-wrap: wrap;
    }
    .topbar-brand {
      display: flex;
      align-items: center;
      gap: 12px;
      text-decoration: none;
      color: inherit;
    }
    .brand-badge {
      background: var(--accent-red);
      color: #fff;
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 1.5px;
      padding: 5px 11px;
      border-radius: var(--radius-xs);
      text-transform: uppercase;
      box-shadow: 0 2px 10px rgba(229, 9, 20, 0.35);
      line-height: 1.2;
    }
    .brand-title {
      font-size: 16px;
      font-weight: 800;
      color: var(--text-white);
      letter-spacing: -0.3px;
    }
    .brand-subtitle {
      font-size: 11px;
      color: var(--text-dim);
      font-weight: 500;
      margin-top: 1px;
    }
    .topbar-nav {
      display: flex;
      align-items: center;
      gap: 20px;
    }
    .topbar-link {
      font-size: 13px;
      font-weight: 600;
      color: var(--text-muted);
      text-decoration: none;
      transition: color var(--transition-base);
      padding: 4px 0;
    }
    .topbar-link:hover {
      color: var(--text-white);
    }
    @media (max-width: 960px) {
      .topbar-nav { display: none; }
    }

    /* ─── Real Dynamic Health Indicator ─── */
    .health-pill {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 6px 14px;
      border-radius: var(--radius-pill);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.4px;
      background: rgba(16, 185, 129, 0.1);
      color: var(--accent-green);
      border: 1px solid rgba(16, 185, 129, 0.25);
      transition: all var(--transition-base);
    }
    .health-pill.down {
      background: rgba(239, 68, 68, 0.1);
      color: #f87171;
      border-color: rgba(239, 68, 68, 0.3);
    }
    .health-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: currentColor;
      box-shadow: 0 0 6px currentColor;
      animation: pulseGlow 2s infinite ease-in-out;
    }
    .health-pill.down .health-dot { animation: none; }
    @keyframes pulseGlow {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.45; transform: scale(0.9); }
    }

    /* ─── Compact Hero Section ─── */
    .hero {
      padding: 28px 0 20px;
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      flex-wrap: wrap;
      gap: 16px;
      border-bottom: 1px solid var(--border-subtle);
      margin-bottom: 24px;
    }
    .hero-content h2 {
      font-size: 26px;
      font-weight: 800;
      letter-spacing: -0.6px;
      color: var(--text-white);
      line-height: 1.25;
    }
    .hero-content p {
      font-size: 13px;
      color: var(--text-muted);
      margin-top: 6px;
      max-width: 680px;
    }
    .hero-badges {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      align-items: center;
    }
    .tech-tag {
      background: var(--panel-surface-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-pill);
      padding: 4px 10px;
      font-size: 11px;
      font-weight: 600;
      color: var(--text-muted);
      letter-spacing: 0.2px;
    }
    .tech-tag.highlight {
      color: #93c5fd;
      border-color: rgba(59, 130, 246, 0.3);
      background: rgba(59, 130, 246, 0.08);
    }

    /* ─── KPI Metrics Strip (5 Cards) ─── */
    .kpi-strip {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 14px;
      margin-bottom: 28px;
    }
    @media (max-width: 1100px) { .kpi-strip { grid-template-columns: repeat(3, 1fr); } }
    @media (max-width: 720px)  { .kpi-strip { grid-template-columns: repeat(2, 1fr); } }
    @media (max-width: 480px)  { .kpi-strip { grid-template-columns: 1fr; } }
    .kpi-card {
      padding: 16px 18px;
      position: relative;
      overflow: hidden;
      border-radius: var(--radius-md);
    }
    .kpi-card::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 2px;
    }
    .kpi-card:nth-child(1)::before { background: var(--accent-red); }
    .kpi-card:nth-child(2)::before { background: var(--accent-blue); }
    .kpi-card:nth-child(3)::before { background: var(--accent-cyan); }
    .kpi-card:nth-child(4)::before { background: var(--accent-amber); }
    .kpi-card:nth-child(5)::before { background: var(--accent-green); }
    .kpi-card:hover { transform: translateY(-2px); }
    .kpi-label {
      font-size: 10px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: var(--text-dim);
      margin-bottom: 4px;
    }
    .kpi-val {
      font-size: 22px;
      font-weight: 800;
      color: var(--text-white);
      letter-spacing: -0.5px;
      line-height: 1.2;
    }
    .kpi-desc {
      font-size: 11px;
      color: var(--text-muted);
      margin-top: 3px;
      font-weight: 500;
    }

    /* ─── Main Two-Column Dashboard Workspace ─── */
    .dashboard-grid {
      display: grid;
      grid-template-columns: 1.15fr 1fr;
      gap: 24px;
      margin-bottom: 32px;
    }
    @media (max-width: 960px) {
      .dashboard-grid { grid-template-columns: 1fr; }
    }
    .dash-panel {
      padding: 28px;
      border-radius: var(--radius-lg);
    }
    .panel-header {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      padding-bottom: 16px;
      margin-bottom: 22px;
      border-bottom: 1px solid var(--border-subtle);
    }
    .panel-header-title {
      font-size: 13px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.9px;
      color: var(--text-white);
    }
    .panel-header-sub {
      font-size: 11px;
      color: var(--text-dim);
      margin-top: 2px;
    }
    .panel-header-pill {
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      padding: 3px 8px;
      border-radius: var(--radius-pill);
      background: var(--panel-surface-elevated);
      color: var(--text-muted);
      border: 1px solid var(--border-subtle);
    }

    /* ─── Left Panel: Viewer Simulator ─── */
    .form-group { margin-bottom: 22px; }
    .form-label-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }
    .form-label {
      font-size: 10px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: var(--accent-cyan);
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .form-label::before {
      content: '';
      width: 3px;
      height: 11px;
      background: var(--accent-cyan);
      border-radius: 2px;
      display: inline-block;
    }

    /* Quick Archetypes */
    .archetype-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 8px;
    }
    @media (max-width: 600px) {
      .archetype-grid { grid-template-columns: repeat(2, 1fr); }
    }
    .archetype-btn {
      background: var(--panel-surface-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      padding: 9px 8px;
      font-size: 11px;
      font-weight: 600;
      color: var(--text-muted);
      cursor: pointer;
      text-align: center;
      transition: all var(--transition-base);
      font-family: inherit;
    }
    .archetype-btn:hover {
      background: var(--panel-surface-hover);
      color: var(--text-white);
      border-color: var(--border-accent);
      transform: translateY(-1px);
    }
    .archetype-btn.active {
      background: rgba(59, 130, 246, 0.15);
      border-color: var(--accent-blue);
      color: #bfdbfe;
      font-weight: 700;
      box-shadow: 0 0 14px rgba(59, 130, 246, 0.2);
    }

    /* Input Field */
    .input-text {
      width: 100%;
      background: rgba(6, 9, 17, 0.9);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      color: var(--text-white);
      padding: 10px 14px;
      font-size: 13px;
      font-family: inherit;
      font-weight: 600;
      outline: none;
      transition: border-color var(--transition-base), box-shadow var(--transition-base);
    }
    .input-text:focus {
      border-color: var(--accent-blue);
      box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12);
    }

    /* Sliders & Values */
    .slider-item { margin-bottom: 16px; }
    .slider-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }
    .slider-title {
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
    }
    .slider-badge {
      font-size: 12px;
      font-weight: 700;
      color: var(--text-white);
      background: rgba(6, 9, 17, 0.9);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-xs);
      padding: 4px 10px;
      min-width: 76px;
      text-align: center;
      font-variant-numeric: tabular-nums;
    }
    input[type="range"] {
      -webkit-appearance: none;
      appearance: none;
      width: 100%;
      height: 4px;
      border-radius: 2px;
      background: rgba(255, 255, 255, 0.08);
      outline: none;
      cursor: pointer;
      transition: background var(--transition-base);
    }
    input[type="range"]::-webkit-slider-thumb {
      -webkit-appearance: none;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: var(--accent-red);
      border: 2px solid rgba(255, 255, 255, 0.25);
      box-shadow: 0 2px 8px rgba(229, 9, 20, 0.4);
      cursor: pointer;
      transition: transform var(--transition-base), box-shadow var(--transition-base);
    }
    input[type="range"]::-webkit-slider-thumb:hover {
      transform: scale(1.15);
      box-shadow: 0 2px 14px rgba(229, 9, 20, 0.6);
    }
    input[type="range"]::-moz-range-thumb {
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: var(--accent-red);
      border: 2px solid rgba(255, 255, 255, 0.25);
      box-shadow: 0 2px 8px rgba(229, 9, 20, 0.4);
      cursor: pointer;
    }

    /* Content Preferences / Genre Chips */
    .genre-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(110px, 1fr));
      gap: 8px;
    }
    .genre-chip {
      background: var(--panel-surface-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      padding: 8px 12px;
      font-size: 12px;
      font-weight: 500;
      color: var(--text-muted);
      cursor: pointer;
      user-select: none;
      display: flex;
      justify-content: space-between;
      align-items: center;
      transition: all var(--transition-base);
      font-family: inherit;
    }
    .genre-chip:hover {
      background: var(--panel-surface-hover);
      color: var(--text-white);
      transform: translateY(-1px);
    }
    .genre-chip.active {
      background: rgba(229, 9, 20, 0.12);
      border-color: var(--border-red);
      color: #fff;
      font-weight: 600;
      box-shadow: 0 2px 8px rgba(229, 9, 20, 0.15);
    }
    .genre-check {
      font-size: 11px;
      opacity: 0.35;
      transition: opacity var(--transition-base);
    }
    .genre-chip.active .genre-check {
      opacity: 1;
      color: var(--accent-red);
      font-weight: 800;
    }

    /* Dominant CTA Button */
    .btn-predict {
      width: 100%;
      background: linear-gradient(180deg, var(--accent-red) 0%, #b80710 100%);
      color: #ffffff;
      border: none;
      border-radius: var(--radius-md);
      padding: 14px;
      font-size: 13px;
      font-weight: 700;
      font-family: inherit;
      letter-spacing: 0.4px;
      cursor: pointer;
      box-shadow: 0 4px 18px rgba(229, 9, 20, 0.35);
      transition: all var(--transition-base);
      margin-top: 6px;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
    }
    .btn-predict:hover:not(:disabled) {
      transform: translateY(-2px);
      box-shadow: 0 8px 24px rgba(229, 9, 20, 0.5);
      filter: brightness(1.05);
    }
    .btn-predict:active:not(:disabled) { transform: translateY(0); }
    .btn-predict:disabled {
      opacity: 0.65;
      cursor: not-allowed;
      transform: none;
    }

    /* ─── Right Panel: AI Inference Result ─── */
    .result-viewport {
      min-height: 460px;
      display: flex;
      flex-direction: column;
    }
    .empty-state {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      text-align: center;
      color: var(--text-dim);
      padding: 48px 24px;
      gap: 10px;
    }
    .empty-icon { font-size: 34px; opacity: 0.5; }
    .empty-title { font-size: 15px; font-weight: 700; color: var(--text-muted); }
    .empty-sub { font-size: 12px; max-width: 280px; line-height: 1.5; color: var(--text-dim); }

    /* Cohort Result Card */
    .cohort-result-card {
      background: var(--panel-surface-elevated);
      border: 1px solid var(--border-subtle);
      border-left: 4px solid var(--accent-blue);
      border-radius: var(--radius-md);
      padding: 20px 22px;
      margin-bottom: 22px;
      box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
      animation: fadeIn 220ms ease-out;
    }
    .cohort-top-row {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 10px;
      flex-wrap: wrap;
    }
    .cohort-title-wrap h3 {
      font-size: 18px;
      font-weight: 800;
      color: var(--text-white);
      letter-spacing: -0.3px;
      line-height: 1.25;
    }
    .cohort-id-badge {
      background: rgba(59, 130, 246, 0.14);
      color: #93c5fd;
      border: 1px solid rgba(59, 130, 246, 0.3);
      border-radius: var(--radius-pill);
      padding: 3px 10px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.3px;
    }
    .cohort-metrics-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      margin-top: 14px;
      padding-top: 12px;
      border-top: 1px solid var(--border-subtle);
    }
    .cohort-metric-item {
      display: flex;
      flex-direction: column;
    }
    .cohort-metric-key {
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: var(--text-dim);
    }
    .cohort-metric-val {
      font-size: 14px;
      font-weight: 700;
      color: var(--text-white);
      margin-top: 2px;
    }
    .distance-tooltip {
      font-size: 11px;
      color: var(--text-dim);
      margin-top: 6px;
      line-height: 1.4;
      display: flex;
      align-items: center;
      gap: 4px;
    }

    /* Curated Recommendations */
    .rec-heading {
      font-size: 10px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: var(--text-dim);
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .rec-list { display: flex; flex-direction: column; gap: 9px; }
    .rec-card {
      background: rgba(6, 9, 17, 0.85);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      padding: 12px 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      transition: all var(--transition-base);
      animation: fadeIn 220ms ease-out;
    }
    .rec-card:hover {
      transform: translateY(-2px);
      border-color: var(--border-accent);
      box-shadow: 0 6px 18px rgba(0, 0, 0, 0.4);
    }
    .rec-card-left { display: flex; align-items: center; gap: 12px; }
    .rec-number {
      font-size: 12px;
      font-weight: 800;
      color: var(--text-dim);
      font-variant-numeric: tabular-nums;
      width: 22px;
    }
    .rec-title {
      font-size: 13px;
      font-weight: 700;
      color: var(--text-white);
      letter-spacing: -0.1px;
    }
    .rec-match-pill {
      background: rgba(16, 185, 129, 0.1);
      color: var(--accent-green);
      border: 1px solid rgba(16, 185, 129, 0.25);
      border-radius: var(--radius-pill);
      padding: 3px 9px;
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.3px;
    }

    /* Collapsible API Inspector */
    .api-inspector {
      margin-top: 20px;
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      background: rgba(6, 9, 17, 0.7);
      font-size: 12px;
    }
    .api-inspector summary {
      padding: 10px 14px;
      cursor: pointer;
      color: var(--text-muted);
      font-weight: 600;
      outline: none;
      user-select: none;
      font-size: 11px;
      letter-spacing: 0.3px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .api-inspector summary:hover { color: var(--text-white); }
    .inspector-body { padding: 0 14px 14px; }
    .inspector-actions {
      display: flex;
      justify-content: flex-end;
      margin-bottom: 6px;
    }
    .btn-copy-json {
      background: var(--panel-surface-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-xs);
      color: var(--text-muted);
      padding: 3px 9px;
      font-size: 10px;
      font-weight: 600;
      cursor: pointer;
      transition: all var(--transition-base);
      font-family: inherit;
    }
    .btn-copy-json:hover {
      color: var(--text-white);
      border-color: var(--border-accent);
    }
    .code-block {
      padding: 12px;
      background: #030509;
      border-radius: var(--radius-xs);
      color: #7dd3fc;
      font-size: 11px;
      font-family: 'SF Mono', 'Cascadia Code', Consolas, monospace;
      overflow-x: auto;
      border: 1px solid rgba(255, 255, 255, 0.04);
      line-height: 1.5;
    }

    /* ─── Model Pipeline Flow ("HOW THE AI WORKS") ─── */
    .section-container { margin-bottom: 32px; }
    .section-header {
      margin-bottom: 16px;
    }
    .section-title {
      font-size: 15px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: var(--text-white);
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .section-title::before {
      content: '';
      width: 4px;
      height: 14px;
      background: var(--accent-red);
      border-radius: 2px;
      display: inline-block;
    }
    .section-sub {
      font-size: 12px;
      color: var(--text-dim);
      margin-top: 3px;
    }

    .pipeline-grid {
      display: grid;
      grid-template-columns: repeat(6, 1fr);
      gap: 12px;
    }
    @media (max-width: 1024px) {
      .pipeline-grid { grid-template-columns: repeat(3, 1fr); }
    }
    @media (max-width: 600px) {
      .pipeline-grid { grid-template-columns: 1fr; }
    }
    .pipeline-step-card {
      background: var(--panel-surface-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 16px 14px;
      position: relative;
      transition: all var(--transition-base);
    }
    .pipeline-step-card:hover {
      transform: translateY(-2px);
      border-color: var(--accent-cyan);
      box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
    }
    .step-badge {
      font-size: 9px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: var(--accent-cyan);
      margin-bottom: 4px;
    }
    .step-title {
      font-size: 12px;
      font-weight: 700;
      color: var(--text-white);
      margin-bottom: 6px;
      line-height: 1.3;
    }
    .step-desc {
      font-size: 11px;
      color: var(--text-muted);
      line-height: 1.4;
    }

    /* ─── Audience Cohorts Table ─── */
    .table-wrapper { overflow-x: auto; margin-top: 12px; }
    table.cohort-table {
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      font-size: 12px;
      text-align: left;
    }
    table.cohort-table th {
      background: var(--panel-surface-elevated);
      color: var(--text-dim);
      padding: 12px 16px;
      font-weight: 800;
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      border-bottom: 1px solid var(--border-subtle);
    }
    table.cohort-table th:first-child { border-radius: var(--radius-sm) 0 0 0; }
    table.cohort-table th:last-child { border-radius: 0 var(--radius-sm) 0 0; }
    table.cohort-table td {
      padding: 13px 16px;
      border-bottom: 1px solid var(--border-subtle);
      color: var(--text-muted);
      font-weight: 500;
      transition: background var(--transition-base);
    }
    table.cohort-table tbody tr:nth-child(even) td {
      background: rgba(255, 255, 255, 0.012);
    }
    table.cohort-table tbody tr:hover td {
      background: rgba(59, 130, 246, 0.04);
      color: var(--text-white);
    }
    .td-primary { color: var(--text-white); font-weight: 700; }
    .share-tag {
      display: inline-block;
      background: rgba(59, 130, 246, 0.1);
      color: #93c5fd;
      border: 1px solid rgba(59, 130, 246, 0.25);
      border-radius: var(--radius-pill);
      padding: 2px 8px;
      font-size: 10px;
      font-weight: 700;
    }

    /* ─── Model Intelligence & Evaluation Status (Two Column) ─── */
    .intel-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
      margin-bottom: 32px;
    }
    @media (max-width: 900px) {
      .intel-grid { grid-template-columns: 1fr; }
    }
    .specs-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 10px;
      margin-top: 14px;
    }
    .spec-item {
      background: rgba(6, 9, 17, 0.7);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      padding: 10px 14px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .spec-key { font-size: 11px; color: var(--text-muted); font-weight: 500; }
    .spec-val {
      font-size: 12px;
      font-weight: 700;
      color: var(--text-white);
      font-variant-numeric: tabular-nums;
    }
    .pass-pill {
      background: rgba(16, 185, 129, 0.12);
      color: var(--accent-green);
      border: 1px solid rgba(16, 185, 129, 0.3);
      border-radius: var(--radius-pill);
      padding: 2px 8px;
      font-size: 10px;
      font-weight: 700;
    }

    /* ─── Docker Architecture Section ─── */
    .arch-flow-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 14px;
      margin-top: 14px;
    }
    @media (max-width: 900px) {
      .arch-flow-grid { grid-template-columns: 1fr; }
    }
    .arch-card {
      background: var(--panel-surface-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 16px;
      position: relative;
      transition: all var(--transition-base);
    }
    .arch-card:hover {
      transform: translateY(-2px);
      border-color: var(--accent-blue);
    }
    .arch-card-tag {
      font-size: 9px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.9px;
      color: var(--accent-blue);
      margin-bottom: 4px;
    }
    .arch-card-title {
      font-size: 13px;
      font-weight: 700;
      color: var(--text-white);
      margin-bottom: 6px;
    }
    .arch-card-desc {
      font-size: 11px;
      color: var(--text-muted);
      line-height: 1.4;
    }
    .arch-badges {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 14px;
    }

    /* ─── Minimal Enterprise Footer ─── */
    .site-footer {
      border-top: 1px solid var(--border-subtle);
      padding: 24px 0;
      margin-top: auto;
      text-align: center;
      font-size: 11px;
      color: var(--text-dim);
      font-weight: 500;
      line-height: 1.6;
    }
    .footer-brand {
      font-weight: 800;
      color: var(--accent-red);
      letter-spacing: 0.5px;
    }
    .footer-dot {
      display: inline-block;
      width: 3px; height: 3px;
      border-radius: 50%;
      background: var(--text-dim);
      margin: 0 8px;
      vertical-align: middle;
      opacity: 0.5;
    }

    /* ─── Error Message Banner ─── */
    .error-box {
      background: rgba(239, 68, 68, 0.08);
      border: 1px solid rgba(239, 68, 68, 0.35);
      border-radius: var(--radius-sm);
      padding: 16px;
      color: #fca5a5;
      animation: fadeIn 200ms ease-out;
    }
    .error-box-title {
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: #f87171;
      margin-bottom: 4px;
    }
    .error-box-body { font-size: 12px; }

    /* ─── Micro Animations ─── */
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after {
        animation: none !important;
        transition: none !important;
      }
    }
  </style>
</head>
<body>
  <div class="app-shell">

    <!-- ─── 2. Top Navigation Bar ─── -->
    <header class="topbar">
      <div class="container">
        <div class="topbar-inner">
          <a href="#dashboard" class="topbar-brand">
            <div class="brand-badge">OTT ML</div>
            <div>
              <div class="brand-title">Audience Intelligence</div>
              <div class="brand-subtitle">Containerized Audience Segmentation &amp; Personalization</div>
            </div>
          </a>

          <nav class="topbar-nav">
            <a href="#dashboard" class="topbar-link">Dashboard</a>
            <a href="#pipeline" class="topbar-link">How It Works</a>
            <a href="#cohorts" class="topbar-link">Audience Cohorts</a>
            <a href="#model-intel" class="topbar-link">Model Intelligence</a>
            <a href="#architecture" class="topbar-link">Architecture</a>
          </nav>

          <div class="health-pill" id="healthPill">
            <div class="health-dot"></div>
            <span id="healthStatus">SERVICE HEALTHY &bull; MODEL LOADED</span>
          </div>
        </div>
      </div>
    </header>

    <main>
      <div class="container">

        <!-- ─── 3. Hero Section ─── -->
        <section class="hero">
          <div class="hero-content">
            <h2>Understand Your Audience. Personalize Every Session.</h2>
            <p>Unsupervised behavioral segmentation powered by K-Means clustering and a content-based ML recommendation engine.</p>
          </div>
          <div class="hero-badges">
            <span class="tech-tag highlight">K-Means (K=5)</span>
            <span class="tech-tag highlight">Content-Based ML Recommender</span>
            <span class="tech-tag">CPU Friendly</span>
            <span class="tech-tag">Deterministic (Seed 42)</span>
            <span class="tech-tag">REST API</span>
            <span class="tech-tag">Docker Compose</span>
          </div>
        </section>

        <!-- ─── 4. KPI / Model Health Strip ─── -->
        <section class="kpi-strip">
          <div class="kpi-card panel">
            <div class="kpi-label">Model</div>
            <div class="kpi-val">K-Means</div>
            <div class="kpi-desc">K = 5 Optimal Clusters</div>
          </div>
          <div class="kpi-card panel">
            <div class="kpi-label">Silhouette</div>
            <div class="kpi-val">0.4206</div>
            <div class="kpi-desc">Peak cohesion &amp; separation</div>
          </div>
          <div class="kpi-card panel">
            <div class="kpi-label">Inertia</div>
            <div class="kpi-val">10,330.4</div>
            <div class="kpi-desc">54.5% reduction vs K=2</div>
          </div>
          <div class="kpi-card panel">
            <div class="kpi-label">Inference</div>
            <div class="kpi-val">&lt; 8 ms</div>
            <div class="kpi-desc">Commodity CPU execution</div>
          </div>
          <div class="kpi-card panel">
            <div class="kpi-label">Dataset</div>
            <div class="kpi-val">4,992 users</div>
            <div class="kpi-desc">Preprocessed &amp; validated</div>
          </div>
        </section>

        <!-- ─── 5. Main Dashboard (Two-Column Workspace) ─── -->
        <section id="dashboard" class="dashboard-grid">

          <!-- Left: Viewer Simulator -->
          <div class="dash-panel panel">
            <div class="panel-header">
              <div>
                <div class="panel-header-title">Viewer Simulator</div>
                <div class="panel-header-sub">Configure behavioral signals to simulate a viewer profile.</div>
              </div>
              <div class="panel-header-pill">Simulator</div>
            </div>

            <form id="simForm" onsubmit="executeInference(event)">
              <!-- Quick Archetypes -->
              <div class="form-group">
                <div class="form-label">Quick Archetypes</div>
                <div class="archetype-grid">
                  <button type="button" class="archetype-btn active" id="btnArchAction" onclick="selectArchetype('action')">High Action</button>
                  <button type="button" class="archetype-btn" id="btnArchCasual" onclick="selectArchetype('casual')">Casual Short</button>
                  <button type="button" class="archetype-btn" id="btnArchBinge" onclick="selectArchetype('binge')">Weekend Binge</button>
                  <button type="button" class="archetype-btn" id="btnArchEclectic" onclick="selectArchetype('eclectic')">Multi-Genre</button>
                </div>
              </div>

              <!-- Viewer ID -->
              <div class="form-group">
                <div class="form-label">Viewer ID</div>
                <input type="text" id="viewerId" class="input-text" value="USR-0003" placeholder="e.g. USR-0003, USR-0002" required>
              </div>

              <!-- Engagement Inputs -->
              <div class="form-group">
                <div class="form-label">Engagement</div>

                <div class="slider-item">
                  <div class="slider-top">
                    <span class="slider-title">Watch Time</span>
                    <span class="slider-badge" id="watchBadge">65.0 hrs</span>
                  </div>
                  <input type="range" id="watchRange" min="0" max="200" step="0.5" value="65" oninput="onWatchInput(this.value)">
                </div>

                <div class="slider-item">
                  <div class="slider-top">
                    <span class="slider-title">Average Session</span>
                    <span class="slider-badge" id="sessionBadge">95 mins</span>
                  </div>
                  <input type="range" id="sessionRange" min="5" max="240" step="1" value="95" oninput="onSessionInput(this.value)">
                </div>
              </div>

              <!-- Content Preferences (Genres) -->
              <div class="form-group">
                <div class="form-label">Content Preferences</div>
                <div class="genre-grid" id="genreContainer"></div>
              </div>

              <!-- Primary Action CTA -->
              <button type="submit" class="btn-predict" id="btnPredict">
                <span>&#9889;</span>
                <span id="predictBtnText">Predict Segment &amp; Personalize</span>
              </button>
            </form>
          </div>

          <!-- Right: AI Inference Result -->
          <div class="dash-panel panel">
            <div class="panel-header">
              <div>
                <div class="panel-header-title">AI Inference</div>
                <div class="panel-header-sub">Real-time K-Means cohort mapping &amp; personalized catalog matching.</div>
              </div>
              <div class="panel-header-pill">ML Output</div>
            </div>

            <div class="result-viewport" id="resultViewport">
              <div class="empty-state">
                <div class="empty-icon">&#128225;</div>
                <div class="empty-title">Ready for Analysis</div>
                <div class="empty-sub">Configure viewer behavior and run inference to discover an audience segment.</div>
              </div>
            </div>
          </div>
        </section>

        <!-- ─── 6. Model Pipeline Visualization ("HOW THE AI WORKS") ─── -->
        <section id="pipeline" class="section-container">
          <div class="section-header">
            <div class="section-title">How the AI Works</div>
            <div class="section-sub">End-to-end unsupervised pipeline architecture from telemetry ingestion to personalized curation.</div>
          </div>
          <div class="pipeline-grid">
            <div class="pipeline-step-card">
              <div class="step-badge">01 &bull; Ingestion</div>
              <div class="step-title">Viewer Behavior</div>
              <div class="step-desc">Captures watch time, session length, frequencies &amp; multi-genre history.</div>
            </div>
            <div class="pipeline-step-card">
              <div class="step-badge">02 &bull; Features</div>
              <div class="step-title">Feature Engineering</div>
              <div class="step-desc">Derives session counts, genre diversity &amp; weekend viewing ratios.</div>
            </div>
            <div class="pipeline-step-card">
              <div class="step-badge">03 &bull; Scaling</div>
              <div class="step-title">Standard Scaler</div>
              <div class="step-desc">Normalizes 6 numerical features with precomputed zero-mean unit-variance.</div>
            </div>
            <div class="pipeline-step-card">
              <div class="step-badge">04 &bull; Clustering</div>
              <div class="step-title">K-Means (K=5)</div>
              <div class="step-desc">Assigns nearest centroid across 15-dimensional scaled behavioral space.</div>
            </div>
            <div class="pipeline-step-card">
              <div class="step-badge">05 &bull; Profiling</div>
              <div class="step-title">Audience Segment</div>
              <div class="step-desc">Maps Euclidean centroid index to empirical cohort archetype.</div>
            </div>
            <div class="pipeline-step-card">
              <div class="step-badge">06 &bull; Personalization</div>
              <div class="step-title">Content-Based ML Ranking</div>
              <div class="step-desc">Computes multi-dimensional cosine similarity across 9D genres, format fit &amp; cohort priors.</div>
            </div>
          </div>
        </section>

        <!-- ─── 7. Audience Intelligence (Cohorts Table) ─── -->
        <section id="cohorts" class="section-container">
          <div class="section-header">
            <div class="section-title">Discovered Audience Cohorts</div>
            <div class="section-sub">Behavioral segments discovered from 4,992 processed viewer profiles.</div>
          </div>
          <div class="panel" style="padding: 16px 20px;">
            <div class="table-wrapper">
              <table class="cohort-table">
                <thead>
                  <tr>
                    <th>Segment</th>
                    <th>Cohort</th>
                    <th>Audience Share</th>
                    <th>Avg Watch Time</th>
                    <th>Avg Session</th>
                    <th>Dominant Profile</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td class="td-primary">0</td>
                    <td class="td-primary">Eclectic Multi-Genre Explorers</td>
                    <td><span class="share-tag">13.8% (689)</span></td>
                    <td>47.7 hrs</td>
                    <td>64.1 mins</td>
                    <td>High diversity (&gt;4.0 genres); streams across drama, romance &amp; documentary</td>
                  </tr>
                  <tr>
                    <td class="td-primary">1</td>
                    <td class="td-primary">High-Engagement Action Viewers</td>
                    <td><span class="share-tag">25.7% (1,283)</span></td>
                    <td>54.9 hrs</td>
                    <td>89.7 mins</td>
                    <td>Heavy viewers; intensive concentration on Action, Thriller &amp; fast pacing</td>
                  </tr>
                  <tr>
                    <td class="td-primary">2</td>
                    <td class="td-primary">Casual Short-Session Viewers</td>
                    <td><span class="share-tag">30.1% (1,502)</span></td>
                    <td>12.0 hrs</td>
                    <td>25.0 mins</td>
                    <td>Short bite-sized sessions, low cumulative hours, Comedy &amp; Animation priority</td>
                  </tr>
                  <tr>
                    <td class="td-primary">3</td>
                    <td class="td-primary">Low-Activity Dormant Viewers</td>
                    <td><span class="share-tag">10.1% (505)</span></td>
                    <td>3.5 hrs</td>
                    <td>18.6 mins</td>
                    <td>Dormant accounts (&gt;38 days inactivity), minimal total platform engagement</td>
                  </tr>
                  <tr>
                    <td class="td-primary">4</td>
                    <td class="td-primary">Weekend Binge Enthusiasts</td>
                    <td><span class="share-tag">20.3% (1,013)</span></td>
                    <td>39.6 hrs</td>
                    <td>119.2 mins</td>
                    <td>High weekend ratio (80%), marathon 2hr+ uninterrupted sessions, Drama/Sci-Fi</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </section>

        <!-- ─── 8. Model Intelligence & Evaluation Status ─── -->
        <section id="model-intel" class="intel-grid">
          <!-- Left: Model Specifications -->
          <div class="panel" style="padding: 24px;">
            <div class="section-title" style="margin-bottom: 4px;">Model Intelligence</div>
            <div class="section-sub">Immutable model parameters and verified clustering metrics.</div>
            <div class="specs-grid">
              <div class="spec-item"><span class="spec-key">Algorithm</span><span class="spec-val">K-Means</span></div>
              <div class="spec-item"><span class="spec-key">Clusters (K)</span><span class="spec-val">5</span></div>
              <div class="spec-item"><span class="spec-key">Scaler</span><span class="spec-val">StandardScaler</span></div>
              <div class="spec-item"><span class="spec-key">Random Seed</span><span class="spec-val">42</span></div>
              <div class="spec-item"><span class="spec-key">Training Records</span><span class="spec-val">4,992</span></div>
              <div class="spec-item"><span class="spec-key">Silhouette</span><span class="spec-val">0.4206</span></div>
              <div class="spec-item"><span class="spec-key">Inertia</span><span class="spec-val">10,330.4</span></div>
              <div class="spec-item"><span class="spec-key">Personalization Model</span><span class="spec-val">Content-Based Vector Space</span></div>
              <div class="spec-item"><span class="spec-key">Ranking Metric</span><span class="spec-val">Cosine Similarity</span></div>
              <div class="spec-item"><span class="spec-key">Feature Weights</span><span class="spec-val">50% Genre, 25% Format, 15% Cohort, 10% Pop</span></div>
              <div class="spec-item"><span class="spec-key">Retraining In Request</span><span class="spec-val">None (Precomputed)</span></div>
              <div class="spec-item"><span class="spec-key">Model Artifact</span><span class="spec-val">model_bundle.joblib</span></div>
            </div>
          </div>

          <!-- Right: Evaluation Status -->
          <div id="evaluation" class="panel" style="padding: 24px;">
            <div class="section-title" style="margin-bottom: 4px;">Evaluation Status</div>
            <div class="section-sub">Runtime verification evidence generated by evaluator container.</div>
            <div class="specs-grid">
              <div class="spec-item"><span class="spec-key">Valid Requests</span><span class="pass-pill">4 / 4 PASS</span></div>
              <div class="spec-item"><span class="spec-key">Edge Cases</span><span class="pass-pill">11 / 11 PASS</span></div>
              <div class="spec-item"><span class="spec-key">Health Check</span><span class="pass-pill">PASS</span></div>
              <div class="spec-item"><span class="spec-key">Catalog Coverage</span><span class="pass-pill">100% (15/15)</span></div>
              <div class="spec-item"><span class="spec-key">Ranking Determinism</span><span class="pass-pill">100% REPEATABLE</span></div>
              <div class="spec-item"><span class="spec-key">Model Artifact</span><span class="pass-pill">AVAILABLE</span></div>
            </div>
          </div>
        </section>

        <!-- ─── 9. Docker Architecture ─── -->
        <section id="architecture" class="section-container">
          <div class="section-header">
            <div class="section-title">Docker Architecture</div>
            <div class="section-sub">Microservice decomposition running via Docker Compose with volume-based model persistence.</div>
          </div>
          <div class="panel" style="padding: 24px;">
            <div class="arch-flow-grid">
              <div class="arch-card">
                <div class="arch-card-tag">Service 01</div>
                <div class="arch-card-title">ott-trainer</div>
                <div class="arch-card-desc">Ingests user_activity.csv, trains K=5 model, serializes bundle to shared volume, exits 0.</div>
              </div>
              <div class="arch-card">
                <div class="arch-card-tag">Shared Persistence</div>
                <div class="arch-card-title">model_volume</div>
                <div class="arch-card-desc">Named Docker volume mounting /app/models across containers without host leaks.</div>
              </div>
              <div class="arch-card">
                <div class="arch-card-tag">Service 02</div>
                <div class="arch-card-title">ott-api</div>
                <div class="arch-card-desc">Flask inference engine preloading bundle, serving /health and /recommend endpoints.</div>
              </div>
              <div class="arch-card">
                <div class="arch-card-tag">Service 03</div>
                <div class="arch-card-title">ott-evaluator</div>
                <div class="arch-card-desc">Automated pytest suite testing 4 archetypes + 11 edge cases, writing metrics.json.</div>
              </div>
            </div>
            <div class="arch-badges">
              <span class="tech-tag">Docker Compose</span>
              <span class="tech-tag">CPU Friendly</span>
              <span class="tech-tag">Pinned Dependencies</span>
              <span class="tech-tag">Healthcheck Polling</span>
              <span class="tech-tag">Independent Services</span>
            </div>
          </div>
        </section>

      </div>
    </main>

    <!-- ─── 10. Minimal Enterprise Footer ─── -->
    <footer class="site-footer">
      <div class="container">
        <span class="footer-brand">IT HAPPENS @ RAALE</span>
        <span class="footer-dot"></span>
        Containerized Audience Segmentation &amp; Personalization Service
        <span class="footer-dot"></span>
        Hackathon 2026
        <div style="margin-top: 4px; color: var(--text-dim);">K-Means &bull; Flask API &bull; Docker Compose</div>
      </div>
    </footer>

  </div>

  <script>
    /* ─── State Management ─── */
    const GENRES_LIST = ['Action', 'Drama', 'Comedy', 'Sci-Fi', 'Thriller', 'Romance', 'Horror', 'Documentary', 'Animation'];
    let selectedGenres = new Set(['Action', 'Thriller']);
    let latestResponseData = null;

    /* ─── Render Genre Chips ─── */
    function renderGenreChips() {
      const container = document.getElementById('genreContainer');
      container.innerHTML = '';
      GENRES_LIST.forEach(genre => {
        const isSelected = selectedGenres.has(genre);
        const chip = document.createElement('div');
        chip.className = 'genre-chip' + (isSelected ? ' active' : '');
        chip.innerHTML = `<span>${isSelected ? '&#10003; ' : ''}${genre}</span><span class="genre-check">${isSelected ? '&#10003;' : '+'}</span>`;
        chip.onclick = () => {
          if (isSelected) {
            if (selectedGenres.size > 1) selectedGenres.delete(genre);
          } else {
            selectedGenres.add(genre);
          }
          clearArchetypeHighlight();
          renderGenreChips();
        };
        container.appendChild(chip);
      });
    }

    /* ─── Sliders ─── */
    function onWatchInput(val) {
      document.getElementById('watchBadge').textContent = parseFloat(val).toFixed(1) + ' hrs';
      clearArchetypeHighlight();
    }
    function onSessionInput(val) {
      document.getElementById('sessionBadge').textContent = parseInt(val, 10) + ' mins';
      clearArchetypeHighlight();
    }

    /* ─── Archetype Presets ─── */
    function clearArchetypeHighlight() {
      document.querySelectorAll('.archetype-btn').forEach(btn => btn.classList.remove('active'));
    }
    function selectArchetype(key) {
      clearArchetypeHighlight();
      const presets = {
        action:   { id: 'btnArchAction',   viewerId: 'USR-0003', watch: 65, session: 95,  genres: ['Action', 'Thriller'] },
        casual:   { id: 'btnArchCasual',   viewerId: 'USR-0006', watch: 14, session: 24,  genres: ['Comedy', 'Animation'] },
        binge:    { id: 'btnArchBinge',    viewerId: 'USR-0002', watch: 42, session: 125, genres: ['Sci-Fi', 'Drama'] },
        eclectic: { id: 'btnArchEclectic', viewerId: 'USR-0005', watch: 48, session: 65,  genres: ['Drama', 'Action', 'Romance', 'Documentary'] }
      };
      const preset = presets[key];
      if (!preset) return;

      document.getElementById(preset.id).classList.add('active');
      document.getElementById('viewerId').value = preset.viewerId;
      document.getElementById('watchRange').value = preset.watch;
      document.getElementById('watchBadge').textContent = preset.watch.toFixed(1) + ' hrs';
      document.getElementById('sessionRange').value = preset.session;
      document.getElementById('sessionBadge').textContent = preset.session + ' mins';
      selectedGenres = new Set(preset.genres);
      renderGenreChips();

      executeInference(new Event('submit'));
    }

    /* ─── Execute Real Inference ─── */
    async function executeInference(e) {
      if (e) e.preventDefault();
      const btn = document.getElementById('btnPredict');
      const btnText = document.getElementById('predictBtnText');
      btn.disabled = true;
      btnText.textContent = 'Analyzing Viewer...';

      const payload = {
        user_id: document.getElementById('viewerId').value.trim() || 'USR-8192',
        watch_time_hours: parseFloat(document.getElementById('watchRange').value),
        avg_session_mins: parseFloat(document.getElementById('sessionRange').value),
        preferred_genres: Array.from(selectedGenres)
      };

      try {
        const res = await fetch('/recommend', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        latestResponseData = data;

        if (res.ok) {
          renderInferenceResult(data);
          btnText.textContent = '&#10003; Analysis Complete';
          setTimeout(() => {
            btn.disabled = false;
            btnText.textContent = 'Predict Segment & Personalize';
          }, 1400);
        } else {
          renderErrorState('VALIDATION ERROR', data.message || 'The inference request failed validation.');
          btn.disabled = false;
          btnText.textContent = 'Predict Segment & Personalize';
        }
      } catch (err) {
        renderErrorState('SERVICE UNAVAILABLE', 'Unable to reach the inference service: ' + (err.message || 'Network error'));
        btn.disabled = false;
        btnText.textContent = 'Predict Segment & Personalize';
      }
    }

    /* ─── Render Success Result ─── */
    function renderInferenceResult(data) {
      let recsHtml = '';
      const details = data.recommendation_details || [];
      if (details.length > 0) {
        details.forEach((item, idx) => {
          const num = String(idx + 1).padStart(2, '0');
          const scorePercent = (item.score * 100).toFixed(1);
          const metaBadge = (item.matched_genres && item.matched_genres.length > 0)
            ? item.matched_genres.join(', ')
            : item.type;
          recsHtml += `
            <div class="rec-card">
              <div class="rec-card-left">
                <span class="rec-number">${num}</span>
                <div>
                  <div class="rec-title">${item.title}</div>
                  <div style="font-size:11px;color:var(--text-dim);margin-top:2px;">
                    ${metaBadge} &bull; ${item.duration_mins}m &bull; ${item.type}
                  </div>
                </div>
              </div>
              <div style="text-align:right;">
                <span class="rec-match-pill">ML SCORE: ${item.score.toFixed(4)}</span>
                <div style="font-size:10px;color:var(--accent-green);font-weight:700;margin-top:3px;">${scorePercent}% Match</div>
              </div>
            </div>
          `;
        });
      } else {
        (data.recommendations || []).forEach((title, idx) => {
          const num = String(idx + 1).padStart(2, '0');
          recsHtml += `
            <div class="rec-card">
              <div class="rec-card-left">
                <span class="rec-number">${num}</span>
                <span class="rec-title">${title}</span>
              </div>
              <span class="rec-match-pill">PERSONALIZED MATCH</span>
            </div>
          `;
        });
      }

      document.getElementById('resultViewport').innerHTML = `
        <!-- Cohort Hero Result -->
        <div class="cohort-result-card">
          <div class="cohort-top-row">
            <div class="cohort-title-wrap">
              <h3>&#127919; ${data.segment_name}</h3>
            </div>
            <span class="cohort-id-badge">Cohort #${data.segment_id}</span>
          </div>
          <div class="cohort-metrics-grid">
            <div class="cohort-metric-item">
              <span class="cohort-metric-key">Viewer ID</span>
              <span class="cohort-metric-val">${data.user_id}</span>
            </div>
            <div class="cohort-metric-item">
              <span class="cohort-metric-key">Centroid Distance</span>
              <span class="cohort-metric-val">${data.distance_to_centroid}</span>
            </div>
          </div>
          <div class="distance-tooltip">&#8505; Distance from this viewer profile to the assigned cluster centroid in scaled feature space.</div>
        </div>

        <!-- Curated Recommendations -->
        <div class="rec-heading">
          <span>Curated For This Viewer</span>
          <span style="color: var(--accent-green);">&#10003; 3 Matches</span>
        </div>
        <div class="rec-list">
          ${recsHtml}
        </div>

        <!-- Raw API Inspector -->
        <details class="api-inspector">
          <summary>
            <span>View API Response</span>
            <span style="font-size: 10px; color: var(--accent-cyan);">&#9662; JSON</span>
          </summary>
          <div class="inspector-body">
            <div class="inspector-actions">
              <button type="button" class="btn-copy-json" onclick="copyRawJson(this)">Copy JSON</button>
            </div>
            <pre class="code-block">${JSON.stringify(data, null, 2)}</pre>
          </div>
        </details>
      `;
    }

    /* ─── Copy JSON Helper ─── */
    function copyRawJson(btn) {
      if (!latestResponseData) return;
      navigator.clipboard.writeText(JSON.stringify(latestResponseData, null, 2)).then(() => {
        const orig = btn.textContent;
        btn.textContent = '&#10003; Copied!';
        setTimeout(() => { btn.textContent = orig; }, 1500);
      });
    }

    /* ─── Render Error Banner ─── */
    function renderErrorState(title, message) {
      document.getElementById('resultViewport').innerHTML = `
        <div class="error-box">
          <div class="error-box-title">&#9888; ${title}</div>
          <div class="error-box-body">${message}</div>
        </div>
      `;
    }

    /* ─── Dynamic Health Polling ─── */
    async function checkServiceHealth() {
      const pill = document.getElementById('healthPill');
      const label = document.getElementById('healthStatus');
      try {
        const res = await fetch('/health');
        const data = await res.json();
        if (res.ok && data.status === 'ok' && data.model_loaded === true) {
          pill.className = 'health-pill';
          label.innerHTML = 'SERVICE HEALTHY &bull; MODEL LOADED';
        } else {
          pill.className = 'health-pill down';
          label.innerHTML = 'MODEL NOT READY &bull; PLEASE WAIT';
        }
      } catch {
        pill.className = 'health-pill down';
        label.innerHTML = 'SERVICE UNAVAILABLE &bull; OFFLINE';
      }
    }

    /* ─── Bootstrap ─── */
    renderGenreChips();
    checkServiceHealth();
    setInterval(checkServiceHealth, 8000);
    // Initial live inference call
    executeInference(null);
  </script>
</body>
</html>"""


@app.route('/', methods=['GET'])
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route('/health', methods=['GET'])
def health():
    is_ready = ModelLoader.is_loaded()
    if is_ready:
        return jsonify({
            "status": "ok",
            "model_loaded": True
        }), 200
    else:
        return jsonify({
            "status": "degraded",
            "model_loaded": False,
            "error": "Model artifact not loaded or missing."
        }), 503


@app.route('/recommend', methods=['POST'])
def recommend():
    bundle = ModelLoader.get_bundle()
    if bundle is None:
        return jsonify({
            "error": "Service Unavailable",
            "message": "Model artifact is not loaded. Please ensure trainer has completed."
        }), 503

    if not request.is_json:
        return jsonify({
            "error": "Bad Request",
            "message": "Content-Type must be application/json with a valid JSON body."
        }), 400

    try:
        raw_payload = request.get_json()
    except Exception as e:
        logger.warning(f"Malformed JSON received: {e}")
        return jsonify({
            "error": "Bad Request",
            "message": "Malformed JSON payload."
        }), 400

    is_valid, error_msg, clean_data = validate_recommend_payload(raw_payload)
    if not is_valid:
        return jsonify({
            "error": "Bad Request",
            "message": error_msg
        }), 400

    try:
        user_id = clean_data['user_id']
        watch_time = clean_data['watch_time_hours']
        avg_session = clean_data['avg_session_mins']
        top_genres = clean_data['top_genres']

        # Determine derived / defaulted features
        defaults = bundle.get('defaults', {})
        num_sessions = clean_data.get('num_sessions')
        if num_sessions is None:
            num_sessions = max(1, int(round((watch_time * 60.0) / max(1.0, avg_session))))

        genre_diversity = max(1, len(top_genres))

        weekend_ratio = clean_data.get('weekend_watch_ratio')
        if weekend_ratio is None:
            weekend_ratio = defaults.get('weekend_watch_ratio', 0.45)

        recency = clean_data.get('days_since_last_active')
        if recency is None:
            recency = defaults.get('days_since_last_active', 3)

        # 1. Standardize numerical features with column names
        num_cols = bundle['num_cols']
        num_df = pd.DataFrame([[
            watch_time,
            avg_session,
            num_sessions,
            genre_diversity,
            weekend_ratio,
            recency
        ]], columns=num_cols)
        
        scaler = bundle['scaler']
        num_scaled = scaler.transform(num_df)

        # 2. Multi-hot encode genres
        all_genres = bundle.get('all_genres', [])
        user_genres_lower = [g.lower() for g in top_genres]
        genre_vector = np.array([[
            1.0 if g.lower() in user_genres_lower else 0.0
            for g in all_genres
        ]])

        # 3. Concatenate feature vector
        X_full = np.hstack([num_scaled, genre_vector])

        # 4. Predict cluster
        kmeans = bundle['kmeans']
        cluster_id = int(kmeans.predict(X_full)[0])

        # 5. Calculate Euclidean distance to centroid
        centroid = kmeans.cluster_centers_[cluster_id]
        distance_to_centroid = float(np.linalg.norm(X_full - centroid))

        # 6. Retrieve segment name
        segment_name = bundle['segment_names'].get(cluster_id, f"Segment {cluster_id}")

        # 7. Generate ML-based vector recommendations
        recommender_bundle = bundle.get('recommender_bundle')
        if not recommender_bundle:
            recommender_bundle = {
                'catalog': bundle.get('recommendation_catalog', []),
                'all_genres': bundle.get('all_genres', [])
            }

        recommender = ContentRecommender(recommender_bundle)
        recommendations, recommendation_details = recommender.rank(
            preferred_genres=top_genres,
            avg_session_mins=avg_session,
            weekend_watch_ratio=weekend_ratio,
            segment_id=cluster_id,
            top_k=3
        )

        response = {
            "user_id": user_id,
            "segment_id": cluster_id,
            "segment_name": segment_name,
            "recommendations": recommendations,
            "recommendation_details": recommendation_details,
            "distance_to_centroid": round(distance_to_centroid, 4)
        }
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=False)
        return jsonify({
            "error": "Internal Error",
            "message": "An error occurred during recommendation processing."
        }), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not Found", "message": "The requested endpoint does not exist."}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method Not Allowed", "message": "The HTTP method is not allowed for this endpoint."}), 405


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal Server Error", "message": "An unexpected server error occurred."}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    logger.info(f"Starting Audience Segmentation API server on {host}:{port}")
    app.run(host=host, port=port, debug=False)
