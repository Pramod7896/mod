"""CSS loading helpers."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.config import ASSETS_DIR


def load_css() -> None:
    """Inject the app stylesheet into Streamlit."""

    css_path = ASSETS_DIR / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
