"""
Standalone test for Market Intelligence Dashboard
"""

import streamlit as st
import sys
from pathlib import Path

# Add project paths
sys.path.append(str(Path(__file__).parent))

def main():
    st.set_page_config(
        page_title="Market Intelligence Test",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 Market Intelligence Dashboard Test")
    
    try:
        from src.market_intelligence import get_market_intelligence_dashboard
        
        st.success("✅ Market Intelligence module imported successfully")
        
        # Initialize dashboard
        dashboard = get_market_intelligence_dashboard()
        st.success("✅ Market Intelligence dashboard initialized")
        
        # Test rendering
        st.header("Testing Dashboard Rendering")
        
        # Try to render the main tab
        try:
            dashboard.render_market_intelligence_tab()
            st.success("✅ Market Intelligence tab rendered successfully")
        except Exception as e:
            st.error(f"❌ Error rendering Market Intelligence tab: {e}")
            st.exception(e)
            
    except ImportError as e:
        st.error(f"❌ Import error: {e}")
        st.info("💡 Make sure all dependencies are installed")
        
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        st.exception(e)

if __name__ == "__main__":
    main()