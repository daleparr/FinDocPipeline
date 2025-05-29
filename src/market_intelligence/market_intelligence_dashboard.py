"""
Market Intelligence Dashboard Integration for BoE Mosaic Lens
Provides G-SIB monitoring, correlation analysis, and systemic risk visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import yaml
from pathlib import Path

class MarketIntelligenceDashboard:
    """
    Market Intelligence Dashboard for G-SIB monitoring and systemic risk analysis
    """
    
    def __init__(self):
        """Initialize market intelligence dashboard"""
        self.logger = logging.getLogger(__name__)
        
        # Initialize session state
        if 'market_data' not in st.session_state:
            st.session_state.market_data = {}
        if 'correlation_analysis' not in st.session_state:
            st.session_state.correlation_analysis = {}
        if 'systemic_risk_analysis' not in st.session_state:
            st.session_state.systemic_risk_analysis = {}
        if 'market_alerts' not in st.session_state:
            st.session_state.market_alerts = []
    
    def render_market_intelligence_tab(self):
        """Render the complete market intelligence tab"""
        st.header("📊 Market Intelligence & G-SIB Monitoring")
        
        # Control panel
        self._render_control_panel()
        
        # Main content based on selected view
        view_option = st.session_state.get('market_view', 'overview')
        
        if view_option == 'overview':
            self._render_overview_dashboard()
        elif view_option == 'gsib_monitoring':
            self._render_gsib_monitoring_panel()
        elif view_option == 'correlation_analysis':
            self._render_correlation_analysis_panel()
        elif view_option == 'systemic_risk':
            self._render_systemic_risk_panel()
        elif view_option == 'earnings_impact':
            self._render_earnings_impact_panel()
        elif view_option == 'alerts':
            self._render_alerts_panel()
    
    def _render_control_panel(self):
        """Render control panel for market intelligence settings"""
        with st.expander("🎛️ Market Intelligence Controls", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                view_option = st.selectbox(
                    "Dashboard View",
                    ["overview", "gsib_monitoring", "correlation_analysis", "systemic_risk", "earnings_impact", "alerts"],
                    format_func=lambda x: {
                        "overview": "📈 Overview",
                        "gsib_monitoring": "🏛️ G-SIB Monitoring",
                        "correlation_analysis": "🔗 Correlation Analysis",
                        "systemic_risk": "⚠️ Systemic Risk",
                        "earnings_impact": "📊 Earnings Impact",
                        "alerts": "🚨 Alerts"
                    }[x]
                )
                st.session_state.market_view = view_option
            
            with col2:
                analysis_period = st.selectbox(
                    "Analysis Period",
                    ["1mo", "3mo", "6mo", "1y"],
                    index=1,
                    help="Time period for market data analysis"
                )
                st.session_state.analysis_period = analysis_period
            
            with col3:
                region_filter = st.selectbox(
                    "Region Filter",
                    ["all", "us_banks", "european_banks", "uk_banks"],
                    format_func=lambda x: {
                        "all": "🌍 All Regions",
                        "us_banks": "🇺🇸 US Banks",
                        "european_banks": "🇪🇺 European Banks",
                        "uk_banks": "🇬🇧 UK Banks"
                    }[x]
                )
                st.session_state.region_filter = region_filter
            
            with col4:
                if st.button("🔄 Refresh Data", type="primary"):
                    self._refresh_market_data()
    
    def _refresh_market_data(self):
        """Refresh market data and analysis"""
        with st.spinner("Refreshing market intelligence data..."):
            try:
                from .gsib_monitor import get_gsib_monitor
                from .sentiment_market_correlator import get_sentiment_market_correlator
                
                # Initialize components
                monitor = get_gsib_monitor()
                correlator = get_sentiment_market_correlator()
                
                # Get analysis parameters
                period = st.session_state.get('analysis_period', '3mo')
                region = st.session_state.get('region_filter', 'all')
                
                # Track G-SIB movements
                market_data = monitor.track_global_gsib_movements(period=period)
                st.session_state.market_data = market_data
                
                # Perform correlation analysis
                if market_data:
                    correlation_analysis = monitor.detect_cross_market_correlations(market_data)
                    st.session_state.correlation_analysis = correlation_analysis
                    
                    # Calculate systemic risk
                    systemic_risk = monitor.calculate_systemic_risk_score(market_data)
                    st.session_state.systemic_risk_analysis = systemic_risk
                    
                    # Generate alerts
                    alerts = monitor.generate_contagion_alerts(correlation_analysis)
                    st.session_state.market_alerts = alerts
                
                st.success("✅ Market intelligence data refreshed successfully")
                
            except Exception as e:
                st.error(f"❌ Error refreshing data: {str(e)}")
                self.logger.error(f"Error refreshing market data: {e}")
    
    def _render_overview_dashboard(self):
        """Render overview dashboard with key metrics"""
        st.subheader("📈 Market Intelligence Overview")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        market_data = st.session_state.get('market_data', {})
        systemic_risk = st.session_state.get('systemic_risk_analysis', {})
        alerts = st.session_state.get('market_alerts', [])
        
        with col1:
            institutions_tracked = len(market_data)
            st.metric("G-SIBs Tracked", institutions_tracked)
        
        with col2:
            risk_level = systemic_risk.get('risk_level', 'UNKNOWN')
            risk_score = systemic_risk.get('systemic_risk_score', 0)
            color = "normal" if risk_level in ['LOW', 'MEDIUM'] else "inverse"
            st.metric("Systemic Risk", risk_level, f"{risk_score:.3f}", delta_color=color)
        
        with col3:
            critical_alerts = len([a for a in alerts if a.get('severity') == 'CRITICAL'])
            high_alerts = len([a for a in alerts if a.get('severity') == 'HIGH'])
            total_alerts = len(alerts)
            st.metric("Active Alerts", total_alerts, f"{critical_alerts} Critical, {high_alerts} High")
        
        with col4:
            correlation_analysis = st.session_state.get('correlation_analysis', {})
            avg_correlation = correlation_analysis.get('summary_stats', {}).get('mean_correlation', 0)
            st.metric("Avg Correlation", f"{avg_correlation:.3f}")
        
        # Charts row
        if market_data:
            col1, col2 = st.columns(2)
            
            with col1:
                self._render_gsib_performance_chart(market_data)
            
            with col2:
                self._render_correlation_heatmap_summary()
        
        # Recent alerts
        if alerts:
            st.subheader("🚨 Recent Alerts")
            self._render_alerts_summary(alerts[:5])  # Show top 5 alerts
    
    def _render_gsib_monitoring_panel(self):
        """Render G-SIB monitoring panel"""
        st.subheader("🏛️ G-SIB Institution Monitoring")
        
        market_data = st.session_state.get('market_data', {})
        
        if not market_data:
            st.warning("No market data available. Please refresh data first.")
            return
        
        # Institution selector
        selected_institutions = st.multiselect(
            "Select Institutions to Monitor",
            list(market_data.keys()),
            default=list(market_data.keys())[:5] if len(market_data) > 5 else list(market_data.keys())
        )
        
        if not selected_institutions:
            st.warning("Please select at least one institution to monitor.")
            return
        
        # Performance comparison
        self._render_performance_comparison(market_data, selected_institutions)
        
        # Individual institution details
        st.subheader("📊 Individual Institution Analysis")
        
        selected_institution = st.selectbox(
            "Select Institution for Detailed Analysis",
            selected_institutions
        )
        
        if selected_institution and selected_institution in market_data:
            self._render_individual_institution_analysis(market_data[selected_institution], selected_institution)
    
    def _render_correlation_analysis_panel(self):
        """Render correlation analysis panel"""
        st.subheader("🔗 Cross-Market Correlation Analysis")
        
        correlation_analysis = st.session_state.get('correlation_analysis', {})
        
        if not correlation_analysis:
            st.warning("No correlation analysis available. Please refresh data first.")
            return
        
        # Correlation matrix heatmap
        if 'correlation_matrix' in correlation_analysis:
            st.subheader("📊 Correlation Matrix")
            self._render_correlation_heatmap(correlation_analysis['correlation_matrix'])
        
        # Summary statistics
        if 'summary_stats' in correlation_analysis:
            st.subheader("📈 Correlation Statistics")
            self._render_correlation_statistics(correlation_analysis['summary_stats'])
    
    def _render_systemic_risk_panel(self):
        """Render systemic risk analysis panel"""
        st.subheader("⚠️ Systemic Risk Analysis")
        
        systemic_risk = st.session_state.get('systemic_risk_analysis', {})
        
        if not systemic_risk:
            st.warning("No systemic risk analysis available. Please refresh data first.")
            return
        
        # Risk score overview
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            risk_score = systemic_risk.get('systemic_risk_score', 0)
            risk_level = systemic_risk.get('risk_level', 'UNKNOWN')
            
            # Risk gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = risk_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Systemic Risk Score"},
                delta = {'reference': 0.5},
                gauge = {
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.4], 'color': "lightgray"},
                        {'range': [0.4, 0.6], 'color': "yellow"},
                        {'range': [0.6, 0.8], 'color': "orange"},
                        {'range': [0.8, 1], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.8
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric("Risk Level", risk_level)
            st.metric("Risk Score", f"{risk_score:.3f}")
        
        with col3:
            component_scores = systemic_risk.get('component_scores', {})
            st.write("**Component Scores:**")
            for component, score in component_scores.items():
                st.write(f"• {component.replace('_', ' ').title()}: {score:.3f}")
    
    def _render_earnings_impact_panel(self):
        """Render earnings impact analysis panel"""
        st.subheader("📊 Earnings Impact Analysis")
        
        market_data = st.session_state.get('market_data', {})
        
        if not market_data:
            st.warning("No market data available. Please refresh data first.")
            return
        
        # Auto-detect institutions from ETL runs
        etl_institutions = self._detect_institutions_from_etl()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Show auto-detected institutions if available
            if etl_institutions:
                st.info(f"🔍 Auto-detected institutions from ETL: {', '.join(etl_institutions[:3])}{'...' if len(etl_institutions) > 3 else ''}")
                
                # Prioritize ETL institutions in selection
                available_tickers = list(market_data.keys())
                prioritized_tickers = []
                
                # Add ETL institutions first (with their ticker mappings)
                for institution in etl_institutions:
                    ticker = self._map_institution_to_ticker(institution)
                    if ticker and ticker in available_tickers:
                        prioritized_tickers.append(f"{ticker} ({institution})")
                        available_tickers.remove(ticker)
                
                # Add remaining tickers
                prioritized_tickers.extend(available_tickers)
                
                selected_option = st.selectbox(
                    "Select Institution",
                    prioritized_tickers,
                    help="🎯 Institutions from ETL runs are prioritized at the top"
                )
                
                # Extract ticker from selection
                selected_ticker = selected_option.split(' (')[0] if ' (' in selected_option else selected_option
            else:
                selected_ticker = st.selectbox(
                    "Select Institution",
                    list(market_data.keys())
                )
        
        with col2:
            # Auto-detect earnings date from ETL data if available
            auto_earnings_date = self._detect_earnings_date_from_etl(selected_ticker if 'selected_ticker' in locals() else None)
            
            if auto_earnings_date:
                st.info(f"🗓️ Auto-detected earnings period: {auto_earnings_date}")
                default_date = auto_earnings_date
            else:
                default_date = pd.Timestamp.now() - pd.Timedelta(days=30)
            
            earnings_date = st.date_input(
                "Earnings Date",
                value=default_date,
                help="📅 Auto-detected from ETL data when available"
            )
        
        if selected_ticker and earnings_date:
            # Perform actual earnings impact analysis
            self._perform_earnings_impact_analysis(selected_ticker, earnings_date, market_data)
    
    def _perform_earnings_impact_analysis(self, ticker: str, earnings_date, market_data: Dict):
        """Perform and display earnings impact analysis"""
        try:
            from .sentiment_market_correlator import get_sentiment_market_correlator
            from datetime import datetime, timedelta
            import pandas as pd
            
            # Convert earnings_date to datetime if needed
            if hasattr(earnings_date, 'date'):
                earnings_date = datetime.combine(earnings_date, datetime.min.time())
            elif isinstance(earnings_date, str):
                earnings_date = datetime.strptime(earnings_date, '%Y-%m-%d')
            
            correlator = get_sentiment_market_correlator()
            
            # Get market data for the selected ticker
            ticker_data = market_data.get(ticker)
            if ticker_data is None or ticker_data.empty:
                st.warning(f"No market data available for {ticker}")
                return
            
            # Create analysis windows - fix timestamp arithmetic
            pre_days = 5
            post_days = 3
            
            # Convert earnings_date to pandas Timestamp for proper arithmetic
            if isinstance(earnings_date, datetime):
                earnings_ts = pd.Timestamp(earnings_date)
            else:
                earnings_ts = pd.Timestamp(earnings_date)
            
            pre_start = earnings_ts - pd.Timedelta(days=pre_days)
            post_end = earnings_ts + pd.Timedelta(days=post_days)
            
            # Filter market data around earnings date
            ticker_data_indexed = ticker_data.copy()
            if not ticker_data_indexed.index.name == 'Date':
                ticker_data_indexed = ticker_data_indexed.reset_index()
                if 'Date' in ticker_data_indexed.columns:
                    ticker_data_indexed['Date'] = pd.to_datetime(ticker_data_indexed['Date'])
                    ticker_data_indexed = ticker_data_indexed.set_index('Date')
            
            # Ensure datetime compatibility for comparison
            if hasattr(ticker_data_indexed.index, 'tz') and ticker_data_indexed.index.tz is not None:
                # Market data has timezone, convert our dates to match
                import pytz
                market_tz = ticker_data_indexed.index.tz
                pre_start = pd.Timestamp(pre_start).tz_localize(market_tz)
                post_end = pd.Timestamp(post_end).tz_localize(market_tz)
                earnings_date = pd.Timestamp(earnings_date).tz_localize(market_tz)
            else:
                # Ensure market data index is timezone-naive
                if hasattr(ticker_data_indexed.index, 'tz_localize'):
                    ticker_data_indexed.index = ticker_data_indexed.index.tz_localize(None)
            
            # Filter data around earnings period
            try:
                earnings_window = ticker_data_indexed[
                    (ticker_data_indexed.index >= pre_start) &
                    (ticker_data_indexed.index <= post_end)
                ]
            except Exception as e:
                self.logger.error(f"Date filtering error: {e}")
                # Fallback: convert everything to string dates for comparison
                ticker_dates = pd.to_datetime(ticker_data_indexed.index).date
                pre_start_date = pre_start.date() if hasattr(pre_start, 'date') else pre_start
                post_end_date = post_end.date() if hasattr(post_end, 'date') else post_end
                
                mask = (ticker_dates >= pre_start_date) & (ticker_dates <= post_end_date)
                earnings_window = ticker_data_indexed.iloc[mask]
            
            if earnings_window.empty:
                st.warning(f"No market data available around earnings date {earnings_date.date()}")
                return
            
            # Display analysis
            st.subheader(f"📊 Earnings Impact Analysis for {ticker}")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Earnings Date", earnings_date.strftime("%Y-%m-%d"))
            
            with col2:
                if 'daily_return' in earnings_window.columns:
                    avg_return = earnings_window['daily_return'].mean()
                    st.metric("Avg Return", f"{avg_return:.4f}")
                else:
                    st.metric("Avg Return", "N/A")
            
            with col3:
                if 'volatility' in earnings_window.columns:
                    avg_vol = earnings_window['volatility'].mean()
                    st.metric("Avg Volatility", f"{avg_vol:.4f}")
                else:
                    st.metric("Avg Volatility", "N/A")
            
            with col4:
                data_points = len(earnings_window)
                st.metric("Data Points", data_points)
            
            # Pre/Post earnings comparison
            st.subheader("📈 Pre vs Post Earnings Analysis")
            
            # Use the same earnings timestamp for consistent comparison
            earnings_ts_for_comparison = pd.Timestamp(earnings_date)
            if hasattr(earnings_window.index, 'tz') and earnings_window.index.tz is not None:
                # Match timezone if market data has timezone
                if earnings_ts_for_comparison.tz is None:
                    earnings_ts_for_comparison = earnings_ts_for_comparison.tz_localize(earnings_window.index.tz)
                else:
                    earnings_ts_for_comparison = earnings_ts_for_comparison.tz_convert(earnings_window.index.tz)
            
            pre_data = earnings_window[earnings_window.index < earnings_ts_for_comparison]
            post_data = earnings_window[earnings_window.index > earnings_ts_for_comparison]
            
            if not pre_data.empty and not post_data.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Pre-Earnings (5 days)**")
                    if 'daily_return' in pre_data.columns:
                        pre_return = pre_data['daily_return'].mean()
                        st.metric("Avg Return", f"{pre_return:.4f}")
                    if 'volatility' in pre_data.columns:
                        pre_vol = pre_data['volatility'].mean()
                        st.metric("Avg Volatility", f"{pre_vol:.4f}")
                
                with col2:
                    st.write("**Post-Earnings (3 days)**")
                    if 'daily_return' in post_data.columns:
                        post_return = post_data['daily_return'].mean()
                        st.metric("Avg Return", f"{post_return:.4f}")
                        
                        # Calculate impact
                        if 'daily_return' in pre_data.columns:
                            impact = post_return - pre_return
                            st.metric("Return Impact", f"{impact:.4f}", delta=f"{impact:.4f}")
                    
                    if 'volatility' in post_data.columns:
                        post_vol = post_data['volatility'].mean()
                        st.metric("Avg Volatility", f"{post_vol:.4f}")
            
            # Chart of price movement around earnings
            if 'Close' in earnings_window.columns:
                st.subheader("📊 Price Movement Around Earnings")
                
                import plotly.graph_objects as go
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=earnings_window.index,
                    y=earnings_window['Close'],
                    mode='lines+markers',
                    name='Close Price',
                    line=dict(color='blue')
                ))
                
                # Add earnings date line - ensure proper timestamp format
                earnings_date_for_chart = pd.Timestamp(earnings_date)
                if hasattr(earnings_window.index, 'tz') and earnings_window.index.tz is not None:
                    # Match timezone if market data has timezone
                    if earnings_date_for_chart.tz is None:
                        # Localize if naive timestamp
                        earnings_date_for_chart = earnings_date_for_chart.tz_localize(earnings_window.index.tz)
                    else:
                        # Convert if already timezone-aware
                        earnings_date_for_chart = earnings_date_for_chart.tz_convert(earnings_window.index.tz)
                
                fig.add_vline(
                    x=earnings_date_for_chart,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Earnings Date"
                )
                
                fig.update_layout(
                    title=f"{ticker} Price Movement Around Earnings",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # NLP Integration placeholder
            st.subheader("🔗 NLP Sentiment Integration")
            st.info("💡 In production, this would show sentiment analysis from earnings call transcripts and correlate with market movements.")
            
            # Load NLP data if available
            nlp_signals = correlator.fetch_nlp_signals(ticker, "Q1 2025")
            if not nlp_signals.empty:
                st.write("**Available NLP Signals:**")
                st.dataframe(nlp_signals.head())
            else:
                st.write("No NLP sentiment data available for this period.")
                
        except Exception as e:
            st.error(f"Error performing earnings impact analysis: {e}")
            self.logger.error(f"Earnings impact analysis error: {e}")
    
    def _detect_institutions_from_etl(self) -> List[str]:
        """Auto-detect institutions from ETL data"""
        try:
            import os
            from pathlib import Path
            
            institutions = []
            data_path = Path("data/metadata/versions")
            
            if data_path.exists():
                # Scan for institution directories
                for item in data_path.iterdir():
                    if item.is_dir() and item.name not in ['UnknownBank', 'RawDataBank']:
                        institutions.append(item.name)
                
                self.logger.info(f"Auto-detected {len(institutions)} institutions from ETL: {institutions}")
            
            return institutions
            
        except Exception as e:
            self.logger.error(f"Error detecting institutions from ETL: {e}")
            return []
    
    def _map_institution_to_ticker(self, institution: str) -> str:
        """Map institution name to stock ticker"""
        # Common institution to ticker mappings
        institution_ticker_map = {
            'Citigroup': 'C',
            'JPMorgan Chase': 'JPM',
            'Bank of America': 'BAC',
            'Wells Fargo': 'WFC',
            'Goldman Sachs': 'GS',
            'Morgan Stanley': 'MS',
            'HSBC': 'HSBC',
            'Barclays': 'BCS',
            'Deutsche Bank': 'DB',
            'Credit Suisse': 'CS',
            'UBS': 'UBS',
            'BNP Paribas': 'BNP.PA',
            'Santander': 'SAN',
            'ING': 'INGA.AS',
            'Nordea': 'NDA-SE.ST',
            'Unicredit': 'UCG.MI',
            'Intesa Sanpaolo': 'ISP.MI',
            'Societe Generale': 'GLE.PA',
            'Standard Chartered': 'STAN.L',
            'Lloyds': 'LLOY.L',
            'NatWest': 'NWG.L',
            'Royal Bank of Canada': 'RY',
            'Toronto-Dominion Bank': 'TD',
            'Bank of Nova Scotia': 'BNS',
            'Bank of Montreal': 'BMO',
            'Canadian Imperial Bank': 'CM',
            'Mitsubishi UFJ': '8306.T',
            'Sumitomo Mitsui': '8316.T',
            'Mizuho': '8411.T',
            'Bank of China': '3988.HK',
            'ICBC': '1398.HK',
            'China Construction Bank': '0939.HK',
            'Agricultural Bank of China': '1288.HK'
        }
        
        # Try exact match first
        if institution in institution_ticker_map:
            return institution_ticker_map[institution]
        
        # Try partial matches
        for inst_name, ticker in institution_ticker_map.items():
            if institution.lower() in inst_name.lower() or inst_name.lower() in institution.lower():
                return ticker
        
        return None
    
    def _detect_earnings_date_from_etl(self, ticker: str) -> datetime:
        """Auto-detect earnings date from ETL metadata"""
        try:
            if not ticker:
                return None
            
            # Map ticker back to institution name
            institution = self._map_ticker_to_institution(ticker)
            if not institution:
                return None
            
            # Look for quarter information in ETL data
            data_path = Path(f"data/metadata/versions/{institution}")
            if data_path.exists():
                for quarter_dir in data_path.iterdir():
                    if quarter_dir.is_dir():
                        quarter_name = quarter_dir.name
                        
                        # Parse quarter to estimate earnings date
                        if "Q1" in quarter_name:
                            year = quarter_name.split()[-1] if len(quarter_name.split()) > 1 else "2025"
                            return datetime(int(year), 4, 15)  # Mid-April for Q1 earnings
                        elif "Q2" in quarter_name:
                            year = quarter_name.split()[-1] if len(quarter_name.split()) > 1 else "2025"
                            return datetime(int(year), 7, 15)  # Mid-July for Q2 earnings
                        elif "Q3" in quarter_name:
                            year = quarter_name.split()[-1] if len(quarter_name.split()) > 1 else "2025"
                            return datetime(int(year), 10, 15)  # Mid-October for Q3 earnings
                        elif "Q4" in quarter_name:
                            year = quarter_name.split()[-1] if len(quarter_name.split()) > 1 else "2025"
                            return datetime(int(year), 1, 15)  # Mid-January for Q4 earnings
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting earnings date from ETL: {e}")
            return None
    
    def _map_ticker_to_institution(self, ticker: str) -> str:
        """Map stock ticker to institution name"""
        ticker_institution_map = {
            'C': 'Citigroup',
            'JPM': 'JPMorgan Chase',
            'BAC': 'Bank of America',
            'WFC': 'Wells Fargo',
            'GS': 'Goldman Sachs',
            'MS': 'Morgan Stanley',
            'HSBC': 'HSBC',
            'BCS': 'Barclays',
            'DB': 'Deutsche Bank',
            'CS': 'Credit Suisse',
            'UBS': 'UBS'
        }
        
        return ticker_institution_map.get(ticker.split(' (')[0] if ' (' in ticker else ticker)
    
    def _render_alerts_panel(self):
        """Render alerts management panel"""
        st.subheader("🚨 Market Intelligence Alerts")
        
        alerts = st.session_state.get('market_alerts', [])
        
        if not alerts:
            st.info("No active alerts. Market conditions appear normal.")
            return
        
        # Alert summary
        col1, col2, col3, col4 = st.columns(4)
        
        critical_alerts = [a for a in alerts if a.get('severity') == 'CRITICAL']
        high_alerts = [a for a in alerts if a.get('severity') == 'HIGH']
        medium_alerts = [a for a in alerts if a.get('severity') == 'MEDIUM']
        low_alerts = [a for a in alerts if a.get('severity') == 'LOW']
        
        with col1:
            st.metric("🔴 Critical", len(critical_alerts))
        with col2:
            st.metric("🟠 High", len(high_alerts))
        with col3:
            st.metric("🟡 Medium", len(medium_alerts))
        with col4:
            st.metric("🟢 Low", len(low_alerts))
        
        # Display alerts
        for i, alert in enumerate(alerts):
            self._render_alert_card(alert, i)
    
    def _render_alert_card(self, alert: Dict[str, Any], index: int):
        """Render individual alert card"""
        severity = alert.get('severity', 'UNKNOWN')
        alert_type = alert.get('alert_type', 'UNKNOWN')
        message = alert.get('message', 'No message')
        timestamp = alert.get('timestamp', datetime.now())
        
        # Color coding by severity
        color_map = {
            'CRITICAL': '#dc3545',
            'HIGH': '#fd7e14',
            'MEDIUM': '#ffc107',
            'LOW': '#28a745'
        }
        
        color = color_map.get(severity, '#6c757d')
        
        with st.container():
            st.markdown(f"""
            <div style="border-left: 4px solid {color}; padding: 10px; margin: 10px 0; background-color: #f8f9fa;">
                <h4 style="color: {color}; margin: 0;">{severity} - {alert_type.replace('_', ' ').title()}</h4>
                <p style="margin: 5px 0;">{message}</p>
                <small style="color: #6c757d;">Generated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}</small>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_gsib_performance_chart(self, market_data: Dict[str, pd.DataFrame]):
        """Render G-SIB performance comparison chart"""
        st.subheader("📈 G-SIB Performance Comparison")
        
        # Extract recent performance data
        performance_data = []
        
        for ticker, df in market_data.items():
            if not df.empty and 'daily_return' in df.columns:
                recent_return = df['daily_return'].iloc[-5:].sum()  # 5-day return
                volatility = df['daily_return'].std() if len(df) > 1 else 0
                
                performance_data.append({
                    'Institution': ticker,
                    '5-Day Return': recent_return,
                    'Volatility': volatility
                })
        
        if performance_data:
            perf_df = pd.DataFrame(performance_data)
            
            fig = px.scatter(
                perf_df, 
                x='Volatility', 
                y='5-Day Return',
                text='Institution',
                title="Risk-Return Profile",
                labels={'Volatility': 'Volatility (Risk)', '5-Day Return': '5-Day Return (%)'}
            )
            
            fig.update_traces(textposition="top center")
            fig.update_layout(height=400)
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_correlation_heatmap_summary(self):
        """Render correlation heatmap summary"""
        correlation_analysis = st.session_state.get('correlation_analysis', {})
        
        if 'correlation_matrix' in correlation_analysis:
            st.subheader("🔗 Correlation Heatmap")
            self._render_correlation_heatmap(correlation_analysis['correlation_matrix'])
    
    def _render_correlation_heatmap(self, correlation_matrix: pd.DataFrame):
        """Render correlation heatmap"""
        if correlation_matrix.empty:
            st.warning("No correlation data available.")
            return
        
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="G-SIB Cross-Correlation Matrix"
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_alerts_summary(self, alerts: List[Dict[str, Any]]):
        """Render summary of recent alerts"""
        for alert in alerts:
            severity = alert.get('severity', 'UNKNOWN')
            message = alert.get('message', 'No message')
            
            if severity == 'CRITICAL':
                st.error(f"🔴 {message}")
            elif severity == 'HIGH':
                st.warning(f"🟠 {message}")
            else:
                st.info(f"🟡 {message}")
    
    def _render_performance_comparison(self, market_data: Dict[str, pd.DataFrame], selected_institutions: List[str]):
        """Render performance comparison for selected institutions"""
        st.subheader("📊 Performance Comparison")
        
        # Create comparison data
        comparison_data = []
        
        for ticker in selected_institutions:
            if ticker in market_data and not market_data[ticker].empty:
                df = market_data[ticker]
                
                if 'daily_return' in df.columns and len(df) > 0:
                    total_return = (1 + df['daily_return']).prod() - 1
                    volatility = df['daily_return'].std() * np.sqrt(252)  # Annualized
                    max_drawdown = df.get('drawdown', pd.Series([0])).min()
                    
                    comparison_data.append({
                        'Institution': ticker,
                        'Total Return': total_return,
                        'Volatility': volatility,
                        'Max Drawdown': max_drawdown,
                        'Sharpe Ratio': total_return / volatility if volatility > 0 else 0
                    })
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            st.dataframe(comp_df, use_container_width=True)
    
    def _render_individual_institution_analysis(self, institution_data: pd.DataFrame, ticker: str):
        """Render detailed analysis for individual institution"""
        if institution_data.empty:
            st.warning(f"No data available for {ticker}")
            return
        
        # Price chart
        if 'Close' in institution_data.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=institution_data.index,
                y=institution_data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue')
            ))
            
            fig.update_layout(
                title=f"{ticker} Price Chart",
                xaxis_title="Date",
                yaxis_title="Price",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        if 'daily_return' in institution_data.columns:
            returns = institution_data['daily_return'].dropna()
            
            with col1:
                st.metric("Latest Return", f"{returns.iloc[-1]:.3f}" if len(returns) > 0 else "N/A")
            with col2:
                st.metric("Avg Return", f"{returns.mean():.3f}" if len(returns) > 0 else "N/A")
            with col3:
                st.metric("Volatility", f"{returns.std():.3f}" if len(returns) > 0 else "N/A")
            with col4:
                st.metric("Max Return", f"{returns.max():.3f}" if len(returns) > 0 else "N/A")
    
    def _render_correlation_statistics(self, summary_stats: Dict[str, float]):
        """Render correlation summary statistics"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mean Correlation", f"{summary_stats.get('mean_correlation', 0):.3f}")
            st.metric("Std Correlation", f"{summary_stats.get('std_correlation', 0):.3f}")
        
        with col2:
            st.metric("Max Correlation", f"{summary_stats.get('max_correlation', 0):.3f}")
            st.metric("Min Correlation", f"{summary_stats.get('min_correlation', 0):.3f}")
        
        with col3:
            st.metric("High Corr Pairs", summary_stats.get('high_correlation_pairs', 0))
            st.metric("Total Pairs", summary_stats.get('total_pairs', 0))


def get_market_intelligence_dashboard() -> MarketIntelligenceDashboard:
    """Factory function to get market intelligence dashboard instance"""
    return MarketIntelligenceDashboard()


if __name__ == "__main__":
    # Example usage for testing
    dashboard = get_market_intelligence_dashboard()
    dashboard.render_market_intelligence_tab()