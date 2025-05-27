"""
Stakeholder Dashboard Launcher

This script launches the stakeholder-friendly risk assessment dashboard
with proper error handling and dependency checking.
"""

import sys
import os
import subprocess
from pathlib import Path
import importlib.util

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All required dependencies are installed")
    return True

def check_components():
    """Check if Phase 4 components are available"""
    try:
        # Add project path
        sys.path.append(str(Path(__file__).parent))
        
        # Test core components
        from scripts.business_intelligence.stakeholder_translator import StakeholderTranslator
        translator = StakeholderTranslator()
        
        print("✅ Stakeholder translator component ready")
        
        # Test with sample data
        sample_result = translator.translate_risk_score({'composite_risk_score': 0.5})
        if 'classification' in sample_result:
            print("✅ Business intelligence layer functional")
        else:
            print("❌ Business intelligence layer test failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Component check failed: {e}")
        return False

def launch_dashboard():
    """Launch the stakeholder dashboard"""
    try:
        dashboard_path = Path(__file__).parent / "stakeholder_dashboard.py"
        
        if not dashboard_path.exists():
            print(f"❌ Dashboard file not found: {dashboard_path}")
            return False
        
        print("🚀 Launching Stakeholder Dashboard...")
        print(f"📍 Dashboard location: {dashboard_path}")
        print("🌐 Opening in your default web browser...")
        print("\n" + "="*60)
        print("STAKEHOLDER RISK ASSESSMENT DASHBOARD")
        print("="*60)
        print("📊 Upload financial documents to get risk insights")
        print("🎯 Get actionable recommendations in plain English")
        print("📈 View trends and comparative analysis")
        print("📤 Export executive summaries")
        print("="*60)
        
        # Launch Streamlit
        result = subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(dashboard_path),
            "--server.headless", "false",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ], cwd=dashboard_path.parent)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Failed to launch dashboard: {e}")
        return False

def launch_demo_mode():
    """Launch dashboard in demo mode with sample data"""
    print("🎭 Demo Mode: Launching with sample data...")
    
    try:
        # Add project path
        sys.path.append(str(Path(__file__).parent))
        
        from scripts.business_intelligence.stakeholder_translator import StakeholderTranslator
        
        translator = StakeholderTranslator()
        
        # Generate sample insights
        sample_analysis = {
            'composite_risk_score': 0.65,
            'anomaly_detection': {'total_anomalies': 7},
            'time_series': {'trend_direction': 'declining'},
            'topic_analysis': {
                'financial_performance': {'percentage': 30, 'average_sentiment': 0.6},
                'regulatory_compliance': {'percentage': 28, 'average_sentiment': 0.4},
                'technology_digital': {'percentage': 20, 'average_sentiment': 0.7},
                'market_conditions': {'percentage': 15, 'average_sentiment': 0.3},
                'operations_strategy': {'percentage': 7, 'average_sentiment': 0.8}
            },
            'risk_drivers': [
                {'topic': 'regulatory_compliance', 'severity': 'high', 'description': 'Increased regulatory scrutiny'},
                {'topic': 'market_conditions', 'severity': 'medium', 'description': 'Market volatility concerns'},
                {'topic': 'operations_strategy', 'severity': 'low', 'description': 'Strong operational performance'}
            ]
        }
        
        print("\n" + "="*60)
        print("DEMO: SAMPLE RISK ASSESSMENT RESULTS")
        print("="*60)
        
        # Risk classification
        risk_info = translator.translate_risk_score(sample_analysis)
        print(f"🏛️ Institution: Demo Bank")
        print(f"⚠️  Risk Level: {risk_info['classification']}")
        print(f"📊 Risk Score: {risk_info['score']}/10")
        print(f"💬 Assessment: {risk_info['message']}")
        
        # Topic insights
        print(f"\n📈 Key Discussion Topics:")
        topics = translator.translate_topics_to_business_language(sample_analysis)
        for topic in topics[:3]:
            print(f"   • {topic['label']}: {topic['percentage']}% ({topic['risk_level']} risk)")
        
        # Recommendations
        recommendations = translator.generate_stakeholder_recommendations(sample_analysis)
        print(f"\n🎯 Immediate Actions Required:")
        for rec in recommendations['immediate_attention'][:2]:
            print(f"   • {rec['action']}")
        
        print(f"\n🟢 Positive Indicators:")
        for rec in recommendations['positive_indicators'][:2]:
            print(f"   • {rec['strength']}")
        
        # Executive summary
        executive_summary = translator.generate_executive_summary("Demo Bank", sample_analysis)
        print(f"\n📋 Executive Summary Preview:")
        print(executive_summary[:200] + "...")
        
        print("\n" + "="*60)
        print("✅ Demo completed successfully!")
        print("🚀 Ready to launch full dashboard with real data")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo mode failed: {e}")
        return False

def main():
    """Main launcher function"""
    print("🏦 Financial Risk Assessment Dashboard Launcher")
    print("=" * 50)
    
    # Check if demo mode requested
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        return launch_demo_mode()
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        return False
    
    # Check components
    print("🔍 Checking Phase 4 components...")
    if not check_components():
        return False
    
    # Launch dashboard
    print("🔍 All checks passed!")
    return launch_dashboard()

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            print("\n❌ Dashboard launch failed")
            print("💡 Try running with --demo flag for a demonstration")
            print("   python launch_stakeholder_dashboard.py --demo")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard launcher interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)