# Sandbox Safety Guide: Integrated Dashboard Testing

## 🛡️ **Safety Overview**

This guide ensures safe testing of the integrated dashboard (Port 8505 + Technical Dashboard) without affecting existing production systems.

## 🎯 **Testing Strategy**

### **Current Production Environment**
- **Port 8505**: Main BoE Supervisor Dashboard (PROTECTED)
- **Port 8509**: Emerging Topics Dashboard (PROTECTED)  
- **Port 8510**: Technical Dashboard Standalone (TESTING)

### **Sandbox Environment**
- **Port 8506**: Sandbox Integrated Dashboard (NEW - SAFE TESTING)
- **Isolated Data**: All test data is separate from production
- **No Dependencies**: Sandbox doesn't affect existing systems

## 🔒 **Protection Mechanisms**

### **1. Port Isolation**
```bash
# Production (PROTECTED)
Port 8505: Main BoE Dashboard
Port 8509: Emerging Topics Dashboard

# Testing (SAFE)
Port 8506: Sandbox Integrated Dashboard
Port 8510: Technical Dashboard Standalone
```

### **2. Data Isolation**
- **Session State Prefix**: All sandbox data uses `sandbox_` prefix
- **Separate Storage**: No shared data with production systems
- **Test Data Only**: Generated synthetic data for testing

### **3. Code Isolation**
- **Separate File**: `sandbox_integrated_dashboard.py` (independent)
- **Import Safety**: Safe imports with error handling
- **No Production Modifications**: Existing dashboards unchanged

## 🧪 **Sandbox Features**

### **Integrated Testing Environment**
1. **Main Dashboard Simulation**: Simulates Port 8505 functionality
2. **Technical Validation**: Full Port 8510 integration
3. **Shared Data Flow**: Tests data sharing between components
4. **Combined Reports**: Tests integrated reporting

### **Safety Features**
- **Visual Indicators**: Clear sandbox branding and warnings
- **Data Generation**: Safe synthetic data creation
- **Error Isolation**: Errors don't affect production
- **Easy Reset**: Clear sandbox data functionality

## 🚀 **Launch Instructions**

### **Step 1: Launch Sandbox (Safe)**
```bash
# Launch sandbox on Port 8506 (safe testing)
cd data_science
streamlit run sandbox_integrated_dashboard.py --server.port 8506
```

### **Step 2: Keep Production Running**
```bash
# Keep existing systems running (if needed)
# Port 8505: Main dashboard (if running)
# Port 8509: Emerging topics (if running)  
# Port 8510: Technical dashboard (currently running)
```

### **Step 3: Test Integration**
1. **Generate Test Data**: Use sandbox data generation
2. **Run Risk Analysis**: Test main dashboard functionality
3. **Technical Validation**: Test technical dashboard integration
4. **Export Reports**: Test combined reporting

## 📊 **Testing Scenarios**

### **Scenario 1: Basic Integration**
1. Generate banking dataset (500 samples)
2. Run risk analysis in main tab
3. Switch to technical validation tab
4. Verify shared data flow
5. Generate combined report

### **Scenario 2: Large Dataset**
1. Generate large dataset (2000 samples)
2. Test performance of both components
3. Verify technical validation scalability
4. Check memory usage and processing time

### **Scenario 3: Error Handling**
1. Test with missing data
2. Test with invalid inputs
3. Verify graceful error handling
4. Ensure production isolation

## 🔍 **Verification Checklist**

### **Before Testing**
- [ ] Confirm production dashboards are not affected
- [ ] Verify sandbox runs on Port 8506 (not 8505)
- [ ] Check all imports are safe and isolated
- [ ] Confirm test data generation works

### **During Testing**
- [ ] Verify data flows between main and technical components
- [ ] Test all dashboard tabs function correctly
- [ ] Check technical validation receives correct data
- [ ] Verify combined reports include both analyses

### **After Testing**
- [ ] Clear sandbox data
- [ ] Verify no production data was affected
- [ ] Document any issues or improvements needed
- [ ] Plan production integration if successful

## ⚠️ **Safety Warnings**

### **DO NOT**
- ❌ Modify existing dashboard files
- ❌ Use production data in sandbox
- ❌ Run sandbox on ports 8505 or 8509
- ❌ Share session state with production systems

### **DO**
- ✅ Use only Port 8506 for sandbox testing
- ✅ Generate synthetic test data
- ✅ Keep sandbox clearly labeled
- ✅ Test thoroughly before production integration

## 🛠️ **Troubleshooting**

### **Common Issues**

**Issue**: Sandbox won't start
**Solution**: Check port 8506 is available, verify dependencies

**Issue**: Technical validation not working
**Solution**: Ensure statistical validation components are installed

**Issue**: Data not sharing between tabs
**Solution**: Check session state initialization and data generation

**Issue**: Performance problems
**Solution**: Reduce sample size, check memory usage

### **Emergency Procedures**

**If Sandbox Affects Production**:
1. Stop sandbox immediately (`Ctrl+C`)
2. Check production dashboards still running
3. Clear any shared session state
4. Restart production if needed

**If Data Corruption**:
1. Clear all sandbox session state
2. Regenerate test data
3. Verify production data integrity
4. Report issue for investigation

## 📈 **Success Metrics**

### **Integration Success**
- [ ] Main dashboard functionality works in sandbox
- [ ] Technical validation receives correct data
- [ ] Shared inference results between both views
- [ ] Combined reports generate successfully
- [ ] No impact on production systems

### **Performance Success**
- [ ] Handles 500+ samples efficiently
- [ ] Technical validation completes in reasonable time
- [ ] Memory usage remains acceptable
- [ ] UI remains responsive

### **Safety Success**
- [ ] Production systems unaffected
- [ ] Data isolation maintained
- [ ] Error handling works correctly
- [ ] Easy to reset and restart

## 🎯 **Next Steps After Successful Testing**

### **If Sandbox Testing Succeeds**
1. **Document Integration Approach**: Record successful integration patterns
2. **Plan Production Integration**: Create step-by-step production deployment
3. **Backup Production**: Ensure production systems are backed up
4. **Gradual Rollout**: Plan phased integration approach

### **Production Integration Plan**
1. **Backup Current Port 8505**: Save existing dashboard
2. **Create Integration Branch**: Version control for changes
3. **Modify Main Dashboard**: Add technical validation tab
4. **Test with Real Data**: Careful testing with actual data
5. **Deploy with Rollback Plan**: Ready to revert if needed

## 📞 **Support**

### **Testing Support**
- **Sandbox Issues**: Check this guide and troubleshooting section
- **Technical Problems**: Review component documentation
- **Integration Questions**: Refer to integration examples

### **Production Support**
- **Before Production**: Complete all sandbox testing
- **During Integration**: Have rollback plan ready
- **After Deployment**: Monitor system performance

---

## 🎉 **Ready for Safe Testing**

The sandbox environment provides a completely safe way to test the integration of the technical dashboard with the main supervisor dashboard. All production systems remain protected while you can thoroughly test the combined functionality.

**Launch Command**: `streamlit run sandbox_integrated_dashboard.py --server.port 8506`