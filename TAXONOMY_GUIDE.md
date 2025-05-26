# 📋 Financial ETL Taxonomy Guide

## 🎯 Purpose
This guide ensures consistent data organization and file naming across all team uploads to the Pure ETL system.

## 🏦 Standardized Institution Names

### US Banks
- **JPMorgan** (JPMorgan Chase & Co.)
- **BankOfAmerica** (Bank of America Corporation)
- **Citigroup** (Citigroup Inc.)
- **WellsFargo** (Wells Fargo & Company)
- **GoldmanSachs** (The Goldman Sachs Group, Inc.)
- **MorganStanley** (Morgan Stanley)
- **USBancorp** (U.S. Bancorp)
- **TrustFinancial** (Truist Financial Corporation)
- **PNCFinancial** (PNC Financial Services Group)
- **CapitalOne** (Capital One Financial Corporation)

### European Banks
- **HSBC** (HSBC Holdings plc)
- **Barclays** (Barclays plc)
- **Lloyds** (Lloyds Banking Group plc)
- **RoyalBankScotland** (NatWest Group plc)
- **StandardChartered** (Standard Chartered plc)
- **Deutsche** (Deutsche Bank AG)
- **UBS** (UBS Group AG)
- **CreditSuisse** (Credit Suisse Group AG)
- **BNPParibas** (BNP Paribas)
- **SocGen** (Société Générale)

## 📅 Quarter and Year Format

### Quarters
- **Q1** (January - March)
- **Q2** (April - June)
- **Q3** (July - September)
- **Q4** (October - December)

### Years
- **2023, 2024, 2025, 2026** (as needed)

## 📄 Document Types

### Primary Document Types
- **EarningsCall** - Quarterly earnings call transcripts
- **Presentation** - Investor presentation slides
- **FinancialSupplement** - Detailed financial data supplements
- **PressRelease** - Official press releases
- **AnnualReport** - Annual reports (10-K, 20-F)
- **QuarterlyReport** - Quarterly reports (10-Q)
- **ProxyStatement** - Proxy statements (DEF 14A)

### Secondary Document Types
- **InvestorUpdate** - General investor updates
- **ConferenceCall** - Conference call transcripts
- **WebcastTranscript** - Webcast transcripts
- **FactSheet** - Company fact sheets
- **Other** - Miscellaneous documents

## 🏷️ File Naming Convention

### Output File Format
```
{Institution}_{Quarter}_{Year}_PureETL_{UploadedBy}_{Timestamp}.csv
```

### Examples
```
JPMorgan_Q1_2025_PureETL_JohnSmith_20250526.csv
Citigroup_Q2_2024_PureETL_SarahJones_20250526.csv
HSBC_Q3_2025_PureETL_TeamMember_20250526.csv
```

## 👤 User Identification

### Format
- **CamelCase** format (no spaces)
- **FirstnameLastname** (e.g., JohnSmith, SarahJones)
- **TeamRole** if preferred (e.g., Analyst1, DataTeam)

### Examples
- JohnSmith
- SarahJones
- DataAnalyst
- TeamLead
- Researcher1

## 📊 Data Organization Benefits

### Consistent Naming
- Easy sorting and filtering
- Clear audit trails
- Professional appearance
- Team coordination

### Search and Discovery
- Quick institution lookup
- Time period filtering
- User accountability
- Processing history

## ✅ Quality Control

### Before Upload
1. Select correct institution from dropdown
2. Choose appropriate quarter and year
3. Enter your name in CamelCase format
4. Verify document type is detected correctly

### After Processing
1. Check output file naming follows convention
2. Verify data quality in CSV output
3. Confirm processing history is updated
4. Save files in organized directory structure

## 🚀 Team Adoption

### Training Steps
1. Review this taxonomy guide
2. Practice with sample documents
3. Use standardized dropdowns in interface
4. Follow naming conventions consistently

### Best Practices
- Always use dropdown selections when available
- Enter custom institutions in CamelCase format
- Include your name for accountability
- Review output files for quality

## 📞 Support

For questions about taxonomy standards or file organization:
- Refer to this guide first
- Check processing history for examples
- Maintain consistency across team uploads
- Update guide as needed for new institutions

---

**Remember**: Consistent taxonomy ensures professional data organization and easy collaboration across the team!