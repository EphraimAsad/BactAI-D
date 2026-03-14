# BactAI-D User Manual

**Version 1.0**
**AI-Powered Bacterial Identification System**

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Requirements](#2-system-requirements)
3. [Installation Guide](#3-installation-guide)
4. [Getting Started](#4-getting-started)
5. [Using BactAI-D](#5-using-bactai-d)
6. [Understanding Results](#6-understanding-results)
7. [Laboratory Workflow Integration](#7-laboratory-workflow-integration)
8. [Standard Operating Procedures](#8-standard-operating-procedures)
9. [Advanced Features](#9-advanced-features)
10. [Troubleshooting](#10-troubleshooting)
11. [FAQ](#11-faq)

---

## 1. Introduction

### What is BactAI-D?

BactAI-D (Bacterial AI-Diagnostics) is an intelligent software system that helps microbiologists identify bacterial genera and species from phenotypic test results. Simply describe your observations in natural language, and BactAI-D will:

- Identify the most likely bacterial genus
- Provide probability scores and confidence levels
- Suggest the most likely species within each genus
- Explain the reasoning behind each identification

### Who Should Use This Software?

- Clinical microbiologists
- Laboratory technicians
- Microbiology students and educators
- Research scientists working with bacterial cultures

### Key Benefits

| Benefit | Description |
|---------|-------------|
| **Speed** | Get identification results in seconds |
| **Accuracy** | Multiple AI methods ensure reliable results |
| **Explainability** | Understand why each identification was made |
| **Ease of Use** | Natural language input - no complex forms |

---

## 2. System Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| **Operating System** | Windows 10/11, macOS 10.15+, or Linux |
| **RAM** | 8 GB (16 GB recommended) |
| **Storage** | 3 GB free space |
| **Python** | Version 3.10 or later |
| **Internet** | Required for initial setup only |

### Recommended for Best Performance

- 16 GB RAM or more
- SSD storage
- Ollama installed for AI-powered explanations

---

## 3. Installation Guide

### Step 1: Install Python

**Windows:**
1. Go to [python.org/downloads](https://python.org/downloads)
2. Download Python 3.10 or later
3. Run the installer
4. **IMPORTANT:** Check the box "Add Python to PATH"
5. Click "Install Now"

**Verify installation:**
```
Open Command Prompt and type: python --version
You should see: Python 3.10.x (or higher)
```

### Step 2: Install Ollama (Recommended)

Ollama provides AI-powered explanations for identification results.

1. Go to [ollama.com](https://ollama.com)
2. Download and install Ollama for your operating system
3. Open a terminal/command prompt and run:
   ```
   ollama pull llama3.2:3b
   ```
4. Wait for the download to complete (~2 GB)

**Note:** BactAI-D works without Ollama, but will use simpler template-based explanations.

### Step 3: Extract BactAI-D

1. Locate your `BactAID_vX.X_XXXXXXXX.zip` file
2. Right-click and select "Extract All..."
3. Choose a location (e.g., `C:\BactAID` or your Documents folder)
4. Click "Extract"

### Step 4: Install Dependencies

**Windows (Easy Method):**
1. Open the extracted `BactAID` folder
2. Double-click `BactAID_Launcher.bat`
3. Dependencies will install automatically on first run

**Manual Method:**
1. Open Command Prompt
2. Navigate to the BactAID folder:
   ```
   cd C:\path\to\BactAID
   ```
3. Install dependencies:
   ```
   pip install -r backend\requirements.txt
   ```

---

## 4. Getting Started

### Starting BactAI-D

**Method 1: Launcher (Recommended)**
1. Open the BactAID folder
2. Double-click `BactAID_Launcher.bat`
3. Wait for the server to start
4. Your browser will open automatically to the interface

**Method 2: Command Line**
1. Open Command Prompt
2. Navigate to BactAID folder
3. Run:
   ```
   python backend\app.py
   ```
4. Open your browser to: `http://localhost:8000`

### First-Time Startup

On first launch, you may see:
- Dependencies being installed (wait for completion)
- Model files being downloaded (one-time only)
- Ollama status messages

This is normal and only happens once.

### Verifying Installation

Once started, you should see:
- A web interface with an input text area
- "Identify" button
- Status indicators showing system health

---

## 5. Using BactAI-D

### Basic Identification

**Step 1: Enter Your Observations**

In the text area, describe the bacterial phenotype using natural language. For example:

```
Gram positive cocci arranged in clusters, catalase positive,
coagulase positive, beta hemolysis on blood agar,
golden yellow pigment, ferments mannitol
```

**Step 2: Click "Identify"**

Press the "Identify" button or hit Enter.

**Step 3: Review Results**

Results appear showing:
- Top 5 genus matches with probabilities
- Confidence bands for each match
- Detailed explanation cards

### Input Guidelines

**What to Include:**

| Test Type | Example Descriptions |
|-----------|---------------------|
| **Gram Stain** | "Gram positive", "Gram negative", "Gram variable" |
| **Morphology** | "cocci", "rods", "coccobacilli", "spiral" |
| **Arrangement** | "clusters", "chains", "pairs", "tetrads" |
| **Biochemical Tests** | "catalase positive", "oxidase negative" |
| **Growth** | "aerobic", "anaerobic", "facultative" |
| **Hemolysis** | "beta hemolysis", "alpha hemolysis", "gamma (none)" |
| **Motility** | "motile", "non-motile" |
| **Spores** | "spore forming", "non-spore forming" |

**Tips for Best Results:**

1. **Be specific**: "Gram positive cocci in clusters" is better than "Gram positive"
2. **Include multiple tests**: More information = more accurate results
3. **Use standard terminology**: "catalase positive" rather than "cat+"
4. **Mention negatives too**: "oxidase negative, indole negative"

### Sample Inputs

**Example 1: Staphylococcus aureus**
```
Gram positive cocci in clusters, catalase positive, coagulase positive,
mannitol fermenter, beta hemolysis, golden pigment
```

**Example 2: Escherichia coli**
```
Gram negative rod, lactose fermenter, indole positive,
motile, oxidase negative, facultative anaerobe
```

**Example 3: Streptococcus pneumoniae**
```
Gram positive diplococci, alpha hemolytic, optochin sensitive,
bile soluble, catalase negative
```

---

## 6. Understanding Results

### Results Table

The top results are displayed in a table:

| Column | Meaning |
|--------|---------|
| **Genus** | The bacterial genus name |
| **Probability** | Likelihood this is the correct match (%) |
| **Odds** | Relative odds (e.g., "1 in 3") |
| **Confidence** | Quality band of the identification |

### Confidence Bands

| Band | Probability | Interpretation |
|------|-------------|----------------|
| **Excellent Identification** | ≥90% | Very high confidence, likely correct |
| **Good Identification** | 80-89% | High confidence, probably correct |
| **Acceptable Identification** | 65-79% | Moderate confidence, needs verification |
| **Low Discrimination** | <65% | Multiple possibilities, additional tests needed |

### Genus Cards

Click on any genus to see detailed information:

**KEY TRAITS** - Phenotypic characteristics that match the genus
- Shows which of your input traits support this identification

**CONFLICTS** - Any traits that don't match typical genus characteristics
- Helps identify potential misidentifications or atypical strains

**CONCLUSION** - AI-generated summary explaining the identification
- Natural language explanation of why this genus was selected

**SPECIES BEST MATCH** - Most likely species within the genus
- Based on species-specific markers in your input

### Interpreting Results

**Strong Match (>90%)**
- First choice is very likely correct
- Confidence: Excellent
- Action: Report identification

**Moderate Match (70-90%)**
- First choice is probable
- Confidence: Good/Acceptable
- Action: Consider confirmatory tests

**Weak Match (<70%)**
- Multiple genera possible
- Confidence: Low Discrimination
- Action: Perform additional biochemical tests

---

## 7. Laboratory Workflow Integration

### Overview: Implementing BactAI-D in Your Lab

BactAI-D is designed to integrate seamlessly into existing microbiology laboratory workflows. This section provides detailed guidance on implementing the system in clinical and research settings.

### 7.1 Pre-Implementation Checklist

Before deploying BactAI-D, complete the following:

| Step | Action | Responsible | Status |
|------|--------|-------------|--------|
| 1 | Verify system requirements met | IT/Lab Manager | ☐ |
| 2 | Install Python 3.10+ on workstations | IT | ☐ |
| 3 | Install Ollama and download model | IT | ☐ |
| 4 | Extract BactAI-D to shared or local drive | IT | ☐ |
| 5 | Run initial test with known organism | Lab Supervisor | ☐ |
| 6 | Train staff on system usage | Lab Supervisor | ☐ |
| 7 | Establish validation protocols | Quality Manager | ☐ |
| 8 | Document in laboratory procedures | Quality Manager | ☐ |

### 7.2 Recommended Laboratory Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SPECIMEN PROCESSING                               │
│  Receive specimen → Culture → Isolate pure colony                   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PRIMARY IDENTIFICATION                            │
│  Gram stain → Colony morphology → Basic biochemicals                │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    BACTAI-D CONSULTATION                            │
│  Enter observations → Review AI identification → Note confidence    │
└─────────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
        ┌───────────────────┐   ┌───────────────────┐
        │ HIGH CONFIDENCE   │   │ LOW CONFIDENCE    │
        │ (≥80%)            │   │ (<80%)            │
        └───────────────────┘   └───────────────────┘
                    │                       │
                    ▼                       ▼
        ┌───────────────────┐   ┌───────────────────┐
        │ Proceed with      │   │ Perform additional│
        │ confirmatory test │   │ biochemical tests │
        │ if required       │   │ Re-run BactAI-D   │
        └───────────────────┘   └───────────────────┘
                    │                       │
                    └───────────┬───────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FINAL IDENTIFICATION                              │
│  Document results → Report → Archive                                │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.3 Data Entry Best Practices

#### Systematic Observation Recording

Record observations in this standardized order for consistent results:

```
1. GRAM STAIN RESULT
   "Gram [positive/negative/variable]"

2. CELL MORPHOLOGY
   "[cocci/rods/coccobacilli/spiral/pleomorphic]"

3. CELL ARRANGEMENT
   "[clusters/chains/pairs/tetrads/palisades/single]"

4. COLONY CHARACTERISTICS (if relevant)
   "colony: [size], [color], [texture], [hemolysis type]"

5. OXYGEN REQUIREMENTS
   "[aerobic/anaerobic/facultative/microaerophilic]"

6. PRIMARY BIOCHEMICALS
   "catalase [positive/negative], oxidase [positive/negative]"

7. SECONDARY BIOCHEMICALS
   "[test name] [positive/negative]" for each test performed

8. SPECIAL CHARACTERISTICS
   "[motility/spores/pigment/odor] if observed"
```

#### Example: Complete Data Entry

```
Gram positive cocci in clusters
Colony: medium, golden yellow, opaque, beta hemolysis on BAP
Facultative anaerobe
Catalase positive, coagulase positive
Mannitol fermentation positive
DNase positive
Novobiocin sensitive
```

### 7.4 Multi-User Environment Setup

For laboratories with multiple workstations:

**Option A: Shared Network Installation**
```
1. Install BactAI-D on a network drive (e.g., \\server\apps\BactAID)
2. Each workstation runs the launcher from the network location
3. Pro: Single installation, easy updates
4. Con: Requires network connectivity, slower startup
```

**Option B: Local Installation per Workstation**
```
1. Copy BactAI-D folder to each workstation
2. Each workstation runs independently
3. Pro: Works offline, faster performance
4. Con: Multiple installations to maintain
```

**Option C: Client-Server Setup**
```
1. Run BactAI-D server on a dedicated machine
2. Other workstations access via browser (http://server-ip:8000)
3. Pro: Centralized, works on any device with browser
4. Con: Server must always be running
```

### 7.5 Quality Control Procedures

#### Daily QC Check

Before first use each day:
1. Start BactAI-D
2. Run the standard QC test:
   ```
   Gram positive cocci in clusters, catalase positive, coagulase positive
   ```
3. Verify result shows "Staphylococcus" with >90% confidence
4. Document QC result in daily log

#### Weekly Validation

Test with 3 known organisms covering different groups:

| QC Organism | Expected Result | Acceptable Range |
|-------------|-----------------|------------------|
| S. aureus ATCC 25923 | Staphylococcus ≥95% | ≥90% |
| E. coli ATCC 25922 | Escherichia ≥95% | ≥90% |
| P. aeruginosa ATCC 27853 | Pseudomonas ≥95% | ≥90% |

---

## 8. Standard Operating Procedures

### SOP-001: Bacterial Identification Using BactAI-D

**Purpose:** To establish a standardized procedure for bacterial identification using the BactAI-D system.

**Scope:** All microbiology laboratory personnel performing bacterial identification.

**Materials Required:**
- Computer with BactAI-D installed
- Gram stain results
- Biochemical test results
- Colony observation notes

### Procedure

#### Step 1: Specimen Preparation (Pre-requisite)

1.1. Ensure a pure culture has been isolated
1.2. Complete Gram stain examination
1.3. Record colony morphology
1.4. Perform primary biochemical tests (minimum: catalase, oxidase)
1.5. Perform additional biochemicals as indicated by Gram stain

#### Step 2: System Startup

2.1. Double-click `BactAID_Launcher.bat`
2.2. Wait for the browser window to open
2.3. Verify the interface loads correctly
2.4. Check Ollama status indicator (green = AI explanations available)

#### Step 3: Data Entry

3.1. Click in the text input area
3.2. Enter observations in natural language, including:
   - Gram stain result and morphology
   - Cell arrangement
   - Colony characteristics
   - All biochemical test results (positive AND negative)
   - Any special characteristics observed

3.3. Example entry format:
```
Gram negative rod, oxidase positive, catalase positive,
lactose non-fermenter, motile, citrate positive,
growth at 42°C, blue-green pigment, grape-like odor
```

#### Step 4: Analysis

4.1. Click the "Identify" button
4.2. Wait for results to appear (typically 2-10 seconds)
4.3. Review the results table showing top 5 matches

#### Step 5: Result Interpretation

5.1. Examine the probability and confidence band of the top result

| Confidence Band | Action Required |
|-----------------|-----------------|
| Excellent (≥90%) | Proceed to Step 6 |
| Good (80-89%) | Proceed to Step 6, consider confirmatory test |
| Acceptable (65-79%) | Perform additional tests, re-analyze |
| Low (<65%) | Perform additional tests, re-analyze |

5.2. Review the explanation card for the top genus
5.3. Check for any conflicts noted by the system
5.4. Review species-level identification if provided

#### Step 6: Confirmation

6.1. For Excellent/Good identifications:
   - Compare AI result with expected organism based on specimen source
   - Verify key characteristics match
   - Perform confirmatory test if required by lab protocol

6.2. For Acceptable/Low identifications:
   - Note which additional tests are suggested
   - Perform additional biochemical tests
   - Re-enter complete data and re-analyze

#### Step 7: Documentation

7.1. Record in laboratory information system:
   - Organism identification
   - Confidence level from BactAI-D
   - Confirmatory tests performed
   - Final reported identification

7.2. Archive the BactAI-D result if required by lab policy

#### Step 8: System Shutdown

8.1. Close the browser window
8.2. Close the command prompt window (or press Ctrl+C)
8.3. BactAI-D server will stop automatically

---

### SOP-002: BactAI-D Troubleshooting

**Purpose:** To provide guidance for resolving common BactAI-D issues.

#### Issue: Application Won't Start

```
Step 1: Verify Python installation
   - Open Command Prompt
   - Type: python --version
   - Should show Python 3.10 or higher

Step 2: If Python not found
   - Reinstall Python with "Add to PATH" checked
   - Restart computer

Step 3: If Python found but app won't start
   - Navigate to BactAID folder in Command Prompt
   - Run: pip install -r backend\requirements.txt
   - Try starting again
```

#### Issue: Low Confidence Results

```
Step 1: Review input data
   - Is Gram stain result included?
   - Are morphology and arrangement specified?
   - Are both positive AND negative test results included?

Step 2: Add more information
   - Include colony characteristics
   - Add results from additional biochemical tests
   - Specify growth conditions

Step 3: Check for data entry errors
   - Verify test results are correctly recorded
   - Ensure no contradictory information entered
```

#### Issue: Conflicting Results

```
Step 1: Review the CONFLICTS section in the result card
   - Identify which traits are conflicting

Step 2: Verify original test results
   - Re-check biochemical test readings
   - Consider repeating questionable tests

Step 3: Consider atypical strains
   - Some organisms show atypical reactions
   - Document and proceed with additional identification methods
```

---

### SOP-003: Adding Custom Organisms to Database

**Purpose:** To document the procedure for adding new bacterial records to BactAI-D.

#### Procedure

1. **Prepare organism data**
   - Collect complete phenotypic profile
   - Document genus and species name
   - Note key identifying characteristics

2. **Edit bacteria database**
   - Open `data/bacteria_db.xlsx` in Excel
   - Add new row with organism data
   - Follow existing column format
   - Save file

3. **Create knowledge base entry (optional)**
   - Create folder: `data/rag/knowledge_base/[GenusName]/`
   - Create `genus.json` with genus-level traits
   - Create `species.json` with species-level traits
   - Follow format of existing entries

4. **Rebuild indexes**
   - Start BactAI-D
   - Go to Training tab
   - Click "Build RAG Index"
   - Wait for completion

5. **Validate addition**
   - Enter phenotype for new organism
   - Verify correct identification
   - Document validation result

---

## 9. Advanced Features

### Training Tab

Access via the "Training" tab in the interface.

| Function | Description |
|----------|-------------|
| **Evaluate Parsers** | Test parser accuracy on gold standards |
| **Train Gold** | Retrain from gold test cases |
| **Train Weights** | Relearn parser reliability weights |
| **Train Genus** | Retrain the ML genus classifier |
| **Build RAG Index** | Rebuild the knowledge base index |

### Using the LLM Parser

Check "Use LLM Parser" for enhanced text understanding:
- Better handling of non-standard terminology
- Improved natural language interpretation
- Slightly slower but more robust

### API Access

For programmatic access, use the REST API:

```python
import requests

response = requests.post('http://localhost:8000/api/identify', json={
    'text': 'Gram positive cocci, catalase positive',
    'use_llm': False
})

results = response.json()
print(results['top5_table'])
```

---

## 10. Troubleshooting

### Common Issues

#### "Python is not recognized"

**Problem:** Python not in system PATH

**Solution:**
1. Reinstall Python
2. Check "Add Python to PATH" during installation
3. Restart your computer

#### "Module not found" errors

**Problem:** Dependencies not installed

**Solution:**
```
pip install -r backend\requirements.txt
```

#### Browser shows "Not Found" or blank page

**Problem:** Wrong URL or server not running

**Solution:**
1. Ensure you're accessing `http://localhost:8000` (not 3000)
2. Check that the server is running (terminal shows activity)
3. Try refreshing the page

#### "Ollama not found" warning

**Problem:** Ollama not installed or not running

**Solution:**
1. Install Ollama from [ollama.com](https://ollama.com)
2. Run: `ollama pull llama3.2:3b`
3. Ensure Ollama is running before starting BactAI-D

**Note:** BactAI-D works without Ollama using template explanations.

#### Slow first startup

**Problem:** Models being downloaded

**Solution:** This is normal on first run. Wait for completion (may take several minutes).

#### Low confidence results

**Problem:** Insufficient input information

**Solution:**
1. Add more phenotypic tests to your description
2. Include both positive and negative results
3. Be specific about morphology and arrangement

### Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| "Database not found" | Missing data files | Re-extract the zip completely |
| "Port 8000 in use" | Another instance running | Close other BactAI-D windows |
| "Connection refused" | Server crashed | Restart the application |

---

## 11. FAQ

### General Questions

**Q: Do I need an internet connection?**
A: Only for initial setup. Once installed, BactAI-D works offline.

**Q: Is my data sent anywhere?**
A: No. All processing happens locally on your computer.

**Q: What bacteria can BactAI-D identify?**
A: The database includes 122 bacterial records covering 100+ genera commonly encountered in clinical microbiology.

**Q: How accurate is BactAI-D?**
A: Accuracy depends on input quality. With complete phenotype data, identification accuracy exceeds 90% for common pathogens.

### Technical Questions

**Q: Can I add new bacteria to the database?**
A: Yes. Add records to `data/bacteria_db.xlsx` and genus knowledge files to `data/rag/knowledge_base/`.

**Q: Why are some explanations simpler than others?**
A: If Ollama is not running, BactAI-D uses template-based explanations. Install Ollama for AI-powered detailed explanations.

**Q: Can I use a different LLM model?**
A: Yes. Set the environment variable `BACTAI_OLLAMA_MODEL` to any Ollama-supported model.

### Usage Questions

**Q: What if I get "Low Discrimination"?**
A: This means multiple genera are possible. Add more biochemical test results for better discrimination.

**Q: Should I include uncertain results?**
A: It's better to omit tests you're unsure about. Incorrect information reduces accuracy.

**Q: Can I identify mixed cultures?**
A: BactAI-D is designed for pure cultures. For mixed cultures, describe each organism separately.

---

## Quick Reference Card

### Starting BactAI-D
```
Double-click: BactAID_Launcher.bat
   or
Command line: python backend\app.py
Browser: http://localhost:8000
```

### Sample Input
```
Gram positive cocci in clusters
Catalase positive
Coagulase positive
Beta hemolysis
```

### Confidence Guide
- **≥90%** → Excellent → Report
- **80-89%** → Good → Likely correct
- **65-79%** → Acceptable → Verify
- **<65%** → Low → More tests needed

### Support
- Check `INSTALL.txt` for setup help
- See `README.md` for technical details
- Report issues to the development team

---

*BactAI-D User Manual v1.0*
*© 2024 BactAI-D Development Team*
