# Manual Installation Guide

### 1. **Set Up a Virtual Environment**
   To avoid conflicts with other packages, create a virtual environment:
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows use: myenv\Scripts\activate
   ```

### 2. **Manually Install Packages**

You can use `pip` to install most of the packages directly


   - Install `pandas`, `numpy`, `scipy`, `matplotlib`, and `seaborn`:
     ```bash
     pip install pandas numpy scipy matplotlib seaborn
     ```
   - Install `scikit-learn`:
     ```bash
     pip install scikit-learn
     ```

#### 2.4. **Genetic Algorithm**
   - Install the `deap` package:
     ```bash
     pip install deap
     ```

#### 2.5. **Progress Bar**
   - Install `alive-progress` for progress bars:
     ```bash
     pip install alive-progress
     ```

#### 2.6. **Numexpr**
   - Install `numexpr` for efficient numerical expressions:
     ```bash
     pip install numexpr
     ```

#### 2.7. **Bioinformatics and Network Analysis**
   - Install `bioservices` for KEGG, `networkx` for network analysis:
     ```bash
     pip install bioservices networkx
     ```

#### 2.8. **Web Scraping**
   - Install `requests`, `beautifulsoup4` for web scraping:
     ```bash
     pip install requests beautifulsoup4
     ```

#### 2.9. **Dynamic Time Warping**
- Install `fastdtw` for dynamic time warping:
     ```bash
     pip install fastdtw
     ```

### 3. **Verify Installations**
   After installing, verify by importing the modules in a Python shell:
   ```python
   python -c "import logging, pandas, numpy, scipy, matplotlib, pickle, os, glob, argparse, deap, statistics, math, copy, itertools, operator, random, seaborn, numexpr, bioservices, networkx, requests, re"
   ```