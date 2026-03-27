## Live App

**[thegrowthanalysisapp.streamlit.app](https://thegrowthanalysisapp.streamlit.app/)**

## Abstract

TheGrowthAnalysisApp is a Streamlit-based application for analyzing microbial growth curves from plate reader time-series data. Users upload raw optical density measurements and a plate map, configure analysis parameters, and the app computes baseline-corrected growth statistics including maximum OD, maximum specific growth rate (μ_max), lag time, and exponential phase boundaries across all wells.

The app supports multiple analysis methods:
- **Non-parametric methods**: Sliding Window and Spline fitting for data-driven analysis
- **Parametric models**: Logistic, Gompertz, Richards, and Baranyi-Roberts models for theory-driven fitting

Key features include:
- Configurable phase boundary detection with adjustable μ_max thresholds (default 50%)
- Interactive quality control with lasso selection for manual data point selection
- Automatic replicate detection and averaging
- Comprehensive visualization options (OD curves, 1st derivative, specific growth rate μ)
- Bulk export of curated results and publication-ready plots

## Installation

### Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system (recommended)
- OR Python 3.9+ with pip
- Git (optional, for cloning the repository)

### Setup Instructions

#### Option 1: Using Conda (Recommended)

1. **Clone or download the repository**

   ```bash
   git clone https://github.com/yourusername/TheGrowthAnalysisApp.git
   cd TheGrowthAnalysisApp
   ```

2. **Create the conda environment**

   Use the provided `environment.yaml` file to create a conda environment with all required dependencies:

   ```bash
   conda env create -f environment.yaml
   ```

3. **Activate the environment**

   ```bash
   conda activate growth_curves_app_env
   ```

#### Option 2: Using pip

1. **Clone or download the repository**

   ```bash
   git clone https://github.com/yourusername/TheGrowthAnalysisApp.git
   cd TheGrowthAnalysisApp
   ```

2. **Create a virtual environment (optional but recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Launch the Streamlit app:

```bash
streamlit run app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`.

- Ensure you're in the project directory when running `streamlit run app.py`

## How to Use the App

### 1. Prepare Your Input Files

- **Plate reader data**: A time-series table with time in the first column and OD measurements for each well in subsequent columns
- **Plate map**: A table that maps wells to sample metadata (strain, condition, replicate, etc.)

Sample files are included in the repo:
- `example_data.xlsx`
- `example_plate_map.xls`

#### Required Header Formats

**Plate reader data** (first column must be `Time`, remaining columns must be wells like `A1`...`H12`):

```text
Time | A1   | A2   | A3   | ... | H12
0    | 0.05 | 0.06 | 0.05 | ... | 0.04
12   | 0.08 | 0.09 | 0.07 | ... | 0.06
```

**Plate map** (first column must be `rows`, remaining columns must be numeric `1`...`12`):

```text
rows | 1              | 2              | 3     | ... | 12
A    | Sample1_Cond1  | Sample1_Cond2  | BLANK | ... |
B    | Sample2_Cond1  | Sample2_Cond2  |       | ... |
```

- Samples with the same name will be treated as replicates
- Use 'BLANK' for blank wells (used for baseline correction)
- Leave cells empty for wells to ignore
- The first '_' is used to split strain and condition labels for visualization

### 2. Upload and Configure (Upload & Analyse Page)

1. **Upload files**
   - Upload your plate reader data file (Excel format)
   - Upload your plate map (Excel format)
   - Click "Load plate" to validate and store the data

2. **Configure preprocessing parameters**
   - **Time unit**: Select the unit used in your data file (seconds, minutes, or hours)
   - **Pathlength**: Set the optical pathlength of your plate reader (default: 0.42 cm)
   - **Time range**: Clip the time series to focus on relevant growth phases
   - **Blank subtraction**: Enable to subtract the mean of BLANK wells
   - **Exclude wells**: Manually exclude contaminated or problematic wells

3. **Select analysis method**
   - **Sliding Window (non-parametric)**: Uses a sliding window of data points to calculate growth rate
   - **Spline (non-parametric)**: Fits a smoothing spline to the data for flexible curve fitting
   - **Parametric models**: Logistic, Gompertz, Richards, or Baranyi-Roberts models

4. **Configure phase boundary detection**
   - **Lag phase cutoff**: Fraction of μ_max used to define lag phase end (default: 0.5)
   - **Exponential phase cutoff**: Fraction of μ_max used to define exponential phase end (default: 0.5)
   - Higher values = more conservative phase detection

5. **Set 'No Growth' thresholds**
   - Minimum data points required
   - Minimum signal-to-noise ratio
   - Minimum growth rate threshold

6. **Click "Update parameters and analyse selected plate"**

### 3. Review and Curate Fits (Check Growth Fits Page)

- **View individual well growth curves** with fitted parameters
- **Toggle log scale** to view exponential growth more clearly
- **Toggle annotations** to show/hide phase boundaries and growth metrics
- **Interactive lasso selection**: Click and drag on the plot to select specific data points for re-analysis
- **Adjust phase boundaries** using sliders
- **Set maximum OD** manually if needed
- **Mark wells as "No Growth"** to exclude from analysis
- **Re-analyse** individual wells with current settings
- **Exclude wells** from the analysis entirely
- **Navigate between wells** using arrow buttons or keyboard shortcuts (Left/Right arrows)

The page displays three plots:
- Main growth curve with annotations (OD vs Time)
- 1st derivative (dOD/dt vs Time)
- Specific growth rate (μ vs Time)

### 4. Visualize Results (Plate Overviews & Create Visualizations Pages)

**Plate Overviews**
- View all wells in a 96-well plate layout
- Color-coded by growth metrics or sample groups

**Create Visualizations**
- Generate publication-ready plots
- Compare replicates and conditions
- Customize plot appearance

### 5. Export Results (Download Analyzed Data Page)

Export options include:

**Tabulated Data**
- Baseline-corrected time series (CSV)
- Growth statistics per well (CSV)
- Growth statistics averaged per sample (CSV)
- Analysis parameters used (CSV)

**Global Plots**
- Baseline plot (blank well measurements)
- Plate view (96-well overview)
- Replicates plot (grouped by sample name)

**Well Level Plots**
- OD growth curves with annotations
- 1st Derivative (dOD/dt)
- Specific growth rate (μ)

**Annotation Options**
- Phase boundaries
- Time at μ_max
- OD at μ_max
- Maximum OD
- μ_max point
- Fitted model curve

All exports are packaged in a single ZIP file for easy download.

## Analysis Methods

### Non-Parametric Methods

**Sliding Window**
- Slides a fixed-size window across the growth curve
- Fits a linear regression to log-transformed OD at each position
- Maximum slope = μ_max

**Spline**
- Fits a smoothing spline to log-transformed OD
- Derivative of spline gives growth rate at each time point
- More flexible and smoother than sliding window

### Parametric Models

**Logistic**
- Classic S-shaped curve with symmetric inflection point
- Most commonly used for microbial growth

**Gompertz**
- Asymmetric S-curve
- Often fits bacterial growth better than logistic

**Richards**
- Generalized logistic with shape parameter
- Most flexible, use when other models don't fit well

**Baranyi-Roberts**
- Mechanistic model with physiological lag parameter
- Accounts for cell adaptation during lag phase

### Phase Boundary Detection

All methods use the same approach:
1. **Lag phase end**: First time point where μ exceeds the threshold (default: 50% of μ_max)
2. **Exponential phase end**: First time point after μ_max where μ drops below threshold (default: 50% of μ_max)

The threshold is configurable and affects how conservatively phase transitions are detected.

## Tips

- Start with the example files to validate the workflow before using your own data
- If uploads fail, confirm your files are Excel-compatible and have consistent well names
- Use the "Check Growth Fits" page to quality-control your results before export
- Higher phase boundary thresholds (closer to 0.5) give more conservative phase detection
- Lower phase boundary thresholds (closer to 0.1) detect transitions earlier
- The lasso selection tool is useful for excluding outliers or focusing on specific growth phases

## Dependencies

- Python 3.9+
- streamlit
- pandas
- numpy
- scipy
- plotly
- openpyxl (for Excel file handling)
- xlrd (for .xls file support)
- growthcurves (core analysis package)
- streamlit_sortables
- streamlit-aggrid

See `requirements.txt` or `environment.yaml` for specific versions.

## License

MIT

## Citation

t.b.d

## Support

For issues, questions, or feature requests, please open an issue on the GitHub repository.
