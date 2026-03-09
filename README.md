# Who Votes in Chicago? Participation and Representativeness in Presidential Elections, 2008-2024

**Group 37: Rajat Kanti Paul & Sakkhi Raheel**
PPHA 30538 - Data Analytics & Visualization for Public Policy - University of Chicago

## Project Overview

This project examines how socioeconomic and demographic characteristics of Chicago's neighborhoods relate to voter turnout across five presidential elections (2008, 2012, 2016, 2020, 2024).
We combined precinct-level election results with Census demographic data using area-weighted spatial interpolation to create a unified precinct-by-year panel dataset. 
The analysis includes choropleth maps, scatter plots with a PCA-based socioeconomic status index, related hitmap and an interactive Streamlit dashboard.

### Streamlit Dashboard Online
The dashboard is deployed at: https://uchicago-final-project-chicago-voter-turnout.streamlit.app

The dashboard provides interactive exploration of turnout patterns with a choropleth map, demographic scatter plots, and summary statistics across election years.

## Repository Structure

```
.
├── code/
│   ├── final_project.qmd        # Main analysis (renders to PDF)
│   └── app.py                   # Creates interactive dashboard
├── data/
│   ├── raw-data/
│   │   ├── Boundaries_-_City_20260206.geojson
│   │   ├── Census Tract Shapefile (ACS Merging)/
│   │   ├── Chicago precincts shapefiles/
│   │   ├── Data - Raw ACS (demographic variables _ income, education, race, renters, age)/
│   │   └── Data - Voter turnout and registration(chicagoelections.gov)/
│   └── derived-data/
│       ├── ACS Clean Data/
│       ├── voter-turnout-and-registration-data/
│       └── master_panel.csv
├── requirements.txt
├── final_project.pdf
├── .gitignore
└── README.md
```

## Data Sources

| Source | Description | Location |
|--------|-------------|----------|
| U.S. Census Bureau ACS 5-Year | Demographic variables by census tract (downloaded via API) | `data/raw-data/Data - Raw ACS (...)` |
| Chicago Board of Elections | Precinct-level voter registration and ballots cast (.xls files) | `data/raw-data/Data - Voter turnout (...)` |
| Census Bureau TIGER/Line | 2020 census tract boundary shapefiles | `data/raw-data/Census Tract Shapefile (...)` |
| Chicago Data Portal | Precinct boundary GeoJSON files and city boundary | `data/raw-data/Chicago precincts shapefiles/` |

### Manual Data Downloads

The following files must be placed manually in `data/raw-data/` before running the code:

1. **Voter turnout files** (.xls format) from [Chicago Board of Elections](https://chicagoelections.gov): 
Place `2008.xls`, `2012.xls`, `2016.xls`, `2020.xls`, `2024.xls` in `data/raw-data/Data - Voter turnout and registration(chicagoelections.gov)/`

2. **Census tract shapefile** 
from [TIGER/Line Shapefiles]: (https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html) 
Download the 2020 Illinois tract shapefile and place all component files in `data/raw-data/Census Tract Shapefile (ACS Merging)/`

3. **Precinct boundary GeoJSON files** 
from [Chicago Data Portal]:
- For 2024: https://data.cityofchicago.org/Facilities-Geographic-Boundaries/Boundaries-Ward-Precincts-2023-/6piy-vbxa/about_data
- For 2016 and 2020: https://data.cityofchicago.org/Facilities-Geographic-Boundaries/Boundaries-Ward-Precincts-2013-2022-/nvke-umup/about_data
- For 2008 and 2012: https://data.cityofchicago.org/Facilities-Geographic-Boundaries/Precincts2010/2d4k-r48m/about_data

Place the three precinct boundary files in `data/raw-data/Chicago precincts shapefiles/`

4. **City boundary GeoJSON** 
from [Chicago Data Portal]: https://data.cityofchicago.org/Facilities-Geographic-Boundaries/Boundaries-City/qqq8-j68g/about_data 
Place `Boundaries_-_City_<date_of_download>.geojson` in `data/raw-data/`

ACS demographic data is downloaded automatically via the Census API when the code runs.

## Setup and Installation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### Required packages

- streamlit
- pandas
- numpy
- geopandas
- altair
- pydeck
- scikit-learn
- linearmodels
- statsmodels
- seaborn
- vl-convert-python
- xlrd
- requests
- openpyxl
- lxml
- html5lib

### 2. Set Census API key (optional but recommended)

The ACS data download uses the Census Bureau API. Set your API key as an environment variable:

```bash
export CENSUS_API_KEY="your_key"
```

One can obtain a free key at [https://api.census.gov/data/key_signup.html](https://api.census.gov/data/key_signup.html). The code will still run without a key but may be rate-limited.

## Running the Analysis

### Render the PDF report

```bash
cd code
quarto render final_project.qmd
```

This runs the entire pipeline from scratch: downloads ACS data via API, cleans voter turnout files, performs spatial interpolation, builds the master panel dataset, and generates all visualizations. 
It takes approximately 15 minutes due to API calls and spatial overlay computations.

The Quarto file uses a Python ROOT variable to build all file paths:
```python
ROOT = Path(os.getcwd()).resolve().parent
```

By default, this assumes that one would run the above bash code to render pdf report.

In that case:
- os.getcwd() is <project-root>/code.
- Path(os.getcwd()).resolve().parent moves one level up to <project-root>.
- All paths like ROOT / "data" / "raw-data" / ... point to the data/ folder shown in the directory tree above.

If final_project.qmd is saved somewhere else or run it from a different working directory, ROOT must be updated so that it still points to the project’s top-level folder that contains code/ and data/ folders.

### Run the Streamlit dashboard on Local System

```bash
cd code
streamlit run app.py
```

The Streamlit app uses a ROOT variable to locate data and other project files:
```python
ROOT = Path(__file__).resolve().parent.parent
```

By default, this assumes the file app.py lives in the code/ folder and one runs the bash code given above to run streamlit app locally.

In that setup:
- __file__ is <project-root>/code/app.py.
- Path(__file__).resolve().parent is <project-root>/code.
- Path(__file__).resolve().parent.parent moves one level up to <project-root>.

If app.py is moved or saved in a different location, ROOT must be updated so that it still evaluates to the directory that directly contains data/ folder.
