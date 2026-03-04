"""
Chicago Voter Turnout and Demographic Dashboard
Streamlit app for Group 37 - Rajat Kanti Paul and Sakkhi Raheel

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import altair as alt
import pydeck as pdk
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Page configuration
st.set_page_config(
    page_title="Chicago Voter Turnout Dashboard",
    layout="wide",
)

alt.data_transformers.disable_max_rows()

# Constants and file paths
ROOT         = Path(".")
PRECINCT_DIR = ROOT / "Chicago precincts shapefiles"
MASTER_CSV   = ROOT / "data" / "master_panel.csv"

CRS_PROJ = "EPSG:26916"
CRS_GEO  = "EPSG:4326"

ELECTION_YEARS = [2008, 2012, 2016, 2020, 2024]

PRECINCT_FILES = {
    2008: "Precincts (-2010)(2008 and 2012 elections).geojson",
    2012: "Precincts (-2010)(2008 and 2012 elections).geojson",
    2016: "Precincts(2013-2022) (2016 and 2020 elections).geojson",
    2020: "Precincts(2013-2022) (2016 and 2020 elections).geojson",
    2024: "Precincts(2023-) (2024 elections).geojson",
}

# Radio button label to master_panel column mapping
DEMO_OPTIONS = {
    "SES":              "ses_pca_0_100",
    "Median Income":    "median_hh_income",
    "Education":        "pct_college",
    "Age Distribution": "pct_18_29",
}

# Readable axis labels for each column
DEMO_AXIS_LABELS = {
    "ses_pca_0_100":     "SES Index (PCA, 0-100)",
    "median_hh_income":  "Median Household Income ($)",
    "pct_college":       "Bachelor's Degree or Higher (%)",
    "pct_18_29":         "Population Aged 18-29 (%)",
}

# Scatter dot color per demographic
DEMO_COLORS = {
    "ses_pca_0_100":     "#D84315",
    "median_hh_income":  "#1565C0",
    "pct_college":       "#2E7D32",
    "pct_18_29":         "#6A1B9A",
}


def spearman_corr(x, y):
    """Compute Spearman rank correlation using pandas .rank()."""
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 10:
        return np.nan
    return valid["x"].rank().corr(valid["y"].rank())


def turnout_to_color(rate):
    """Map turnout 0-1 to RGB color: red to yellow to green."""
    if pd.isna(rate):
        return [180, 180, 180, 100]
    r = float(rate)
    if r < 0.5:
        t = r / 0.5
        return [255, int(255 * t), 0, 180]
    else:
        t = (r - 0.5) / 0.5
        return [int(255 * (1 - t)), int(255 - 127 * t), 0, 180]


def minmax_safe(s):
    """Min-max scaling to [0,1] with protection against constant series."""
    s = pd.to_numeric(s, errors="coerce")
    lo = s.min()
    hi = s.max()
    denom = hi - lo
    if pd.isna(denom) or denom == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - lo) / denom


# Cached data loading
@st.cache_data
def load_master():
    """Load the pre-built master panel CSV and compute SES PCA index."""
    df = pd.read_csv(MASTER_CSV)
    df["precinct_id"] = df["precinct_id"].astype(str)

    # Compute PCA-based SES index within each election year
    pca_vars = ["median_hh_income", "pct_college", "pct_renter", "pct_18_29"]
    pca_out = []

    for yr, sub in df.groupby("election_year"):
        complete = sub[["election_year", "precinct_id"] + pca_vars].dropna().copy()
        if len(complete) < 10:
            continue

        X = complete[pca_vars].copy()
        # Flip so higher = more advantaged
        X["pct_renter"] = -X["pct_renter"]
        X["pct_18_29"]  = -X["pct_18_29"]

        Xs = StandardScaler().fit_transform(X)
        pc1 = PCA(n_components=1).fit_transform(Xs).ravel()

        tmp = complete[["election_year", "precinct_id"]].copy()
        tmp["ses_pca"] = pc1
        pca_out.append(tmp)

    if pca_out:
        ses_df = pd.concat(pca_out, ignore_index=True)
        df = df.merge(ses_df, on=["election_year", "precinct_id"], how="left")
    else:
        df["ses_pca"] = np.nan

    # Rescale within year to 0-100
    df["ses_pca_0_100"] = (
        df.groupby("election_year")["ses_pca"].transform(minmax_safe) * 100
    )

    return df


@st.cache_resource
def load_precinct_gdf(year):
    """Load the correct precinct GeoJSON for a given election year."""
    fname = PRECINCT_FILES[year]
    gdf = gpd.read_file(PRECINCT_DIR / fname).to_crs(CRS_PROJ)

    gdf["ward_num"]     = pd.to_numeric(gdf["ward"],     errors="coerce")
    gdf["precinct_num"] = pd.to_numeric(gdf["precinct"], errors="coerce")
    gdf = gdf.dropna(subset=["ward_num", "precinct_num"]).copy()
    gdf["ward_num"]     = gdf["ward_num"].astype(int)
    gdf["precinct_num"] = gdf["precinct_num"].astype(int)

    gdf["precinct_id"] = (
        gdf["ward_num"].apply(lambda w: f"{w:02d}")
        + gdf["precinct_num"].apply(lambda p: f"{p:03d}")
    )
    gdf = gdf.drop_duplicates("precinct_id").reset_index(drop=True)
    return gdf[["precinct_id", "geometry"]]


# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    ["Welcome", "Dashboard"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Group 37 - Rajat Kanti Paul & Sakkhi Raheel\n\n"
    "PPHA 30538 - University of Chicago"
)


# WELCOME PAGE
if page == "Welcome":

    st.title("Chicago Voter Turnout & Demographic Analysis")
    st.subheader("Presidential Elections - 2008 to 2024")
    st.markdown("---")

    # Project description
    st.markdown(
        """
        ### Project Description

        This dashboard explores how **socioeconomic and demographic
        characteristics** of Chicago's neighborhoods relate to **voter
        turnout** across five presidential elections (2008, 2012, 2016,
        2020, 2024).

        The project bridges two datasets that live at different geographic
        levels:

        * **American Community Survey (ACS) 5-Year Estimates** - demographic
          data (income, education, race, age, housing tenure) reported at the
          **census-tract** level (~800 tracts in Chicago).
        * **Chicago Board of Elections voter records** - precinct-level
          registration and ballot counts (~2,200 precincts).

        Because census tracts and voting precincts **do not share
        boundaries**, we use **area-weighted spatial interpolation** (via
        `gpd.sjoin` and geometric intersection) to estimate precinct-level
        demographics from overlapping tracts, enabling direct comparison with
        turnout data.
        """
    )

    # Key findings
    st.markdown("### Summary of Key Findings")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            """
            **Income & Education**

            The strongest and most consistent positive correlations with
            turnout. Precincts in the highest income quintile consistently
            turn out at materially higher rates than those in the lowest
            quintile - a gap that persists across all five elections but
            narrows in high-enthusiasm years (2008, 2020).

            **Renter Share**

            Consistently negatively correlated with turnout. Residential
            mobility disrupts registration and weakens the community ties
            that facilitate political participation.
            """
        )

    with col_b:
        st.markdown(
            """
            **Racial Composition**

            In 2008 (Obama's first run), majority-Black precincts showed
            elevated turnout relative to other years - consistent with
            candidate-driven mobilization. In other cycles, higher minority
            share generally correlates with lower turnout, reflecting
            structural barriers.

            **Youth Share**

            Precincts with more residents aged 18-29 tend to show lower
            turnout, consistent with national patterns of lower youth
            participation.
            """
        )

    # Data sources
    st.markdown("### Data Sources")
    st.table(
        pd.DataFrame(
            {
                "Source": [
                    "U.S. Census Bureau ACS 5-Year",
                    "Chicago Board of Elections",
                    "Census Bureau TIGER/Line",
                    "Chicago Data Portal",
                ],
                "Description": [
                    "Demographic variables by census tract",
                    "Precinct-level voter registration & ballots cast",
                    "2020 census-tract boundary shapefiles",
                    "Precinct boundary & city boundary GeoJSON files",
                ],
            }
        )
    )

    # Policy implications
    st.markdown(
        """
        ### Policy Implications

        > Structural barriers - time off work, transportation, registration
        > complexity - disproportionately affect lower-income residents.
        > Targeted interventions include **expanded early voting**,
        > **automatic voter registration**, **same-day registration**, and
        > **community-based civic engagement programs** in minority-majority
        > precincts.

        *Navigate to the **Dashboard** page using the sidebar to explore the
        data interactively.*
        """
    )


# DASHBOARD PAGE
else:

    st.title("Voter Turnout Dashboard")

    # Load data
    master = load_master()

    # Summary Statistics Table (top of page)
    st.subheader("Summary Statistics by Election Year")

    summary = master.groupby("election_year").agg(
        n_precincts      = ("precinct_id",      "count"),
        avg_turnout      = ("turnout_rate",      "mean"),
        median_turnout   = ("turnout_rate",      "median"),
        avg_reg_rate     = ("registration_rate", lambda x: x.dropna().mean()),
        avg_income_k     = ("median_hh_income",  lambda x: x.dropna().mean() / 1000),
        avg_pct_college  = ("pct_college",  lambda x: x.dropna().mean()),
        avg_pct_black    = ("pct_black",    lambda x: x.dropna().mean()),
        avg_pct_hispanic = ("pct_hispanic", lambda x: x.dropna().mean()),
        avg_pct_renter   = ("pct_renter",   lambda x: x.dropna().mean()),
        avg_pct_18_29    = ("pct_18_29",    lambda x: x.dropna().mean()),
    ).round(3)

    summary.columns = [
        "N Precincts", "Avg Turnout", "Median Turnout", "Avg Reg Rate",
        "Avg Income ($K)", "Avg % College", "Avg % Black",
        "Avg % Hispanic", "Avg % Renter", "Avg % 18-29",
    ]
    summary.index.name = "Election Year"

    # Format for display
    disp = summary.copy()
    pct_cols = [
        "Avg Turnout", "Median Turnout", "Avg Reg Rate",
        "Avg % College", "Avg % Black", "Avg % Hispanic",
        "Avg % Renter", "Avg % 18-29",
    ]
    for c in pct_cols:
        disp[c] = (disp[c] * 100).round(1).astype(str) + "%"
    disp["Avg Income ($K)"] = "$" + disp["Avg Income ($K)"].round(1).astype(str) + "K"
    disp["N Precincts"] = disp["N Precincts"].astype(int)

    st.dataframe(disp, use_container_width=True, height=230)

    st.markdown("---")

    # Year selector (shared control for both charts below)
    selected_year = st.select_slider(
        "Select Election Year",
        options=ELECTION_YEARS,
        value=2024,
        key="year_selector",
    )
    
    year_data = master[master["election_year"] == selected_year].copy()

    # Two columns: Choropleth on left, Scatter on right
    col_map, col_scatter = st.columns(2)

    # LEFT: Choropleth - Turnout Rate by Precinct (pydeck)
    with col_map:
        st.subheader(f"Turnout Rate by Precinct - {selected_year}")

        prec_gdf = load_precinct_gdf(selected_year)
        map_merge = prec_gdf.merge(
            year_data[["precinct_id", "turnout_rate"]],
            on="precinct_id",
            how="left",
        )
        map_geo = gpd.GeoDataFrame(map_merge, geometry="geometry").to_crs(CRS_GEO)

        # Convert to GeoJSON and inject fill colors
        geo_json = json.loads(map_geo.to_json())
        for feat in geo_json["features"]:
            rate = feat["properties"].get("turnout_rate")
            feat["properties"]["fill_color"] = turnout_to_color(rate)
            if rate is not None and not pd.isna(rate):
                feat["properties"]["turnout_pct"] = f"{rate * 100:.1f}%"
            else:
                feat["properties"]["turnout_pct"] = "N/A"

        layer = pdk.Layer(
            "GeoJsonLayer",
            data=geo_json,
            get_fill_color="properties.fill_color",
            get_line_color=[255, 255, 255, 120],
            get_line_width=10,
            pickable=True,
            stroked=True,
            filled=True,
        )

        view = pdk.ViewState(
            latitude=41.83,
            longitude=-87.68,
            zoom=10,
            pitch=0,
        )

        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view,
            map_style="mapbox://styles/mapbox/light-v10",
            tooltip={
                "text": "Precinct: {precinct_id}\nTurnout: {turnout_pct}"
            },
        )
        st.pydeck_chart(deck, height=520, key=f"deck_{selected_year}")

        # Color legend for the choropleth
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:6px; margin-top:4px;">
              <span style="font-size:13px;">Low (0%)</span>
              <div style="height:14px; flex:1; background: linear-gradient(to right,
                rgb(255,0,0), rgb(255,255,0), rgb(0,128,0));
                border-radius:3px;"></div>
              <span style="font-size:13px;">High (100%)</span>
            </div>
            <div style="text-align:center; font-size:12px; color:gray;
                 margin-top:2px;">Turnout Rate</div>
            """,
            unsafe_allow_html=True,
        )

    # RIGHT: Scatter - Turnout vs selected demographic (Altair)
    with col_scatter:
        st.subheader(f"Turnout vs Demographics - {selected_year}")

        demo_choice = st.radio(
            "Select Demographic Variable",
            list(DEMO_OPTIONS.keys()),
            index=2,
            horizontal=True,
        )

        demo_col  = DEMO_OPTIONS[demo_choice]
        x_label   = DEMO_AXIS_LABELS[demo_col]
        dot_color = DEMO_COLORS[demo_col]

        scatter_df = year_data[["turnout_rate", demo_col]].dropna().copy()

        # Convert proportions to percentages for the plot axis
        # ses_pca_0_100 is already 0-100, no conversion needed
        if demo_col.startswith("pct_"):
            scatter_df[demo_col] = scatter_df[demo_col] * 100

        # Convert income to thousands for readability
        if demo_col == "median_hh_income":
            scatter_df["income_k"] = scatter_df[demo_col] / 1000
            x_col = "income_k"
            x_label = "Median Household Income ($K)"
            x_fmt = "$.0f"
        elif demo_col == "ses_pca_0_100":
            x_col = demo_col
            x_fmt = ".1f"
        else:
            x_col = demo_col
            x_fmt = ".1f"

        # Convert turnout to percentage
        scatter_df["turnout_pct"] = scatter_df["turnout_rate"] * 100

        # Compute Spearman r for the title
        r = spearman_corr(scatter_df[x_col], scatter_df["turnout_pct"])

        # Build chart
        base = alt.Chart(scatter_df)

        # Set x-axis scale (fixed 0-100 for SES, auto for others)
        if demo_col == "ses_pca_0_100":
            x_scale = alt.Scale(domain=[0, 100])
        else:
            x_scale = alt.Scale()

        dots = base.mark_circle(
            size=12, opacity=0.3, color=dot_color,
        ).encode(
            x=alt.X(f"{x_col}:Q", title=x_label, scale=x_scale),
            y=alt.Y("turnout_pct:Q",
                     title="Turnout Rate (%)",
                     scale=alt.Scale(domain=[0, 100])),
            tooltip=[
                alt.Tooltip(f"{x_col}:Q", title=x_label, format=x_fmt),
                alt.Tooltip("turnout_pct:Q", title="Turnout %",
                            format=".1f"),
            ],
        )

        regression = base.transform_regression(
            x_col, "turnout_pct",
        ).mark_line(
            color="red", strokeWidth=2.5,
        ).encode(
            x=alt.X(f"{x_col}:Q"),
            y=alt.Y("turnout_pct:Q"),
        )

        r_text = f"  (Spearman r = {r:+.3f})" if not np.isnan(r) else ""
        chart_title = f"Turnout vs {demo_choice} - {selected_year}{r_text}"

        scatter_chart = (
            (dots + regression)
            .properties(width=480, height=460, title=chart_title)
        )
        st.altair_chart(scatter_chart, use_container_width=True,
                        key=f"scatter_{selected_year}_{demo_choice}")
