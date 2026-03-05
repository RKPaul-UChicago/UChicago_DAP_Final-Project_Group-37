"""
Chicago Voter Turnout and Demographic Dashboard
Streamlit app for Group 37 - Rajat Kanti Paul and Sakkhi Raheel
Run with: streamlit run app.py
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

#  Page configuration 
st.set_page_config(
    page_title="Chicago Voter Turnout Dashboard",
    layout="wide",
)
alt.data_transformers.disable_max_rows()

#  Constants & paths
ROOT = Path(".")
PRECINCT_DIR = ROOT / "Chicago precincts shapefiles"
MASTER_CSV = ROOT / "data" / "master_panel.csv"
CRS_PROJ = "EPSG:26916"
CRS_GEO = "EPSG:4326"

ELECTION_YEARS = [2008, 2012, 2016, 2020, 2024]

PRECINCT_FILES = {
    2008: "Precincts (-2010)(2008 and 2012 elections).geojson",
    2012: "Precincts (-2010)(2008 and 2012 elections).geojson",
    2016: "Precincts(2013-2022) (2016 and 2020 elections).geojson",
    2020: "Precincts(2013-2022) (2016 and 2020 elections).geojson",
    2024: "Precincts(2023-) (2024 elections).geojson",
}

# Radio-button label to master_panel column
DEMO_OPTIONS = {
    "SES": "ses_pca_0_100",
    "Median Income": "median_hh_income",
    "Education": "pct_college",
    "Age Distribution": "pct_18_29",
}

DEMO_AXIS_LABELS = {
    "ses_pca_0_100": "SES Index (PCA, 0-100)",
    "median_hh_income": "Median Household Income ($)",
    "pct_college": "Bachelor's Degree or Higher (%)",
    "pct_18_29": "Population Aged 18-29 (%)",
}

DEMO_COLORS = {
    "ses_pca_0_100": "#D84315",
    "median_hh_income": "#1565C0",
    "pct_college": "#2E7D32",
    "pct_18_29": "#6A1B9A",
}


#  Helper functions 
def spearman_corr(x, y):
    """Compute Spearman rank correlation using pandas .rank()."""
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 10:
        return np.nan
    return valid["x"].rank().corr(valid["y"].rank())


def turnout_to_color(rate):
    """Map turnout 0-1 to RGB list: red to yellow to green."""
    if rate is None or pd.isna(rate):
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
    lo, hi = s.min(), s.max()
    denom = hi - lo
    if pd.isna(denom) or denom == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - lo) / denom


#  Cached data loaders 
@st.cache_data
def load_master():
    """Load the pre-built master panel CSV and compute SES PCA index."""
    df = pd.read_csv(MASTER_CSV)
    df["precinct_id"] = df["precinct_id"].astype(str)

    # PCA-based SES index within each election year
    pca_vars = ["median_hh_income", "pct_college", "pct_renter", "pct_18_29"]
    pca_out = []
    for yr, sub in df.groupby("election_year"):
        complete = sub[["election_year", "precinct_id"] + pca_vars].dropna().copy()
        if len(complete) < 10:
            continue
        X = complete[pca_vars].copy()
        X["pct_renter"] = -X["pct_renter"]
        X["pct_18_29"] = -X["pct_18_29"]
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

    df["ses_pca_0_100"] = (
        df.groupby("election_year")["ses_pca"].transform(minmax_safe) * 100
    )
    return df


@st.cache_resource
def load_precinct_gdf(year):
    """Load the correct precinct GeoJSON for a given election year."""
    fname = PRECINCT_FILES[year]
    gdf = gpd.read_file(PRECINCT_DIR / fname).to_crs(CRS_PROJ)
    gdf["ward_num"] = pd.to_numeric(gdf["ward"], errors="coerce")
    gdf["precinct_num"] = pd.to_numeric(gdf["precinct"], errors="coerce")
    gdf = gdf.dropna(subset=["ward_num", "precinct_num"]).copy()
    gdf["ward_num"] = gdf["ward_num"].astype(int)
    gdf["precinct_num"] = gdf["precinct_num"].astype(int)
    gdf["precinct_id"] = (
        gdf["ward_num"].apply(lambda w: f"{w:02d}")
        + gdf["precinct_num"].apply(lambda p: f"{p:03d}")
    )
    gdf = gdf.drop_duplicates("precinct_id").reset_index(drop=True)
    return gdf[["precinct_id", "geometry"]]


#  Sidebar navigation 
page = st.sidebar.radio("Navigation", ["Overview", "Dashboard"], index=0)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Group 37 - Rajat Kanti Paul & Sakkhi Raheel\n\n"
    "PPHA 30538 - University of Chicago"
)
#  WELCOME  PAGE
if page == "Overview":
    st.title("Chicago Voter Turnout & Demographic Analysis")
    st.subheader("Presidential Elections - 2008 to 2024")
    st.markdown("---")

    st.markdown(
        """
### Project Description

This dashboard explores how **socioeconomic and demographic characteristics**
of Chicago's neighborhoods relate to **voter turnout** across five presidential
elections (2008, 2012, 2016, 2020, 2024).

The project bridges two datasets that live at different geographic levels:

* **American Community Survey (ACS) 5-Year Estimates** – demographic data
  (income, education, race, age, housing tenure) reported at the
  **census-tract** level (~850 tracts in Chicago).
* **Chicago Board of Elections voter records** – precinct-level registration
  and ballot counts (~1,200 precincts).

Because census tracts and voting precincts **do not share boundaries**, we use
**area-weighted spatial interpolation** (via `gpd.sjoin` and geometric
intersection) to estimate precinct-level demographics from overlapping tracts,
enabling direct comparison with turnout data.
"""
    )

    st.markdown("### Summary of Key Findings")

    st.markdown(
            """
**Income & Education**
The strongest and most consistent positive correlations with turnout.
Precincts in the highest income quintile consistently turn out at
materially higher rates than those in the lowest quintile – a gap that
persists across all five elections but narrows in high-enthusiasm years
(2008, 2020).

**Youth Share**
Precincts with more residents aged 18–29 tend to show lower turnout,
consistent with national patterns of lower youth participation.

**Renter Share**
Consistently negatively correlated with turnout. Residential mobility
disrupts registration and weakens the community ties that facilitate
political participation.

**Racial Composition**
In 2008 (Obama's first run), majority-Black precincts showed elevated
turnout relative to other years – consistent with candidate-driven
mobilization. In other cycles, higher minority share generally correlates
with lower turnout, reflecting structural barriers.

"""
        )

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

    st.markdown(
        """
### Policy Implications

Structural barriers – time off work, transportation, registration complexity – disproportionately affect lower-income residents. Targeted interventions include **expanded early voting**, **automatic voter registration**, **same-day registration**, and
 **community-based civic engagement programs** in minority-majority precincts.

*Navigate to the **Dashboard** page using the sidebar to explore the
data interactively.*
"""
    )


#                            DASHBOARD  PAGE

else:
    st.title("Voter Turnout Dashboard")

    # Load data
    master = load_master()

    # 1. Summary Statistics Table
    st.subheader("Demographic Statistics by Election Year")

    summary = master.groupby("election_year").agg(
        avg_income_k=("median_hh_income", lambda x: x.dropna().mean() / 1000),
        avg_pct_college=("pct_college", lambda x: x.dropna().mean()),
        avg_pct_black=("pct_black", lambda x: x.dropna().mean()),
        avg_pct_hispanic=("pct_hispanic", lambda x: x.dropna().mean()),
        avg_pct_renter=("pct_renter", lambda x: x.dropna().mean()),
        avg_pct_18_29=("pct_18_29", lambda x: x.dropna().mean()),
    ).round(3)

    summary = summary.reset_index()

    summary = summary.reset_index(drop=True)

    summary.columns = [
        "Election Year",
        "Avg Income ($K)", "Avg % College", "Avg % Black",
        "Avg % Hispanic", "Avg % Renter", "Avg % 18-29",
    ]

    disp = summary.copy()
    pct_cols = [
        "Avg % College", "Avg % Black", "Avg % Hispanic",
        "Avg % Renter", "Avg % 18-29",
    ]
    for c in pct_cols:
        disp[c] = (disp[c] * 100).round(1).astype(str) + "%"

    disp["Avg Income ($K)"] = "$" + disp["Avg Income ($K)"].round(1).astype(str) + "K"
    disp["Election Year"] = disp["Election Year"].astype(int)

    # Fancy styling: colored header boxes, subtle row striping, right alignment
    header_bg = "#32BD36"
    header_text = "#ffffff"

    styled = (
        disp.style
        .hide(axis="index")
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", header_bg),
                        ("color", header_text),
                        ("font-weight", "600"),
                        ("padding", "6px 8px"),
                        ("text-align", "center"),
                    ],
                },
            ]
        )
    )

    st.markdown(
        styled.to_html(index=False),
        unsafe_allow_html=True,
    )


    # 2. Year Selector (centered)
    col_left, col_center, col_right = st.columns([1, 1, 1])

    
    with col_center:
        selected_year = st.segmented_control(
            "",
            options=ELECTION_YEARS,
            selection_mode="single",
            default=2024,
            format_func=lambda y: str(y),
            key="year_segmented",
            width="content",
        )

    year_data = master[master["election_year"] == selected_year].copy()

    # Styled info cards for the selected year
    n_precincts = len(year_data)
    avg_turnout = year_data["turnout_rate"].mean() * 100
    avg_reg = year_data["registration_rate"].dropna().mean() * 100

    def info_card(color, label, value, subtitle):
        return f"""
        <div style="background:#f9f9f9; border-radius:6px; padding:16px 20px;
                    border-top:4px solid {color}; text-align:center;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <div style="font-size:13px; color:#666; margin-bottom:6px;">{label}</div>
            <div style="font-size:28px; font-weight:700; color:#333;">{value}</div>
            <div style="font-size:11px; color:#999; margin-top:4px;">{subtitle}</div>
        </div>
        """
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            info_card("#E87722", "Number of Precincts",
                      f"{n_precincts:,}",
                      f"Chicago {selected_year} Election"),
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            info_card("#32BD36", "Voter Registration Rate",
                      f"{avg_reg:.1f}%",
                      "Registered Voters / Voting Age Population"),
            unsafe_allow_html=True,
        )

    with m3:
        st.markdown(
            info_card("#1565C0", "Average Turnout Rate",
                      f"{avg_turnout:.1f}%",
                      "Ballots Cast / Registered Voters"),
            unsafe_allow_html=True,
        )
        
    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Two-column layout: Choropleth | Scatter 
    col_map, col_scatter = st.columns(2)

    # LEFT: Choropleth – Turnout Rate by Precinct (pydeck) 
    with col_map:
        st.subheader(f"Turnout Rate by Precinct – {selected_year}")

        # Load the year-specific precinct geometry
        prec_gdf = load_precinct_gdf(selected_year)

        # Merge turnout onto that year's precincts
        map_merge = prec_gdf.merge(
            year_data[["precinct_id", "turnout_rate"]],
            on="precinct_id",
            how="left",
        )

        # ★ FIX: drop precincts with no turnout data (removes grey NA areas)
        map_merge = map_merge[map_merge["turnout_rate"].notna()].copy()

        map_geo = gpd.GeoDataFrame(
            map_merge, geometry="geometry"
        ).to_crs(CRS_GEO)

        # Build GeoJSON with injected fill colours
        geo_json = json.loads(map_geo.to_json())
        for feat in geo_json["features"]:
            rate = feat["properties"].get("turnout_rate")
            feat["properties"]["fill_color"] = turnout_to_color(rate)
            feat["properties"]["turnout_pct"] = (
                f"{rate * 100:.1f}%"
                if rate is not None and not pd.isna(rate)
                else "N/A"
            )

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

        # Colour legend
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

    # RIGHT: Scatter – Turnout vs Demographics (Altair)
    with col_scatter:
        st.subheader(f"Turnout vs Demographics – {selected_year}")

        demo_choice = st.radio(
            "Select Demographic Variable",
            list(DEMO_OPTIONS.keys()),
            index=2,
            horizontal=True,
        )

        demo_col = DEMO_OPTIONS[demo_choice]
        x_label = DEMO_AXIS_LABELS[demo_col]
        dot_color = DEMO_COLORS[demo_col]

        scatter_df = year_data[["turnout_rate", demo_col]].dropna().copy()

        # Convert proportions → percentages for display
        if demo_col.startswith("pct_"):
            scatter_df[demo_col] = scatter_df[demo_col] * 100

        # Income in thousands
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

        scatter_df["turnout_pct"] = scatter_df["turnout_rate"] * 100

        r = spearman_corr(scatter_df[x_col], scatter_df["turnout_pct"])

        base = alt.Chart(scatter_df)

        x_scale = (
            alt.Scale(domain=[0, 100])
            if demo_col == "ses_pca_0_100"
            else alt.Scale()
        )

        dots = base.mark_circle(
            size=12, opacity=0.3, color=dot_color,
        ).encode(
            x=alt.X(f"{x_col}:Q", title=x_label, scale=x_scale),
            y=alt.Y(
                "turnout_pct:Q",
                title="Turnout Rate (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            tooltip=[
                alt.Tooltip(f"{x_col}:Q", title=x_label, format=x_fmt),
                alt.Tooltip("turnout_pct:Q", title="Turnout %", format=".1f"),
            ],
        )

        regression = base.transform_regression(
            x_col, "turnout_pct",
        ).mark_line(
            color="red", strokeWidth=2.5, clip=True,
        ).encode(
            x=alt.X(f"{x_col}:Q"),
            y=alt.Y("turnout_pct:Q"),
        )

        r_text = f"  (Spearman r = {r:+.3f})" if not np.isnan(r) else ""
        chart_title = f"Turnout vs {demo_choice} – {selected_year}{r_text}"

        scatter_chart = (dots + regression).properties(
            width=480, height=460, title=chart_title,
        )
        st.altair_chart(
            scatter_chart,
            use_container_width=True,
            key=f"scatter_{selected_year}_{demo_choice}",
        )
