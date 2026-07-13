from __future__ import annotations

import math
import os
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = APP_DIR / "china_grid_cellls_cleaned.gpkg"
LAYER_NAME = "china_grid_cells"
ADM1_ID = "gid_1"
ADM1_NAME = "name_1"
ADM2_NAME = "name_2"
CELL_ID = "cell_id"
YEAR_COLUMNS = {year: f"gdppc_{year}" for year in range(2012, 2023)}

st.set_page_config(
    page_title="China grid-cell GDP per capita",
    page_icon="🗺️",
    layout="wide",
)


@st.cache_data(show_spinner="Loading GeoPackage …")
def load_data(path_string: str) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load the grid polygons and derive ADM1 outlines."""
    path = Path(path_string)
    if not path.exists():
        raise FileNotFoundError(f"GeoPackage not found: {path}")

    grid = gpd.read_file(path, layer=LAYER_NAME, engine="pyogrio")

    required = {
        "geometry",
        CELL_ID,
        ADM1_ID,
        ADM1_NAME,
        ADM2_NAME,
        *YEAR_COLUMNS.values(),
    }
    missing = sorted(required.difference(grid.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    if grid.crs is None:
        raise ValueError("The grid layer has no CRS.")
    if grid.crs.to_epsg() != 4326:
        grid = grid.to_crs(4326)

    grid = grid.loc[grid.geometry.notna() & ~grid.geometry.is_empty].copy()
    if not grid.geometry.is_valid.all():
        grid.geometry = grid.geometry.make_valid()

    adm1 = (
        grid[[ADM1_ID, ADM1_NAME, "geometry"]]
        .dissolve(by=[ADM1_ID, ADM1_NAME], as_index=False)
        .sort_values(ADM1_NAME)
        .reset_index(drop=True)
    )
    return grid, adm1


def colorize(
    values: pd.Series,
    palette: str,
    scaling: str,
    upper_percentile: float,
    opacity: int,
) -> tuple[list[list[int]], float, float]:
    """Convert GDP-per-capita values to RGBA colors."""
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    valid = numeric[np.isfinite(numeric)]
    if valid.empty:
        return [[160, 160, 160, 80] for _ in numeric], 0.0, 1.0

    clipped_upper_raw = float(np.nanpercentile(valid, upper_percentile))
    raw_lower = float(valid.min())
    raw_upper = max(clipped_upper_raw, raw_lower)

    if scaling == "Logarithmic":
        transformed = np.log1p(np.clip(numeric.to_numpy(), a_min=0, a_max=None))
        lower = math.log1p(max(raw_lower, 0.0))
        upper = math.log1p(max(raw_upper, 0.0))
    else:
        transformed = numeric.to_numpy()
        lower = raw_lower
        upper = raw_upper

    if not np.isfinite(upper - lower) or upper <= lower:
        normalized = np.zeros(len(numeric), dtype=float)
    else:
        normalized = np.clip((transformed - lower) / (upper - lower), 0.0, 1.0)

    cmap = mpl.colormaps[palette]
    rgba = np.round(cmap(normalized) * 255).astype(int)
    rgba[:, 3] = opacity

    missing = ~np.isfinite(numeric.to_numpy())
    rgba[missing] = [160, 160, 160, 80]
    return rgba.tolist(), raw_lower, raw_upper


def format_value(value: float, unit_label: str) -> str:
    if pd.isna(value):
        return "No data"
    suffix = f" {unit_label.strip()}" if unit_label.strip() else ""
    return f"{value:,.0f}{suffix}"


def make_legend(palette: str, minimum: float, maximum: float, unit_label: str) -> str:
    cmap = mpl.colormaps[palette]
    stops = []
    for i in range(9):
        frac = i / 8
        r, g, b, _ = (np.array(cmap(frac)) * 255).astype(int)
        stops.append(f"rgb({r},{g},{b}) {frac * 100:.0f}%")
    gradient = ", ".join(stops)
    suffix = f" {unit_label.strip()}" if unit_label.strip() else ""
    return f"""
    <div style="max-width: 520px; margin: 0.15rem 0 0.8rem 0;">
      <div style="height: 13px; border-radius: 3px; background: linear-gradient(90deg, {gradient});"></div>
      <div style="display:flex; justify-content:space-between; font-size:0.82rem; margin-top:3px;">
        <span>{minimum:,.0f}{suffix}</span>
        <span>{maximum:,.0f}{suffix}</span>
      </div>
      <div style="font-size:0.75rem; color:#666;">Values above the upper legend limit use the darkest color.</div>
    </div>
    """


def initial_view(grid: gpd.GeoDataFrame) -> pdk.ViewState:
    xmin, ymin, xmax, ymax = grid.total_bounds
    longitude = (xmin + xmax) / 2
    latitude = (ymin + ymax) / 2
    lon_span = max(xmax - xmin, 1e-6)
    lat_span = max(ymax - ymin, 1e-6)
    zoom = min(math.log2(360 / lon_span), math.log2(170 / lat_span)) + 0.6
    return pdk.ViewState(
        longitude=longitude,
        latitude=latitude,
        zoom=float(np.clip(zoom, 2.5, 5.0)),
        min_zoom=2,
        max_zoom=12,
        pitch=0,
        bearing=0,
    )


st.title("China grid-cell GDP per capita")
st.caption(
    "Interactive grid-cell choropleth for 2012–2022. ADM1 boundaries are drawn as the thicker dark outlines."
)

configured_path = os.environ.get("CHINA_GDPPC_GPKG", str(DEFAULT_DATA_PATH))
try:
    grid_gdf, adm1_gdf = load_data(configured_path)
except Exception as exc:
    st.error(f"Unable to load the GeoPackage: {exc}")
    st.stop()

with st.sidebar:
    st.header("Map controls")
    year = st.select_slider(
        "Year",
        options=list(YEAR_COLUMNS),
        value=max(YEAR_COLUMNS),
    )
    palette = st.selectbox(
        "Color palette",
        options=["viridis", "plasma", "cividis", "magma"],
        index=0,
    )
    scaling = st.radio(
        "Color scaling",
        options=["Linear", "Logarithmic"],
        index=0,
        help="Logarithmic scaling can reveal variation among lower-valued cells when the distribution is highly skewed.",
    )
    upper_percentile = st.slider(
        "Upper color limit (percentile)",
        min_value=90.0,
        max_value=100.0,
        value=99.0,
        step=0.5,
        help="Only the color scale is clipped; tooltip values remain unchanged.",
    )
    opacity = st.slider("Grid opacity", 80, 255, 205, 5)
    unit_label = st.text_input(
        "Unit label",
        value="dataset units",
        help="The source columns do not encode a currency or price-base label, so the app does not assume one.",
    )
    st.divider()
    show_grid_lines = st.checkbox("Show thin grid borders", value=True)
    adm1_width = st.slider("ADM1 border width", 1.0, 6.0, 3.0, 0.5)

value_column = YEAR_COLUMNS[year]
values = grid_gdf[value_column]
fill_colors, legend_min, legend_max = colorize(
    values=values,
    palette=palette,
    scaling=scaling,
    upper_percentile=upper_percentile,
    opacity=opacity,
)

map_gdf = grid_gdf[[CELL_ID, ADM1_ID, ADM1_NAME, ADM2_NAME, value_column, "geometry"]].copy()
map_gdf["fill_color"] = fill_colors
map_gdf["gdppc_label"] = map_gdf[value_column].map(lambda x: format_value(x, unit_label))
map_gdf["year"] = str(year)

metric_columns = st.columns(4)
metric_columns[0].metric("Year", str(year))
metric_columns[1].metric("Features", f"{len(map_gdf):,}")
metric_columns[2].metric("Median", format_value(float(values.median()), unit_label))
metric_columns[3].metric("95th percentile", format_value(float(values.quantile(0.95)), unit_label))

st.markdown(make_legend(palette, legend_min, legend_max, unit_label), unsafe_allow_html=True)

grid_layer = pdk.Layer(
    "GeoJsonLayer",
    data=map_gdf,
    id="grid-cells",
    pickable=True,
    auto_highlight=True,
    filled=True,
    stroked=show_grid_lines,
    get_fill_color="fill_color",
    get_line_color=[255, 255, 255, 70],
    get_line_width=0.35,
    line_width_units="pixels",
    line_width_min_pixels=0.2 if show_grid_lines else 0,
)

adm1_layer = pdk.Layer(
    "GeoJsonLayer",
    data=adm1_gdf,
    id="adm1-boundaries",
    pickable=False,
    filled=False,
    stroked=True,
    get_line_color=[20, 20, 20, 245],
    get_line_width=adm1_width,
    line_width_units="pixels",
    line_width_min_pixels=adm1_width,
)

deck = pdk.Deck(
    layers=[grid_layer, adm1_layer],
    initial_view_state=initial_view(grid_gdf),
    map_provider="carto",
    map_style=pdk.map_styles.CARTO_LIGHT,
    tooltip={
        "html": (
            "<b>GDP per capita ({year})</b>: {gdppc_label}<br/>"
            "<b>ADM1</b>: {name_1}<br/>"
            "<b>ADM2</b>: {name_2}<br/>"
            "<b>Cell ID</b>: {cell_id}"
        ),
        "style": {
            "backgroundColor": "rgba(24, 24, 24, 0.92)",
            "color": "white",
            "fontSize": "13px",
        },
    },
)

st.pydeck_chart(deck, width="stretch", height=720)

with st.expander("Dataset and map notes"):
    st.markdown(
        f"""
- GeoPackage layer: `{LAYER_NAME}`
- CRS: `{grid_gdf.crs}`
- Coverage: `{min(YEAR_COLUMNS)}–{max(YEAR_COLUMNS)}`
- ADM1 identifier and label: `{ADM1_ID}`, `{ADM1_NAME}`
- GDP-per-capita field shown: `{value_column}`
- Polygon features: `{len(grid_gdf):,}`; ADM1 regions: `{adm1_gdf[ADM1_ID].nunique():,}`
- Source file: [`china_grid_cellls_cleaned.gpkg`](https://github.com/cesarechevarria/scratch/blob/ea6d50d07843346ec5ea51083eedab743dc9de88/china_grid_cellls_cleaned.gpkg)

The upper-percentile control clips only the color domain to prevent extreme values from flattening most of the visual variation. Hover tooltips always show the original values.
        """
    )
