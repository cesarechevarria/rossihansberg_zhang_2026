from pathlib import Path

code = r'''from __future__ import annotations

import math
import os
import shutil
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

import geopandas as gpd
import matplotlib as mpl
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st


# ---------------------------------------------------------------------
# Data configuration
# ---------------------------------------------------------------------

# Raw GitHub URL for the GeoPackage.
DATA_URL = (
    "https://raw.githubusercontent.com/"
    "cesarechevarria/scratch/"
    "ea6d50d07843346ec5ea51083eedab743dc9de88/"
    "china_grid_cellls_cleaned.gpkg"
)

DATA_DIRECTORY = Path(
    os.getenv(
        "STREAMLIT_DATA_DIR",
        Path(tempfile.gettempdir()) / "china_gdppc_app",
    )
)

DEFAULT_DATA_PATH = DATA_DIRECTORY / "china_grid_cellls_cleaned.gpkg"

LAYER_NAME = "china_grid_cells"

ADM1_NAME = "name_1"
ADM2_NAME = "name_2"
CELL_ID = "cell_id"

YEAR_COLUMNS = {
    year: f"gdppc_{year}"
    for year in range(2012, 2023)
}


# ---------------------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="China grid-cell GDP per capita",
    page_icon="🗺️",
    layout="wide",
)


# ---------------------------------------------------------------------
# Download GeoPackage
# ---------------------------------------------------------------------

@st.cache_resource(show_spinner="Downloading GeoPackage from GitHub...")
def get_geopackage(
    url: str = DATA_URL,
    destination: Path = DEFAULT_DATA_PATH,
) -> Path:
    """Download the GeoPackage once and return its local path."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0:
        with destination.open("rb") as existing_file:
            header = existing_file.read(16)

        if header == b"SQLite format 3\x00":
            return destination

        destination.unlink(missing_ok=True)

    partial_path = destination.with_suffix(destination.suffix + ".part")

    request = Request(
        url,
        headers={"User-Agent": "china-gdppc-streamlit-app"},
    )

    try:
        with urlopen(request, timeout=180) as response:
            status = getattr(response, "status", 200)

            if status != 200:
                raise RuntimeError(
                    f"GitHub returned HTTP status {status}."
                )

            with partial_path.open("wb") as output_file:
                shutil.copyfileobj(response, output_file)

        if not partial_path.exists() or partial_path.stat().st_size == 0:
            raise RuntimeError("The downloaded GeoPackage is empty.")

        with partial_path.open("rb") as downloaded_file:
            header = downloaded_file.read(16)

        if header != b"SQLite format 3\x00":
            raise RuntimeError(
                "The downloaded file is not a valid GeoPackage. "
                "Check that DATA_URL is a raw GitHub URL."
            )

        partial_path.replace(destination)
        return destination

    except Exception:
        partial_path.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------
# Load spatial data
# ---------------------------------------------------------------------

@st.cache_data(show_spinner="Loading GeoPackage...")
def load_data(path_string: str) -> gpd.GeoDataFrame:
    """Load and validate the grid-cell layer."""
    path = Path(path_string)

    if not path.exists():
        raise FileNotFoundError(f"GeoPackage not found: {path}")

    grid = gpd.read_file(
        path,
        layer=LAYER_NAME,
        engine="pyogrio",
    )

    required_columns = {
        CELL_ID,
        ADM1_NAME,
        ADM2_NAME,
        *YEAR_COLUMNS.values(),
    }

    missing_columns = sorted(required_columns.difference(grid.columns))

    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing_columns)
        )

    if grid.crs is None:
        raise ValueError("The GeoPackage layer does not have a CRS.")

    if grid.crs.to_epsg() != 4326:
        grid = grid.to_crs(epsg=4326)

    grid = grid.loc[
        grid.geometry.notna() & ~grid.geometry.is_empty
    ].copy()

    if not grid.geometry.is_valid.all():
        grid.geometry = grid.geometry.make_valid()

    return grid


# ---------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------

def colorize(
    values: pd.Series,
    palette: str,
    upper_percentile: float,
    opacity: int,
) -> tuple[list[list[int]], float, float]:
    """Convert GDP-per-capita values to RGBA colors."""
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    valid = numeric[np.isfinite(numeric)]

    if valid.empty:
        missing_colors = [
            [160, 160, 160, 80]
            for _ in numeric
        ]
        return missing_colors, 0.0, 1.0

    lower = float(valid.min())
    upper = float(np.nanpercentile(valid, upper_percentile))

    if not np.isfinite(upper - lower) or upper <= lower:
        normalized = np.zeros(len(numeric), dtype=float)
    else:
        normalized = np.clip(
            (numeric.to_numpy() - lower) / (upper - lower),
            0.0,
            1.0,
        )

    color_map = mpl.colormaps[palette]
    rgba = np.round(color_map(normalized) * 255).astype(int)
    rgba[:, 3] = opacity

    missing = ~np.isfinite(numeric.to_numpy())
    rgba[missing] = [160, 160, 160, 80]

    return rgba.tolist(), lower, upper


def format_value(value: float) -> str:
    """Format GDP-per-capita values without adding units."""
    if pd.isna(value):
        return "No data"

    return f"{value:,.0f}"


def make_legend(
    palette: str,
    minimum: float,
    maximum: float,
) -> str:
    """Generate an HTML color-gradient legend."""
    color_map = mpl.colormaps[palette]
    stops = []

    for index in range(9):
        fraction = index / 8
        red, green, blue, _ = (
            np.array(color_map(fraction)) * 255
        ).astype(int)

        stops.append(
            f"rgb({red},{green},{blue}) {fraction * 100:.0f}%"
        )

    gradient = ", ".join(stops)

    return f"""
    <div style="max-width: 520px; margin: 0.15rem 0 0.8rem 0;">
        <div style="
            height: 13px;
            border-radius: 3px;
            background: linear-gradient(90deg, {gradient});
        ">
        </div>

        <div style="
            display: flex;
            justify-content: space-between;
            font-size: 0.82rem;
            margin-top: 3px;
        ">
            <span>{minimum:,.0f}</span>
            <span>{maximum:,.0f}</span>
        </div>
    </div>
    """


def initial_view(grid: gpd.GeoDataFrame) -> pdk.ViewState:
    """Calculate the initial center and zoom from the data bounds."""
    xmin, ymin, xmax, ymax = grid.total_bounds

    longitude = (xmin + xmax) / 2
    latitude = (ymin + ymax) / 2

    longitude_span = max(xmax - xmin, 1e-6)
    latitude_span = max(ymax - ymin, 1e-6)

    zoom = min(
        math.log2(360 / longitude_span),
        math.log2(170 / latitude_span),
    ) + 0.6

    return pdk.ViewState(
        longitude=longitude,
        latitude=latitude,
        zoom=float(np.clip(zoom, 2.5, 5.0)),
        min_zoom=2,
        max_zoom=12,
        pitch=0,
        bearing=0,
    )


# ---------------------------------------------------------------------
# Load application data
# ---------------------------------------------------------------------

st.title("China grid-cell GDP per capita")

st.caption(
    "Interactive grid-cell map for 2012–2022."
)

configured_url = os.environ.get(
    "CHINA_GDPPC_GPKG_URL",
    DATA_URL,
)

try:
    geopackage_path = get_geopackage(url=configured_url)
    grid_gdf = load_data(str(geopackage_path))

except Exception as exc:
    st.error(f"Unable to load the GeoPackage: {exc}")
    st.stop()


# ---------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------

with st.sidebar:
    st.header("Map controls")

    year = st.select_slider(
        "Year",
        options=list(YEAR_COLUMNS),
        value=max(YEAR_COLUMNS),
    )

    palette = st.selectbox(
        "Color palette",
        options=[
            "viridis",
            "plasma",
            "cividis",
            "magma",
        ],
        index=0,
    )

    upper_percentile = st.slider(
        "Upper color limit (percentile)",
        min_value=90.0,
        max_value=100.0,
        value=99.0,
        step=0.5,
        help=(
            "Values above this percentile receive the darkest color. "
            "Tooltip values are unchanged."
        ),
    )

    opacity = st.slider(
        "Grid opacity",
        min_value=80,
        max_value=255,
        value=205,
        step=5,
    )

    show_grid_lines = st.checkbox(
        "Show grid borders",
        value=True,
    )


# ---------------------------------------------------------------------
# Prepare selected year
# ---------------------------------------------------------------------

value_column = YEAR_COLUMNS[year]
values = grid_gdf[value_column]

fill_colors, legend_min, legend_max = colorize(
    values=values,
    palette=palette,
    upper_percentile=upper_percentile,
    opacity=opacity,
)

map_gdf = grid_gdf[
    [
        CELL_ID,
        ADM1_NAME,
        ADM2_NAME,
        value_column,
        "geometry",
    ]
].copy()

map_gdf["fill_color"] = fill_colors
map_gdf["gdppc_label"] = map_gdf[value_column].map(format_value)
map_gdf["year"] = str(year)


# ---------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------

st.markdown(
    make_legend(
        palette=palette,
        minimum=legend_min,
        maximum=legend_max,
    ),
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Grid-cell layer
# ---------------------------------------------------------------------

grid_layer = pdk.Layer(
    "GeoJsonLayer",
    data=map_gdf,
    id="grid-cells",
    pickable=True,
    auto_highlight=True,
    filled=True,
    stroked=show_grid_lines,
    get_fill_color="fill_color",
    get_line_color=[255, 255, 255, 80],
    get_line_width=0.35,
    line_width_units="pixels",
    line_width_min_pixels=0.2 if show_grid_lines else 0,
)


# ---------------------------------------------------------------------
# Render map without a basemap
# ---------------------------------------------------------------------

deck = pdk.Deck(
    layers=[grid_layer],
    initial_view_state=initial_view(grid_gdf),
    map_style=None,
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

st.pydeck_chart(
    deck,
    use_container_width=True,
    height=720,
)


# ---------------------------------------------------------------------
# Dataset notes
# ---------------------------------------------------------------------

with st.expander("Dataset notes"):
    st.markdown(
        f"""
- GeoPackage layer: `{LAYER_NAME}`
- CRS: `{grid_gdf.crs}`
- Years: `{min(YEAR_COLUMNS)}–{max(YEAR_COLUMNS)}`
- Displayed field: `{value_column}`
- Grid cells: `{len(grid_gdf):,}`
- Cached file: `{geopackage_path}`
        """
    )
'''

output_path = Path("/mnt/data/app_from_github_simplified.py")
output_path.write_text(code, encoding="utf-8")
print(output_path)
