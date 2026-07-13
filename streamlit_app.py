from __future__ import annotations

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
PALETTE = "magma"

YEAR_COLUMNS = {
    year: f"gdppc_{year}"
    for year in range(2012, 2023)
}


# ---------------------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Per capita GDP",
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
    """
    Download the GeoPackage once and return its local path.

    Streamlit reuses the downloaded file during subsequent reruns.
    """
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists() and destination.stat().st_size > 0:
        with destination.open("rb") as existing_file:
            header = existing_file.read(16)

        if header == b"SQLite format 3\x00":
            return destination

        destination.unlink(missing_ok=True)

    partial_path = destination.with_suffix(
        destination.suffix + ".part"
    )

    request = Request(
        url,
        headers={
            "User-Agent": "china-gdppc-streamlit-app",
        },
    )

    try:
        with urlopen(request, timeout=180) as response:
            status = getattr(response, "status", 200)

            if status != 200:
                raise RuntimeError(
                    f"GitHub returned HTTP status {status}."
                )

            with partial_path.open("wb") as output_file:
                shutil.copyfileobj(
                    response,
                    output_file,
                )

        if (
            not partial_path.exists()
            or partial_path.stat().st_size == 0
        ):
            raise RuntimeError(
                "The downloaded GeoPackage is empty."
            )

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
def load_data(
    path_string: str,
) -> gpd.GeoDataFrame:
    """
    Load and validate the grid-cell layer.
    """
    path = Path(path_string)

    if not path.exists():
        raise FileNotFoundError(
            f"GeoPackage not found: {path}"
        )

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

    missing_columns = sorted(
        required_columns.difference(grid.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    if grid.crs is None:
        raise ValueError(
            "The GeoPackage layer does not have a CRS."
        )

    if grid.crs.to_epsg() != 4326:
        grid = grid.to_crs(epsg=4326)

    grid = grid.loc[
        grid.geometry.notna()
        & ~grid.geometry.is_empty
    ].copy()

    if not grid.geometry.is_valid.all():
        grid.geometry = grid.geometry.make_valid()

    return grid


# ---------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------

def colorize(
    values: pd.Series,
    upper_percentile: int,
    opacity: int,
) -> list[list[int]]:
    """
    Convert GDP-per-capita values to Magma RGBA colors.
    """
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).astype(float)

    valid = numeric[
        np.isfinite(numeric)
    ]

    if valid.empty:
        return [
            [160, 160, 160, 80]
            for _ in numeric
        ]

    lower = float(valid.min())

    upper = float(
        np.nanpercentile(
            valid,
            upper_percentile,
        )
    )

    if (
        not np.isfinite(upper - lower)
        or upper <= lower
    ):
        normalized = np.zeros(
            len(numeric),
            dtype=float,
        )
    else:
        normalized = np.clip(
            (
                numeric.to_numpy()
                - lower
            )
            / (
                upper
                - lower
            ),
            0.0,
            1.0,
        )

    color_map = mpl.colormaps[PALETTE]

    rgba = np.round(
        color_map(normalized) * 255
    ).astype(int)

    rgba[:, 3] = opacity

    missing = ~np.isfinite(
        numeric.to_numpy()
    )

    rgba[missing] = [
        160,
        160,
        160,
        80,
    ]

    return rgba.tolist()


def format_value(
    value: float,
) -> str:
    """
    Format GDP-per-capita values without a unit label.
    """
    if pd.isna(value):
        return "No data"

    return f"{value:,.0f}"


def initial_view(
    grid: gpd.GeoDataFrame,
) -> pdk.ViewState:
    """
    Calculate the initial center and zoom from the data bounds.
    """
    xmin, ymin, xmax, ymax = grid.total_bounds

    longitude = (
        xmin + xmax
    ) / 2

    latitude = (
        ymin + ymax
    ) / 2

    longitude_span = max(
        xmax - xmin,
        1e-6,
    )

    latitude_span = max(
        ymax - ymin,
        1e-6,
    )

    zoom = min(
        math.log2(
            360 / longitude_span
        ),
        math.log2(
            170 / latitude_span
        ),
    ) + 0.6

    return pdk.ViewState(
        longitude=longitude,
        latitude=latitude,
        zoom=float(
            np.clip(
                zoom,
                2.5,
                5.0,
            )
        ),
        min_zoom=2,
        max_zoom=12,
        pitch=0,
        bearing=0,
    )


# ---------------------------------------------------------------------
# Load application data
# ---------------------------------------------------------------------

st.title(
    "Per capita GDP"
)

st.caption(
    "China, grid cells (Mendez, n.d.; Rossi-Hansberg & Zhang, 2026)"
)

configured_url = os.environ.get(
    "CHINA_GDPPC_GPKG_URL",
    DATA_URL,
)

try:
    geopackage_path = get_geopackage(
        url=configured_url,
    )

    grid_gdf = load_data(
        str(geopackage_path)
    )

except Exception as exc:
    st.error(
        f"Unable to load the GeoPackage: {exc}"
    )
    st.stop()


# ---------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------

with st.sidebar:

    year = st.selectbox(
        "Year",
        options=list(YEAR_COLUMNS),
        index=len(YEAR_COLUMNS) - 1,
    )

    upper_percentile = st.slider(
        "Percentile stretch",
        min_value=90,
        max_value=100,
        value=95,
        step=1,
    )

    opacity = st.slider(
        "Grid opacity",
        min_value=100,
        max_value=200,
        value=180,
        step=5,
    )


# ---------------------------------------------------------------------
# Prepare selected year
# ---------------------------------------------------------------------

value_column = YEAR_COLUMNS[year]

values = grid_gdf[
    value_column
]

fill_colors = colorize(
    values=values,
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

map_gdf["gdppc_label"] = map_gdf[
    value_column
].map(
    format_value
)

map_gdf["year"] = str(year)


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
    stroked=False,
    get_fill_color="fill_color",
)


# ---------------------------------------------------------------------
# Render map without a basemap
# ---------------------------------------------------------------------

deck = pdk.Deck(
    layers=[
        grid_layer,
    ],
    initial_view_state=initial_view(
        grid_gdf
    ),
    map_provider=None,
    map_style=None,
    tooltip={
        "html": (
            "<b>Per capita GDP ({year})</b>: "
            "{gdppc_label}<br/>"
            "<b>ADM1</b>: {name_1}<br/>"
            "<b>ADM2</b>: {name_2}<br/>"
            "<b>Cell ID</b>: {cell_id}"
        ),
        "style": {
            "backgroundColor": (
                "rgba(24, 24, 24, 0.92)"
            ),
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
