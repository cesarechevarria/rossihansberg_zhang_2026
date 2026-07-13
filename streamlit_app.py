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

# Use the raw GitHub URL, not the normal /blob/ page URL.
DATA_URL = (
    "https://raw.githubusercontent.com/"
    "cesarechevarria/scratch/"
    "ea6d50d07843346ec5ea51083eedab743dc9de88/"
    "china_grid_cellls_cleaned.gpkg"
)

# The GeoPackage is downloaded into the system's temporary directory.
DATA_DIRECTORY = Path(
    os.getenv(
        "STREAMLIT_DATA_DIR",
        Path(tempfile.gettempdir()) / "china_gdppc_app",
    )
)

DEFAULT_DATA_PATH = DATA_DIRECTORY / "china_grid_cellls_cleaned.gpkg"

LAYER_NAME = "china_grid_cells"

ADM1_ID = "gid_1"
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

@st.cache_resource(
    show_spinner="Downloading GeoPackage from GitHub..."
)
def get_geopackage(
    url: str = DATA_URL,
    destination: Path = DEFAULT_DATA_PATH,
) -> Path:
    """
    Download the GeoPackage from GitHub and return its local path.

    The file is cached locally so Streamlit does not download it during
    every widget rerun.
    """
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Reuse an existing valid GeoPackage.
    if destination.exists() and destination.stat().st_size > 0:
        with destination.open("rb") as existing_file:
            header = existing_file.read(16)

        if header == b"SQLite format 3\x00":
            return destination

        # Remove invalid or incomplete cached files.
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

        if not partial_path.exists():
            raise RuntimeError(
                "The GeoPackage download was not created."
            )

        if partial_path.stat().st_size == 0:
            raise RuntimeError(
                "The downloaded GeoPackage is empty."
            )

        # GeoPackages are SQLite databases, so validate the SQLite header.
        with partial_path.open("rb") as downloaded_file:
            header = downloaded_file.read(16)

        if header != b"SQLite format 3\x00":
            raise RuntimeError(
                "The downloaded file is not a valid GeoPackage. "
                "Check that DATA_URL is a raw GitHub URL."
            )

        # Atomic replacement prevents Streamlit from reading a partial file.
        partial_path.replace(destination)

        return destination

    except Exception:
        partial_path.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------
# Load and prepare spatial data
# ---------------------------------------------------------------------

@st.cache_data(
    show_spinner="Loading GeoPackage..."
)
def load_data(
    path_string: str,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Load the grid-cell layer and derive dissolved ADM1 boundaries.
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
        ADM1_ID,
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

    # Pydeck expects longitude and latitude coordinates.
    if grid.crs.to_epsg() != 4326:
        grid = grid.to_crs(epsg=4326)

    # Remove empty geometries.
    grid = grid.loc[
        grid.geometry.notna()
        & ~grid.geometry.is_empty
    ].copy()

    # Repair invalid geometries when necessary.
    if not grid.geometry.is_valid.all():
        grid.geometry = grid.geometry.make_valid()

    # Dissolve all grid cells belonging to the same ADM1 region.
    adm1 = (
        grid[
            [
                ADM1_ID,
                ADM1_NAME,
                "geometry",
            ]
        ]
        .dissolve(
            by=[ADM1_ID, ADM1_NAME],
            as_index=False,
        )
        .sort_values(ADM1_NAME)
        .reset_index(drop=True)
    )

    return grid, adm1


# ---------------------------------------------------------------------
# Color and display helpers
# ---------------------------------------------------------------------

def colorize(
    values: pd.Series,
    palette: str,
    scaling: str,
    upper_percentile: float,
    opacity: int,
) -> tuple[list[list[int]], float, float]:
    """
    Convert GDP-per-capita values into RGBA colors.
    """
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).astype(float)

    valid = numeric[np.isfinite(numeric)]

    if valid.empty:
        missing_colors = [
            [160, 160, 160, 80]
            for _ in numeric
        ]
        return missing_colors, 0.0, 1.0

    raw_lower = float(valid.min())

    clipped_upper = float(
        np.nanpercentile(
            valid,
            upper_percentile,
        )
    )

    raw_upper = max(
        clipped_upper,
        raw_lower,
    )

    if scaling == "Logarithmic":
        transformed = np.log1p(
            np.clip(
                numeric.to_numpy(),
                a_min=0,
                a_max=None,
            )
        )

        lower = math.log1p(
            max(raw_lower, 0.0)
        )

        upper = math.log1p(
            max(raw_upper, 0.0)
        )

    else:
        transformed = numeric.to_numpy()
        lower = raw_lower
        upper = raw_upper

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
            (transformed - lower)
            / (upper - lower),
            0.0,
            1.0,
        )

    color_map = mpl.colormaps[palette]

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

    return (
        rgba.tolist(),
        raw_lower,
        raw_upper,
    )


def format_value(
    value: float,
    unit_label: str,
) -> str:
    """
    Format GDP-per-capita values for metrics and tooltips.
    """
    if pd.isna(value):
        return "No data"

    unit_label = unit_label.strip()

    suffix = (
        f" {unit_label}"
        if unit_label
        else ""
    )

    return f"{value:,.0f}{suffix}"


def make_legend(
    palette: str,
    minimum: float,
    maximum: float,
    unit_label: str,
) -> str:
    """
    Generate an HTML color-gradient legend.
    """
    color_map = mpl.colormaps[palette]

    stops = []

    for index in range(9):
        fraction = index / 8

        red, green, blue, _ = (
            np.array(
                color_map(fraction)
            )
            * 255
        ).astype(int)

        stops.append(
            f"rgb({red},{green},{blue}) "
            f"{fraction * 100:.0f}%"
        )

    gradient = ", ".join(stops)

    unit_label = unit_label.strip()

    suffix = (
        f" {unit_label}"
        if unit_label
        else ""
    )

    return f"""
    <div style="
        max-width: 520px;
        margin: 0.15rem 0 0.8rem 0;
    ">
        <div style="
            height: 13px;
            border-radius: 3px;
            background: linear-gradient(
                90deg,
                {gradient}
            );
        ">
        </div>

        <div style="
            display: flex;
            justify-content: space-between;
            font-size: 0.82rem;
            margin-top: 3px;
        ">
            <span>{minimum:,.0f}{suffix}</span>
            <span>{maximum:,.0f}{suffix}</span>
        </div>

        <div style="
            font-size: 0.75rem;
            color: #666;
        ">
            Values above the upper legend limit use the darkest color.
        </div>
    </div>
    """


def initial_view(
    grid: gpd.GeoDataFrame,
) -> pdk.ViewState:
    """
    Calculate an initial map center and zoom from the dataset bounds.
    """
    xmin, ymin, xmax, ymax = grid.total_bounds

    longitude = (xmin + xmax) / 2
    latitude = (ymin + ymax) / 2

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
    "China grid-cell GDP per capita"
)

st.caption(
    "Interactive grid-cell choropleth for 2012–2022. "
    "ADM1 boundaries are shown as thicker dark outlines."
)

# This environment variable is optional. It allows the GitHub URL to be
# replaced without editing the source code.
configured_url = os.environ.get(
    "CHINA_GDPPC_GPKG_URL",
    DATA_URL,
)

try:
    geopackage_path = get_geopackage(
        url=configured_url,
    )

    grid_gdf, adm1_gdf = load_data(
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

    scaling = st.radio(
        "Color scaling",
        options=[
            "Linear",
            "Logarithmic",
        ],
        index=0,
        help=(
            "Logarithmic scaling can reveal variation "
            "among lower-valued cells when the "
            "distribution is highly skewed."
        ),
    )

    upper_percentile = st.slider(
        "Upper color limit (percentile)",
        min_value=90.0,
        max_value=100.0,
        value=99.0,
        step=0.5,
        help=(
            "Only the color scale is clipped. "
            "Tooltip values remain unchanged."
        ),
    )

    opacity = st.slider(
        "Grid opacity",
        min_value=80,
        max_value=255,
        value=205,
        step=5,
    )

    unit_label = st.text_input(
        "Unit label",
        value="dataset units",
        help=(
            "The source fields do not encode a currency "
            "or price-base label, so the app does not "
            "assume one."
        ),
    )

    st.divider()

    show_grid_lines = st.checkbox(
        "Show thin grid borders",
        value=True,
    )

    adm1_width = st.slider(
        "ADM1 border width",
        min_value=1.0,
        max_value=6.0,
        value=3.0,
        step=0.5,
    )


# ---------------------------------------------------------------------
# Prepare selected year
# ---------------------------------------------------------------------

value_column = YEAR_COLUMNS[year]

values = grid_gdf[value_column]

fill_colors, legend_min, legend_max = colorize(
    values=values,
    palette=palette,
    scaling=scaling,
    upper_percentile=upper_percentile,
    opacity=opacity,
)

map_gdf = grid_gdf[
    [
        CELL_ID,
        ADM1_ID,
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
    lambda value: format_value(
        value,
        unit_label,
    )
)

map_gdf["year"] = str(year)


# ---------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------

metric_columns = st.columns(4)

metric_columns[0].metric(
    "Year",
    str(year),
)

metric_columns[1].metric(
    "Grid cells",
    f"{len(map_gdf):,}",
)

metric_columns[2].metric(
    "Median",
    format_value(
        float(values.median()),
        unit_label,
    ),
)

metric_columns[3].metric(
    "95th percentile",
    format_value(
        float(values.quantile(0.95)),
        unit_label,
    ),
)


# ---------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------

st.markdown(
    make_legend(
        palette=palette,
        minimum=legend_min,
        maximum=legend_max,
        unit_label=unit_label,
    ),
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Pydeck layers
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
    get_line_color=[
        255,
        255,
        255,
        70,
    ],
    get_line_width=0.35,
    line_width_units="pixels",
    line_width_min_pixels=(
        0.2
        if show_grid_lines
        else 0
    ),
)

adm1_layer = pdk.Layer(
    "GeoJsonLayer",
    data=adm1_gdf,
    id="adm1-boundaries",
    pickable=False,
    filled=False,
    stroked=True,
    get_line_color=[
        20,
        20,
        20,
        245,
    ],
    get_line_width=adm1_width,
    line_width_units="pixels",
    line_width_min_pixels=adm1_width,
)


# ---------------------------------------------------------------------
# Render map
# ---------------------------------------------------------------------

deck = pdk.Deck(
    layers=[
        grid_layer,
        adm1_layer,
    ],
    initial_view_state=initial_view(
        grid_gdf
    ),
    map_provider="carto",
    map_style=pdk.map_styles.CARTO_LIGHT,
    tooltip={
        "html": (
            "<b>GDP per capita ({year})</b>: "
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
    width="stretch",
    height=720,
)


# ---------------------------------------------------------------------
# Dataset notes
# ---------------------------------------------------------------------

with st.expander(
    "Dataset and map notes"
):
    st.markdown(
        f"""
- GeoPackage layer: `{LAYER_NAME}`
- CRS: `{grid_gdf.crs}`
- Coverage: `{min(YEAR_COLUMNS)}–{max(YEAR_COLUMNS)}`
- ADM1 identifier: `{ADM1_ID}`
- ADM1 label: `{ADM1_NAME}`
- GDP-per-capita field shown: `{value_column}`
- Grid-cell features: `{len(grid_gdf):,}`
- ADM1 regions: `{adm1_gdf[ADM1_ID].nunique():,}`
- Cached file location: `{geopackage_path}`
- [Source GeoPackage on GitHub](
  https://github.com/cesarechevarria/scratch/blob/ea6d50d07843346ec5ea51083eedab743dc9de88/china_grid_cellls_cleaned.gpkg
)

The upper-percentile control clips only the color domain to
prevent extreme observations from flattening most of the map's
visual variation. Hover tooltips continue to show the original
values.
        """
    )
