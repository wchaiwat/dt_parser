import streamlit as st
import polars as pl
import plotly.express as px
import python_calamine
import os
import io
import json

st.set_page_config(page_title="Universal 5G Analyzer", layout="wide")
st.title("🛰️ Universal RF Drive Test Analyzer")
st.info("Upload CSV/XLSX files from Actix, Nemo, or DingLi to generate reports and color-coded maps automatically.")

uploaded_files = st.file_uploader("Upload CSV or XLSX files", type=['csv', 'xlsx'], accept_multiple_files=True)

CARTO_LAYERS = [{
    "below": "traces",
    "sourcetype": "raster",
    "source": [
        "https://a.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
        "https://b.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
        "https://c.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
        "https://d.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
    ]
}]

@st.cache_data
def get_sheet_names(file_name: str, file_bytes: bytes) -> list[str]:
    workbook = python_calamine.CalamineWorkbook.from_filelike(io.BytesIO(file_bytes))
    return workbook.sheet_names

@st.cache_data
def read_file(file_name: str, file_bytes: bytes, sheet_name=None) -> pl.DataFrame:
    if file_name.lower().endswith(".xlsx"):
        return pl.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, engine="calamine")
    else:
        try:
            return pl.read_csv(io.BytesIO(file_bytes), encoding="utf8")
        except Exception:
            return pl.read_csv(io.BytesIO(file_bytes), encoding="latin1")

@st.cache_data
def get_preview(file_name: str, file_bytes: bytes, sheet_name=None) -> pl.DataFrame:
    return read_file(file_name, file_bytes, sheet_name=sheet_name).head(3)

@st.cache_data
def build_combined(
    file_order: tuple,
    file_bytes_map: dict,
    col_selections: dict,
    sheet_selections: dict
) -> tuple:
    all_frames     = {}
    all_groups     = set()
    available_kpis = []

    for fname in file_order:
        fbytes = file_bytes_map[fname]
        cols   = col_selections.get(fname)
        sheet  = sheet_selections.get(fname)
        if not cols:
            continue
        try:
            raw_df = read_file(fname, fbytes, sheet_name=sheet)
            keep   = list({cols["lat"], cols["lon"], cols["group_col"]} | set(cols["numeric_cols"]))
            raw_df = raw_df.select([c for c in keep if c in raw_df.columns])
            raw_df = raw_df.rename({cols["lat"]: "LAT", cols["lon"]: "LON"})
            if cols["group_col"] != "GROUP_ID":
                raw_df = raw_df.rename({cols["group_col"]: "GROUP_ID"})
            raw_df = raw_df.with_columns(pl.col("GROUP_ID").cast(pl.Utf8))
            all_frames[fname] = raw_df
            all_groups.update(raw_df["GROUP_ID"].unique().to_list())
            if not available_kpis:
                available_kpis = cols["numeric_cols"]
        except Exception as e:
            st.error(f"Error loading {fname}: {e}")

    all_groups = sorted([g for g in all_groups if g is not None])
    combined   = pl.concat(list(all_frames.values()), how="diagonal") if all_frames else pl.DataFrame()
    return combined, all_groups, available_kpis

def compute_indicators(series: pl.Series, indicators: list[str]) -> dict:
    result = {}
    s = series.drop_nulls()
    for ind in indicators:
        if   ind == "Mean"  : result["Mean"]   = round(s.mean(), 2)
        elif ind == "Median": result["Median"] = round(s.median(), 2)
        elif ind == "Min"   : result["Min"]    = round(s.min(), 2)
        elif ind == "Max"   : result["Max"]    = round(s.max(), 2)
        elif ind == "P5"    : result["P5"]     = round(s.quantile(0.05), 2)
        elif ind == "P25"   : result["P25"]    = round(s.quantile(0.25), 2)
        elif ind == "P75"   : result["P75"]    = round(s.quantile(0.75), 2)
        elif ind == "P95"   : result["P95"]    = round(s.quantile(0.95), 2)
        elif ind == "Std"   : result["Std"]    = round(s.std(), 2)
        elif ind == "Count" : result["Count"]  = int(s.len())
    return result

def make_map_html(df, kpis: list[str], title: str) -> str:
    import json
    pdf = df.to_pandas()

    if "LAT" not in pdf.columns or "LON" not in pdf.columns:
        return "<html><body><h2>Error: LAT/LON columns missing</h2></body></html>"

    for col in ["LAT", "LON"] + kpis:
        if col in pdf.columns:
            pdf[col] = pdf[col].round(6)

    data_json       = pdf[["LAT", "LON", "GROUP_ID"] + [k for k in kpis if k in pdf.columns]].to_json(orient="records")
    kpis_json       = json.dumps(kpis)
    groups          = sorted(pdf["GROUP_ID"].dropna().unique().tolist())
    groups_json     = json.dumps(groups)
    kpi_ranges      = {k: [float(pdf[k].min()), float(pdf[k].max())]
                       for k in kpis if k in pdf.columns}
    kpi_ranges_json = json.dumps(kpi_ranges)
    center_lat      = float(pdf["LAT"].mean())
    center_lon      = float(pdf["LON"].mean())
    first_kpi       = kpis[0] if kpis else ""

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html, body {{ width: 100%; height: 100%; overflow: hidden; font-family: Arial, sans-serif; }}
    #map {{ width: 100vw; height: 100vh; }}

    #controls {{
      position: fixed;
      top: 12px; left: 12px;
      z-index: 9999;
      background: rgba(255,255,255,0.97);
      padding: 12px 16px;
      border-radius: 10px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.25);
      min-width: 240px;
      max-height: 90vh;
      overflow-y: auto;
    }}
    #controls h3 {{
      margin: 0 0 10px 0; font-size: 14px; color: #333;
      border-bottom: 1px solid #ddd; padding-bottom: 6px;
    }}
    #controls label {{
      font-size: 12px; font-weight: bold; color: #555;
      display: block; margin: 8px 0 4px 0;
    }}
    #controls select {{
      width: 100%; padding: 5px; border-radius: 5px;
      border: 1px solid #ccc; font-size: 12px;
    }}
    .group-item {{
      display: flex; align-items: center; gap: 6px;
      margin: 3px 0; font-size: 12px; color: #333; cursor: pointer;
    }}
    .group-item input {{ cursor: pointer; accent-color: #0066cc; }}
    #toggle-btn {{
      font-size: 11px; color: #0066cc; cursor: pointer;
      text-decoration: underline; background: none;
      border: none; padding: 0; margin-bottom: 6px; display: block;
    }}
    #point-count {{
      font-size: 11px; color: #888;
      margin-top: 8px; border-top: 1px solid #eee; padding-top: 6px;
    }}

    /* Colorbar */
    #colorbar {{
      position: fixed;
      right: 20px; top: 50%;
      transform: translateY(-50%);
      z-index: 9999;
      background: rgba(255,255,255,0.92);
      padding: 10px 8px;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.2);
      display: flex; flex-direction: column; align-items: center; gap: 4px;
    }}
    #colorbar-title {{
      font-size: 10px; font-weight: bold; color: #333;
      text-align: center; max-width: 80px;
      word-break: break-all; margin-bottom: 4px;
    }}
    #colorbar-gradient {{
      width: 18px; height: 200px;
      background: linear-gradient(to bottom,
        rgb(26,150,65), rgb(255,255,191), rgb(253,174,97), rgb(215,25,28));
      border: 1px solid #ccc; border-radius: 3px;
    }}
    #colorbar-labels {{
      display: flex; flex-direction: column;
      justify-content: space-between;
      height: 200px; font-size: 10px; color: #444;
      position: absolute; right: 34px; top: 42px;
    }}

    #title-overlay {{
      position: fixed; top: 12px; left: 50%;
      transform: translateX(-50%);
      z-index: 9998;
      background: rgba(255,255,255,0.92);
      padding: 5px 20px; border-radius: 6px;
      font-size: 14px; font-weight: bold;
      box-shadow: 0 2px 6px rgba(0,0,0,0.15);
      pointer-events: none; white-space: nowrap;
    }}
  </style>
</head>
<body>

<div id="map"></div>

<div id="controls">
  <h3>🗺️ Map Controls</h3>
  <label>KPI</label>
  <select id="kpi-select" onchange="updateMap()"></select>
  <label style="margin-top:10px;">Groups / Files</label>
  <button id="toggle-btn" onclick="toggleAll()">Select All / None</button>
  <div id="group-checkboxes"></div>
  <div id="point-count"></div>
</div>

<div id="title-overlay">{title}</div>

<div id="colorbar">
  <div id="colorbar-title">{first_kpi}</div>
  <div style="position:relative;">
    <div id="colorbar-gradient"></div>
    <div id="colorbar-labels">
      <span id="cb-max">0</span>
      <span id="cb-mid">0</span>
      <span id="cb-min">0</span>
    </div>
  </div>
</div>

<script>
const ALL_DATA      = {data_json};
const ALL_KPIS      = {kpis_json};
const ALL_GROUPS    = {groups_json};
const KPI_RANGES    = {kpi_ranges_json};
const CENTER_LAT    = {center_lat};
const CENTER_LON    = {center_lon};

// ── Init Leaflet map ─────────────────────────────────────────────────────────
const map = L.map("map").setView([CENTER_LAT, CENTER_LON], 13);
L.tileLayer("https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}.png", {{
    attribution: "© CartoDB",
    subdomains: "abcd",
    maxZoom: 19
}}).addTo(map);

// ── Layer group for dots ─────────────────────────────────────────────────────
let dotLayer = L.layerGroup().addTo(map);

// ── KPI dropdown ─────────────────────────────────────────────────────────────
const kpiSelect = document.getElementById("kpi-select");
ALL_KPIS.forEach(k => {{
    const opt = document.createElement("option");
    opt.value = k; opt.text = k;
    kpiSelect.appendChild(opt);
}});

// ── Group checkboxes ─────────────────────────────────────────────────────────
const cbContainer = document.getElementById("group-checkboxes");
ALL_GROUPS.forEach(g => {{
    const safeId = "cb_" + g.replace(/[^a-zA-Z0-9]/g, "_");
    const div    = document.createElement("div");
    div.className = "group-item";
    div.innerHTML = `<input type="checkbox" id="${{safeId}}" value="${{g}}" checked
                      onchange="updateMap()">
                     <span onclick="document.getElementById('${{safeId}}').click()">${{g}}</span>`;
    cbContainer.appendChild(div);
}});

function getSelectedGroups() {{
    return ALL_GROUPS.filter(g => {{
        const el = document.getElementById("cb_" + g.replace(/[^a-zA-Z0-9]/g, "_"));
        return el && el.checked;
    }});
}}

let allSelected = true;
function toggleAll() {{
    allSelected = !allSelected;
    ALL_GROUPS.forEach(g => {{
        const el = document.getElementById("cb_" + g.replace(/[^a-zA-Z0-9]/g, "_"));
        if (el) el.checked = allSelected;
    }});
    updateMap();
}}

// ── Color scale ──────────────────────────────────────────────────────────────
function getColor(val, vmin, vmax) {{
    const t = Math.max(0, Math.min(1, (val - vmin) / (vmax - vmin || 1)));
    const stops = [
        [0.0,  [215, 25,  28 ]],
        [0.4,  [253, 174, 97 ]],
        [0.7,  [255, 255, 191]],
        [1.0,  [26,  150, 65 ]]
    ];
    for (let i = 0; i < stops.length - 1; i++) {{
        const [t0, c0] = stops[i];
        const [t1, c1] = stops[i+1];
        if (t >= t0 && t <= t1) {{
            const lt  = (t - t0) / (t1 - t0);
            const rgb = c0.map((v,j) => Math.round(v + lt*(c1[j]-v)));
            return `rgb(${{rgb[0]}},${{rgb[1]}},${{rgb[2]}})`;
        }}
    }}
    return "rgb(26,150,65)";
}}

// ── Update map ───────────────────────────────────────────────────────────────
function updateMap() {{
    const kpi            = kpiSelect.value;
    const selectedGroups = getSelectedGroups();

    const filtered = ALL_DATA.filter(d =>
        selectedGroups.includes(String(d.GROUP_ID)) &&
        d[kpi] !== null && d[kpi] !== undefined && !isNaN(d[kpi]) &&
        d.LAT !== null && d.LON !== null
    );

    document.getElementById("point-count").textContent =
        `Showing ${{filtered.length.toLocaleString()}} points`;

    const [vmin, vmax] = KPI_RANGES[kpi] || [-140, 0];
    const vmid = ((vmin + vmax) / 2).toFixed(1);

    // Update colorbar
    document.getElementById("colorbar-title").textContent = kpi;
    document.getElementById("cb-max").textContent = vmax.toFixed(1);
    document.getElementById("cb-mid").textContent = vmid;
    document.getElementById("cb-min").textContent = vmin.toFixed(1);

    // Clear and redraw dots
    dotLayer.clearLayers();

    filtered.forEach(d => {{
        const color = getColor(d[kpi], vmin, vmax);
        const val   = typeof d[kpi] === "number" ? d[kpi].toFixed(2) : d[kpi];
        L.circleMarker([d.LAT, d.LON], {{
            radius: 5,
            fillColor: color,
            color: "transparent",
            fillOpacity: 0.9,
            weight: 0
        }}).bindTooltip(`<b>${{d.GROUP_ID}}</b><br>${{kpi}}: ${{val}}`,
            {{sticky: true, opacity: 0.9}})
          .addTo(dotLayer);
    }});
}}

updateMap();
</script>
</body>
</html>"""

# ── Session state ────────────────────────────────────────────────────────────
if "stat_group_count" not in st.session_state:
    st.session_state.stat_group_count = 1

if uploaded_files:
    file_bytes_map = {f.name: f.read() for f in uploaded_files}


    sheet_selections = {}
    col_selections   = {}

    # ── Step 1: File Configuration ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Step 1 · File Configuration")

    for f in uploaded_files:
        fbytes = file_bytes_map[f.name]
        with st.expander(f"⚙️ Configure: {f.name}", expanded=True):
            if f.name.lower().endswith(".xlsx"):
                sheet_names    = get_sheet_names(f.name, fbytes)
                selected_sheet = st.selectbox(
                    "Select Sheet", options=sheet_names, key=f"sheet_{f.name}"
                )
                sheet_selections[f.name] = selected_sheet
            else:
                sheet_selections[f.name] = None
                selected_sheet           = None

            try:
                full_df      = read_file(f.name, fbytes, sheet_name=selected_sheet)
                preview_df   = get_preview(f.name, fbytes, sheet_name=selected_sheet)
                all_cols     = full_df.columns
                numeric_cols = [c for c in all_cols if full_df[c].dtype in (
                    pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64)]

                st.caption(f"Found {len(all_cols)} columns · {len(full_df)} rows")
                c1, c2, c3 = st.columns(3)
                with c1:
                    lat = st.selectbox("📍 Latitude column", all_cols,
                        index=next((i for i,c in enumerate(all_cols) if "LAT" in c.upper()), 0),
                        key=f"lat_{f.name}")
                with c2:
                    lon = st.selectbox("📍 Longitude column", all_cols,
                        index=next((i for i,c in enumerate(all_cols) if "LON" in c.upper()), 0),
                        key=f"lon_{f.name}")
                with c3:
                    default_grp = next(
                        (i for i,c in enumerate(all_cols)
                         if any(k in c.upper() for k in ["FILENAME","FILE_NAME","ROUTE","SOURCE"])), 0)
                    group_col = st.selectbox(
                        "🗂️ Group / FileName column", options=all_cols,
                        index=default_grp,
                        help="Column whose unique values will be used as selectable groups",
                        key=f"grp_{f.name}")

                col_selections[f.name] = {
                    "lat": lat, "lon": lon,
                    "group_col": group_col,
                    "numeric_cols": numeric_cols,
                }
                st.dataframe(preview_df.to_pandas(), use_container_width=True)

            except Exception as e:
                st.error(f"Could not read file: {e}")

    # ── Build combined dataframe (cached) ────────────────────────────────────
    combined, all_groups, available_kpis = build_combined(
        file_order       = tuple(f.name for f in uploaded_files),
        file_bytes_map   = file_bytes_map,
        col_selections   = col_selections,
        sheet_selections = sheet_selections
    )

    if not all_groups or not available_kpis:
        st.warning("Please configure at least one file to continue.")
        st.stop()

    # ── Step 2: Plot Settings ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🗺️ Step 2 · Map Plot Settings")

    pc1, pc2 = st.columns(2)
    with pc1:
        plot_kpi = st.selectbox(
            "KPI to plot on map (single selection)",
            options=available_kpis,
            index=next((i for i,c in enumerate(available_kpis) if "RSRP" in c.upper()), 0)
        )
    with pc2:
        plot_groups = st.multiselect(
            "Groups to display on map",
            options=all_groups, default=all_groups
        )
    plot_range = st.slider("Color scale range for selected KPI", -140, 0, (-120, -70))

    # ── Step 3: Statistics ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Step 3 · Statistics Settings")

    st_c1, st_c2 = st.columns(2)
    with st_c1:
        stat_kpis = st.multiselect(
            "KPIs to calculate statistics for",
            options=available_kpis,
            default=available_kpis[:min(4, len(available_kpis))]
        )
    with st_c2:
        stat_indicators = st.multiselect(
            "Statistical indicators to show",
            options=["Mean","Median","Min","Max","P5","P25","P75","P95","Std","Count"],
            default=["Mean","P95","Min","Max"]
        )

    st.markdown("##### Define Statistic Groups")
    st.caption("Each group can contain one or more route/file names. Stats will be calculated per group per KPI.")

    bc1, bc2 = st.columns([1,1])
    with bc1:
        if st.button("➕ Add Group"):
            st.session_state.stat_group_count += 1
    with bc2:
        if st.button("➖ Remove Group") and st.session_state.stat_group_count > 1:
            st.session_state.stat_group_count -= 1

    stat_groups_def = []
    for i in range(st.session_state.stat_group_count):
        with st.expander(f"📂 Group {i+1}", expanded=True):
            gc1, gc2 = st.columns([1,3])
            with gc1:
                grp_name = st.text_input(
                    "Group label", value=f"Group {i+1}", key=f"stat_grp_name_{i}"
                )
            with gc2:
                grp_ids = st.multiselect(
                    "Assign route / file names to this group",
                    options=all_groups, default=[],
                    key=f"stat_grp_ids_{i}",
                    help="Select one or more values from the GROUP_ID column"
                )
            stat_groups_def.append({"name": grp_name, "ids": grp_ids})

    # ── Step 4: HTML Export ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🌐 Step 4 · HTML Export Settings")

    hc1, hc2 = st.columns(2)
    with hc1:
        html_kpis = st.multiselect(
            "KPIs to include in interactive HTML map",
            options=available_kpis,
            default=available_kpis[:min(4, len(available_kpis))]
        )
    with hc2:
        html_groups = st.multiselect(
            "Groups to include in HTML map",
            options=all_groups, default=all_groups
        )
    if html_kpis:
        st.info(f"Will generate 1 interactive HTML with {len(html_kpis)} KPIs and KPI/Group filters.")

    # ── Run Analysis ─────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("📊 Run Analysis", use_container_width=True, type="primary"):

        # ── Statistics ───────────────────────────────────────────────────────
        st.subheader("📈 Statistics Results")
        active_groups = [g for g in stat_groups_def if g["ids"]]

        if not active_groups:
            st.warning("No groups defined. Please assign route/file names to at least one group.")
        elif not stat_kpis:
            st.warning("Please select at least one KPI for statistics.")
        else:
            for kpi in stat_kpis:
                if kpi not in combined.columns:
                    st.warning(f"KPI '{kpi}' not found in data — skipping.")
                    continue
                st.markdown(f"#### 📶 {kpi}")
                rows = []
                for grp in active_groups:
                    subset = combined.filter(pl.col("GROUP_ID").is_in(grp["ids"]))
                    if subset.is_empty():
                        continue
                    row = {"Group": grp["name"], "Included IDs": ", ".join(grp["ids"])}
                    row.update(compute_indicators(subset[kpi], stat_indicators))
                    rows.append(row)

                all_ids    = [gid for g in active_groups for gid in g["ids"]]
                all_subset = combined.filter(pl.col("GROUP_ID").is_in(all_ids))
                if not all_subset.is_empty():
                    row = {"Group": "★ ALL", "Included IDs": f"{len(all_ids)} routes"}
                    row.update(compute_indicators(all_subset[kpi], stat_indicators))
                    rows.append(row)

                st.dataframe(
                    pl.DataFrame(rows).to_pandas(),
                    use_container_width=True, hide_index=True
                )

        # ── In-app Map ───────────────────────────────────────────────────────
        st.subheader(f"🗺️ Map · {plot_kpi}")
        map_df = combined.filter(
            pl.col("GROUP_ID").is_in(plot_groups)
        ).drop_nulls(subset=[plot_kpi])

        if not map_df.is_empty():
            map_pdf = map_df.to_pandas()
            fig = px.scatter_map(
                map_pdf,
                lat="LAT", lon="LON", color=plot_kpi,
                color_continuous_scale=[(0,"red"),(0.4,"orange"),(0.7,"yellow"),(1,"green")],
                range_color=list(plot_range),
                hover_data=["GROUP_ID", plot_kpi],
                zoom=13, height=600,
                center=dict(lat=map_pdf["LAT"].mean(), lon=map_pdf["LON"].mean())
            )
            fig.update_layout(
                map=dict(style="white-bg", layers=CARTO_LAYERS),
                margin={"r":0,"t":0,"l":0,"b":0}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No data found for '{plot_kpi}' in the selected groups.")

        # ── HTML Export ──────────────────────────────────────────────────────
        st.subheader("🌐 HTML Export")
        html_df = combined.filter(pl.col("GROUP_ID").is_in(html_groups))

        if not html_df.is_empty() and html_kpis:
            output_dir = "html_output"
            os.makedirs(output_dir, exist_ok=True)

            # Drop nulls only for columns that exist
            valid_kpis  = [k for k in html_kpis if k in html_df.columns]
            export_df   = html_df.drop_nulls(subset=["LAT", "LON"])

            html_str = make_map_html(
                export_df,
                kpis=valid_kpis,
                title="RF Drive Test · Interactive Map"
            )
            path = os.path.join(output_dir, "RF_Map_Interactive.html")
            with open(path, "w", encoding="utf-8") as fp:
                fp.write(html_str)
            with open(path, "rb") as fp:
                st.download_button(
                    label="⬇️ Download Interactive Map HTML",
                    data=fp.read(),
                    file_name="RF_Map_Interactive.html",
                    mime="text/html",
                    key="dl_interactive"
                )
            st.success(f"Interactive map with {len(valid_kpis)} KPIs is ready to download.")
        else:
            st.warning("No data or KPIs selected for HTML export.")

        st.success("🎉 Analysis complete!")