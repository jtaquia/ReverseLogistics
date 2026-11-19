import io
import json
import html
import math
import time
import traceback
import numbers
from collections import defaultdict

import googlemaps
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from vrpy import VehicleRoutingProblem

st.set_page_config(page_title="Ruteo de residuos médicos | VRPy", layout="wide")
st.title("Optimización de recolección de residuos médicos · VRPy + Google Maps")
st.caption("Flota heterogénea por camión, distancias viales de Google y métricas operativas. Cada vehículo sale y regresa al mismo depósito.")

PICKUP_COLS = ["ID_CentroSalud", "latitud", "longitud", "PesoDesperdicio"]
DEPOT_COLS = ["ID_Deposito", "latitud", "longitud", "NumCamiones"]
CAPACITIES = [0.5, 1.0, 1.5]


def read_upload(upload):
    if upload.name.lower().endswith(".csv"):
        raw = upload.getvalue()
        try:
            return pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding="latin-1")
    return pd.read_excel(upload)


def check_data(df, columns, label):
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{label}: faltan columnas: {', '.join(sorted(missing))}")
    df = df.copy()
    for c in ["latitud", "longitud"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[["latitud", "longitud"]].isna().any().any() or not df.latitud.between(-90, 90).all() or not df.longitud.between(-180, 180).all():
        raise ValueError(f"{label}: coordenadas inválidas o vacías")
    return df


def haversine(a, b):
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    x = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 6371.0088 * 2 * math.asin(min(1, math.sqrt(x)))


def build_matrix(points, key, use_google):
    coords = [(float(x), float(y)) for x, y in zip(points.latitud, points.longitud)]
    n = len(coords)
    matrix = np.full((n, n), np.inf)
    np.fill_diagonal(matrix, 0.0)
    if not use_google:
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i, j] = haversine(coords[i], coords[j])
        return matrix
    client = googlemaps.Client(key=key, timeout=30)
    # 10x10 = 100 elementos por solicitud; comprobar límites y tarifas de la API de Google.
    for i in range(0, n, 10):
        for j in range(0, n, 10):
            origins, destinations = coords[i:i+10], coords[j:j+10]
            result = client.distance_matrix(origins, destinations, mode="driving", units="metric")
            if result.get("status") != "OK":
                raise RuntimeError(f"Google Distance Matrix: {result.get('status')}: {result.get('error_message', '')}")
            for ii, row in enumerate(result["rows"]):
                for jj, cell in enumerate(row["elements"]):
                    if cell.get("status") == "OK":
                        matrix[i+ii, j+jj] = cell["distance"]["value"] / 1000.0
    return matrix


def assign_to_depots(pickups, depots, matrix, n_depots):
    # Asignación inicial heurística: depósito con menor distancia vial de ida.
    # No constituye optimización conjunta de localización-asignación-ruteo.
    assignments = {}
    for p in range(len(pickups)):
        choices = [(matrix[d, n_depots+p], d) for d in range(n_depots) if np.isfinite(matrix[d, n_depots+p])]
        if not choices:
            raise ValueError(f"Centro {pickups.iloc[p].ID_CentroSalud}: ningún depósito accesible")
        assignments[p] = min(choices)[1]
    return assignments


def validar_costos_vrpy(G, capacidades_kg):
    """Asegura listas de costos finitos, una entrada por tipo de vehículo."""
    n_types = len(capacidades_kg)
    if n_types == 0:
        raise ValueError("No hay capacidades de vehículos configuradas")
    for u, v, data in G.edges(data=True):
        cost = data.get("cost")
        if isinstance(cost, numbers.Real):
            cost = [float(cost)] * n_types
        elif isinstance(cost, (list, tuple, np.ndarray)):
            cost = list(cost)
            if len(cost) == 1 and n_types > 1:
                cost = cost * n_types
        else:
            raise TypeError(f"Arista {u!r}->{v!r}: cost inválido {cost!r}")
        if len(cost) != n_types:
            raise ValueError(f"Arista {u!r}->{v!r}: {len(cost)} costos, se requieren {n_types}")
        if not all(isinstance(c, numbers.Real) and np.isfinite(c) and c >= 0 for c in cost):
            raise ValueError(f"Arista {u!r}->{v!r}: costos no finitos o negativos: {cost!r}")
        data["cost"] = [float(c) for c in cost]
    for u, v, data in G.edges(data=True):
        assert isinstance(data["cost"], list) and len(data["cost"]) == n_types, (u, v, data)
    return G


def solve_one_depot(depot_index, pickup_indices, pickups, matrix, n_depots, fleet, limit):
    """VRPy mixed_fleet: one capacity per individual truck; one route per truck."""
    if not pickup_indices:
        return [], 0.0
    capacities = [int(round(float(t["Capacidad_tn"]) * 1000)) for t in fleet]
    if not capacities:
        raise ValueError("El depósito no tiene camiones configurados")
    # VRPy mixed_fleet requiere un costo por tipo de vehículo en CADA arista.
    # Agrupamos capacidades idénticas y limitamos su número de vehículos.
    counts = [sum(c == target for c in capacities) for target in (500, 1000, 1500)]
    active = [(cap, count) for cap, count in zip((500, 1000, 1500), counts) if count]
    n_types = len(active)

    def edge_cost(distance_km):
        # Mismo costo (distancia) para todos los tipos: minimiza km.
        # Para minimizar costos monetarios, usar un costo específico por tipo.
        return [float(distance_km)] * n_types

    G = nx.DiGraph()
    G.add_node("Source")
    G.add_node("Sink")
    for p in pickup_indices:
        demand = int(round(float(pickups.iloc[p].PesoDesperdicio)))
        if demand <= 0:
            raise ValueError(f"Centro {pickups.iloc[p].ID_CentroSalud}: demanda debe ser positiva en kg")
        if demand > max(capacities):
            raise ValueError(f"Centro {pickups.iloc[p].ID_CentroSalud}: demanda {demand} kg supera la mayor capacidad del depósito")
        G.add_node(f"P{p}", demand=demand)
    for p in pickup_indices:
        a = n_depots + p
        if np.isfinite(matrix[depot_index, a]):
            G.add_edge("Source", f"P{p}", cost=edge_cost(matrix[depot_index, a]))
        if np.isfinite(matrix[a, depot_index]):
            G.add_edge(f"P{p}", "Sink", cost=edge_cost(matrix[a, depot_index]))
    for p in pickup_indices:
        for q in pickup_indices:
            if p != q and np.isfinite(matrix[n_depots+p, n_depots+q]):
                G.add_edge(f"P{p}", f"P{q}", cost=edge_cost(matrix[n_depots+p, n_depots+q]))
    G.add_edge("Source", "Sink", cost=edge_cost(0.0))
    # Validar todos los arcos antes de entregar el grafo a VRPy.
    G = validar_costos_vrpy(G, [cap for cap, _ in active])
    # Evitar que VRPy modifique el grafo original durante su preprocesamiento.
    # Un tipo por capacidad: la cantidad de camiones de cada tipo limita las rutas.
    problem = VehicleRoutingProblem(
        G.copy(), mixed_fleet=True,
        load_capacity=[int(cap) for cap, _ in active],
        num_vehicles=[int(count) for _, count in active]
    )
    t0 = time.perf_counter()
    problem.solve(time_limit=int(limit))
    elapsed = time.perf_counter() - t0
    best = problem.best_routes
    if not best:
        raise RuntimeError("VRPy no encontró rutas factibles para la flota disponible")
    types = getattr(problem, "best_routes_type", None)
    if types is None:
        raise RuntimeError("La versión instalada de VRPy no informa best_routes_type; se requiere para asignar capacidades de manera segura")
    # VRPy indexa tipos de vehículo desde 0 (algunas versiones usan 1).
    raw_types = list(types.values()) if isinstance(types, dict) else list(types)
    zero_based = 0 in raw_types
    routes = []
    visited = []
    for route_id, nodes in best.items():
        seq = [int(str(node)[1:]) for node in nodes if str(node).startswith("P")]
        if not seq:
            continue
        type_idx = types[route_id] if isinstance(types, dict) else types[route_id]
        type_idx = int(type_idx) if zero_based else int(type_idx) - 1
        if type_idx < 0 or type_idx >= len(active):
            raise RuntimeError("Índice de tipo de vehículo inesperado en VRPy")
        cap_kg = active[type_idx][0]
        visited.extend(seq)
        path = [depot_index] + [n_depots+p for p in seq] + [depot_index]
        distance = sum(float(matrix[a, b]) for a, b in zip(path[:-1], path[1:]))
        load = sum(float(pickups.iloc[p].PesoDesperdicio) for p in seq)
        if load > cap_kg + 1e-5:
            raise RuntimeError("Ruta excede la capacidad del vehículo asignado")
        routes.append({"puntos": seq, "km": distance, "kg": load,
                       "capacidad_kg": cap_kg, "depot_index": depot_index})
    if sorted(visited) != sorted(pickup_indices):
        raise RuntimeError("La solución no cubre todos los centros asignados")
    for cap, count in active:
        if sum(r["capacidad_kg"] == cap for r in routes) > count:
            raise RuntimeError(f"La solución utiliza más camiones de {cap} kg que los disponibles")
    # Asignación identificable de cada ruta a un camión específico.
    by_capacity = defaultdict(list)
    for truck in fleet:
        by_capacity[round(float(truck["Capacidad_tn"])*1000)].append(str(truck["ID_Camion"]))
    for route in routes:
        route["ID_Camion"] = by_capacity[round(route["capacidad_kg"])].pop(0)
    return routes, elapsed


def compute_scenario(name, fleet_df, pickups, depots, matrix, assignment, fixed_cost, km_cost, co2_factor, limit):
    n_depots = len(depots)
    all_routes = []
    total_time = 0.0
    for d in range(n_depots):
        indices = [p for p, dd in assignment.items() if dd == d]
        fleet = fleet_df.loc[fleet_df.ID_Deposito.astype(str) == str(depots.iloc[d].ID_Deposito)].to_dict("records")
        routes, elapsed = solve_one_depot(d, indices, pickups, matrix, n_depots, fleet, limit)
        all_routes.extend(routes)
        total_time += elapsed
    total_km = sum(r["km"] for r in all_routes)
    total_kg = sum(r["kg"] for r in all_routes)
    trips = len(all_routes)
    available = sum(r["capacidad_kg"] for r in all_routes)
    summary = {"Escenario": name, "Viajes": trips, "Distancia_km": round(total_km, 3),
               "Peso_total_kg": round(total_kg, 2), "Utilizacion_pct": round(100*total_kg/available, 2) if available else 0,
               "Km_por_kg": round(total_km/total_kg, 5) if total_kg else 0,
               "Costo_estimado": round(trips*fixed_cost + total_km*km_cost, 2),
               "CO2_estimado_kg": round(total_km*co2_factor, 2), "Tiempo_VRPy_s": round(total_time, 3)}
    details = []
    for i, r in enumerate(all_routes, 1):
        d = r["depot_index"]
        details.append({"Escenario": name, "Ruta": i, "ID_Deposito": str(depots.iloc[d].ID_Deposito),
                        "ID_Camion": r["ID_Camion"],
                        "Secuencia": " → ".join([str(depots.iloc[d].ID_Deposito)] + [str(pickups.iloc[p].ID_CentroSalud) for p in r["puntos"]] + [str(depots.iloc[d].ID_Deposito)]),
                        "Distancia_km": round(r["km"], 3), "Carga_kg": round(r["kg"], 2),
                        "Capacidad_kg": r["capacidad_kg"], "Utilizacion_pct": round(100*r["kg"]/r["capacidad_kg"], 2)})
    return summary, details, all_routes


def fetch_road_shapes(api_key, routes, depots, pickups):
    '''Directions API: obtain actual road polylines in route order, including return to depot.
    Split long routes to stay within the legacy Directions waypoint limit.
    '''
    client = googlemaps.Client(key=api_key, timeout=30)
    palette = ["#e63946", "#2563eb", "#16a34a", "#a855f7", "#ea580c", "#0891b2", "#be185d", "#64748b"]
    shapes, failures = [], []
    for idx, route in enumerate(routes):
        d = route["depot_index"]
        depot = (float(depots.iloc[d].latitud), float(depots.iloc[d].longitud))
        stops = [depot] + [(float(pickups.iloc[p].latitud), float(pickups.iloc[p].longitud)) for p in route["puntos"]] + [depot]
        full_line = []
        try:
            # Maximum 23 intermediate stops per Directions API request.
            for start in range(0, len(stops) - 1, 24):
                chunk = stops[start:min(start + 25, len(stops))]
                if len(chunk) < 2:
                    continue
                responses = client.directions(chunk[0], chunk[-1], waypoints=chunk[1:-1], mode="driving", alternatives=False)
                if not responses:
                    raise RuntimeError("Google Directions no devolvió ruta")
                for leg in responses[0]["legs"]:
                    for step in leg["steps"]:
                        decoded = googlemaps.convert.decode_polyline(step["polyline"]["points"])
                        segment = [[float(pt["lat"]), float(pt["lng"])] for pt in decoded]
                        if full_line and segment and full_line[-1] == segment[0]:
                            segment = segment[1:]
                        full_line.extend(segment)
            shapes.append({"route": idx + 1, "truck": str(route["ID_Camion"]), "depot": str(depots.iloc[d].ID_Deposito),
                           "color": palette[idx % len(palette)], "path": full_line,
                           "stops": [{"lat": float(pickups.iloc[p].latitud), "lng": float(pickups.iloc[p].longitud),
                                      "name": str(pickups.iloc[p].ID_CentroSalud), "order": j + 1} for j, p in enumerate(route["puntos"])]})
        except Exception as exc:
            failures.append(f"Ruta {idx + 1}: {exc}")
    return shapes, failures


def google_routes_html(browser_key, shapes, depots):
    '''Google Maps JavaScript API visualization, geometry computed by Directions API.'''
    depots_data = [{"lat": float(r.latitud), "lng": float(r.longitud), "name": str(r.ID_Deposito)} for r in depots.itertuples()]
    payload = json.dumps({"routes": shapes, "depots": depots_data}, ensure_ascii=False).replace("<", "\\u003c")
    key = html.escape(browser_key, quote=True)
    return '''<!DOCTYPE html><html><head><meta charset="utf-8"><style>
    html,body,#map{height:100%;width:100%;margin:0}#legend{position:absolute;top:10px;left:10px;z-index:2;background:white;
    padding:10px;max-height:35%;overflow:auto;font:12px Arial;box-shadow:0 1px 5px #888;border-radius:6px}
    </style></head><body><div id="map"></div><div id="legend"></div>
    <script>const data = ''' + payload + ''';
    function initMap(){
      const origin=data.depots[0]||{lat:-12.05,lng:-77.05};
      const map=new google.maps.Map(document.getElementById('map'),{center:origin,zoom:11,mapTypeId:'roadmap'});
      const bounds=new google.maps.LatLngBounds();const info=new google.maps.InfoWindow();
      data.depots.forEach(d=>{const pos={lat:d.lat,lng:d.lng};bounds.extend(pos);
        const m=new google.maps.Marker({position:pos,map,title:'Depósito '+d.name,label:'D'});
        m.addListener('click',()=>{info.setContent('Depósito: '+d.name);info.open(map,m)});});
      const legend=[];
      data.routes.forEach(r=>{
        const line=new google.maps.Polyline({path:r.path.map(p=>({lat:p[0],lng:p[1]})),strokeColor:r.color,
          strokeOpacity:0.9,strokeWeight:5,map});
        r.path.forEach(p=>bounds.extend({lat:p[0],lng:p[1]}));
        r.stops.forEach(s=>{const pos={lat:s.lat,lng:s.lng};bounds.extend(pos);
          const m=new google.maps.Marker({position:pos,map,title:s.name,label:String(s.order)});
          m.addListener('click',()=>{info.setContent('Ruta '+r.route+' / '+r.truck+'<br>Parada '+s.order+': '+s.name);info.open(map,m)});
        });
        legend.push('<div><span style="color:'+r.color+'">&#9632;</span> Ruta '+r.route+' — '+r.truck+'</div>');
      });
      document.getElementById('legend').innerHTML=legend.join('');
      if(!bounds.isEmpty())map.fitBounds(bounds);
    }
    window.gm_authFailure=function(){document.getElementById('legend').textContent='Google Maps: error de autenticación. Comprueba la clave de navegador y las restricciones HTTP referrer.'};
    </script><script async defer src="https://maps.googleapis.com/maps/api/js?key=''' + key + '''&callback=initMap"></script></body></html>'''


def excel_bytes(summaries, details):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        summaries.to_excel(writer, sheet_name="Indicadores", index=False)
        details.to_excel(writer, sheet_name="Rutas", index=False)
    return buffer.getvalue()


with st.sidebar:
    st.header("Configuración")
    try:
        secret = st.secrets.get("GOOGLE_MAPS_API_KEY", "")
    except FileNotFoundError:
        secret = ""
    api_key = st.text_input("Google Maps API Key", value=secret, type="password")
    use_google = st.checkbox("Usar distancias viales Google Maps", value=True)
    st.caption("Para visualizar el mapa Google, la clave de navegador debe permitir Maps JavaScript API y estar restringida por HTTP referrer. Puede ser distinta de la clave del servidor.")
    browser_key = st.text_input("Google Maps JavaScript API Key (navegador)", type="password", help="Configura esta clave con restricciones HTTP referrer para el dominio de la aplicación.")
    st.caption("Requiere habilitar Distance Matrix API (Legacy), si está disponible en tu proyecto. Alternativa: distancias geodésicas, no aptas para estimar kilómetros reales de carretera.")
    fixed_cost = st.number_input("Costo fijo por viaje (moneda local)", min_value=0.0, value=80.0)
    km_cost = st.number_input("Costo por km", min_value=0.0, value=2.5)
    co2_factor = st.number_input("Factor de emisión (kg CO₂/km)", min_value=0.0, value=0.8)
    limit = st.slider("Límite VRPy por depósito y escenario (segundos)", 5, 180, 30)

st.subheader("1. Archivos de entrada")
st.write("**Depósitos:** `ID_Deposito, latitud, longitud, NumCamiones`. **Centros:** `ID_CentroSalud, latitud, longitud, PesoDesperdicio` (en kg).")
c1, c2 = st.columns(2)
with c1:
    dep_file = st.file_uploader("Depósitos y flota (CSV o Excel)", type=["csv", "xlsx"], key="dep")
with c2:
    pick_file = st.file_uploader("Centros de salud (CSV o Excel)", type=["csv", "xlsx"], key="pick")

if dep_file is None or pick_file is None:
    st.info("Carga ambos archivos para habilitar el análisis.")
    st.stop()

try:
    depots = check_data(read_upload(dep_file), DEPOT_COLS, "Depósitos")
    pickups = check_data(read_upload(pick_file), PICKUP_COLS, "Centros")
    depots["NumCamiones"] = pd.to_numeric(depots.NumCamiones, errors="raise").astype(int)
    pickups["PesoDesperdicio"] = pd.to_numeric(pickups.PesoDesperdicio, errors="raise")
    if depots.empty or pickups.empty or (depots.NumCamiones <= 0).any() or (pickups.PesoDesperdicio <= 0).any():
        raise ValueError("Se requieren depósitos con camiones y centros con demanda positiva")
    if depots.ID_Deposito.duplicated().any() or pickups.ID_CentroSalud.duplicated().any():
        raise ValueError("Los identificadores deben ser únicos dentro de cada archivo")
except Exception as exc:
    st.error(str(exc))
    st.stop()

with st.expander("Ver datos cargados", expanded=False):
    st.write("Depósitos", depots)
    st.write("Centros", pickups)

st.subheader("2. Configurar la capacidad de cada camión")
st.write("Se genera una fila editable por camión según `NumCamiones` de cada depósito. Selecciona 0.5, 1 o 1.5 toneladas individualmente.")
initial_fleet = []
for _, depot in depots.iterrows():
    for i in range(int(depot.NumCamiones)):
        initial_fleet.append({"ID_Deposito": str(depot.ID_Deposito),
                              "ID_Camion": f"{depot.ID_Deposito}-T{i+1:02d}",
                              "Capacidad_tn": 1.0})
initial_fleet = pd.DataFrame(initial_fleet)
# Conservar edición de capacidades al modificar otros controles, reiniciar si cambia la estructura.
fleet_signature = tuple(zip(initial_fleet.ID_Deposito, initial_fleet.ID_Camion))
if st.session_state.get("fleet_signature") != fleet_signature:
    st.session_state["fleet_signature"] = fleet_signature
    st.session_state["fleet_editor_version"] = st.session_state.get("fleet_editor_version", 0) + 1
fleet = st.data_editor(
    initial_fleet, hide_index=True, use_container_width=True, num_rows="fixed",
    key=f"fleet_editor_{st.session_state['fleet_editor_version']}",
    disabled=["ID_Deposito", "ID_Camion"],
    column_config={"Capacidad_tn": st.column_config.SelectboxColumn(
        "Capacidad (tn)", options=[0.5, 1.0, 1.5], required=True)},
)
if fleet.Capacidad_tn.isna().any() or not fleet.Capacidad_tn.isin(CAPACITIES).all():
    st.error("Cada camión debe tener una capacidad de 0.5, 1 o 1.5 toneladas")
    st.stop()
st.caption("Cada camión puede realizar como máximo un viaje por escenario. Las rutas de un depósito se optimizan conjuntamente con VRPy mixed_fleet.")
st.subheader("3. Ejecutar análisis")
if st.button("Calcular rutas e indicadores", type="primary"):
    if use_google and not api_key:
        st.error("Ingresa tu API Key o configura GOOGLE_MAPS_API_KEY en Secrets.")
        st.stop()
    points = pd.concat([depots[["latitud", "longitud"]], pickups[["latitud", "longitud"]]], ignore_index=True)
    try:
        with st.spinner("Obteniendo matriz de distancias..."):
            matrix = build_matrix(points, api_key, use_google)
        assignment = assign_to_depots(pickups, depots, matrix, len(depots))
        summary, detail, raw_routes = compute_scenario("Flota individual configurada", fleet, pickups, depots, matrix, assignment,
                                           fixed_cost, km_cost, co2_factor, limit)
        st.session_state["results"] = pd.DataFrame([summary])
        st.session_state["routes"] = pd.DataFrame(detail)
        st.session_state["points"] = points
        st.session_state["distance_source"] = "Google Maps (carretera)" if use_google else "Haversine (línea recta)"
        st.session_state["fleet"] = fleet.copy()
        st.session_state["raw_routes"] = raw_routes
        st.session_state["map_shapes"] = None
        st.session_state["map_failures"] = []
    except Exception as exc:
        st.error(f"No se pudo completar la optimización: {exc}")
        st.code(traceback.format_exc(), language="python")
        st.info("Si el grafo pasó la validación, el traceback permitirá localizar la operación interna de VRPy que falla.")
        st.stop()

if "results" in st.session_state:
    res, routes = st.session_state["results"], st.session_state["routes"]
    st.subheader("4. Indicadores de la flota configurada")
    st.caption("Distancias: " + st.session_state["distance_source"] + ". El tiempo es el tiempo de resolución de VRPy sumado por depósito; no incluye las consultas a Google.")
    st.dataframe(res, use_container_width=True, hide_index=True)
    a, b, c = st.columns(3)
    a.metric("Menos viajes", f"{res.loc[res.Viajes.idxmin(), 'Escenario']} · {res.Viajes.min()}")
    b.metric("Menor distancia", f"{res.loc[res.Distancia_km.idxmin(), 'Escenario']} · {res.Distancia_km.min():.1f} km")
    c.metric("Menor costo", f"{res.loc[res.Costo_estimado.idxmin(), 'Escenario']} · {res.Costo_estimado.min():.2f}")
    st.subheader("5. Visualización")
    metric = st.selectbox("Indicador", ["Viajes", "Distancia_km", "Utilizacion_pct", "Costo_estimado", "Km_por_kg", "Tiempo_VRPy_s", "CO2_estimado_kg"])
    st.bar_chart(res.set_index("Escenario")[[metric]])
    st.write("**Compromiso entre viajes, distancia y costo**")
    if len(res) > 1:
        st.scatter_chart(res, x="Distancia_km", y="Costo_estimado", size="Viajes", color="Escenario")
    else:
        st.caption("Para comparar compromisos entre escenarios, ejecuta otras configuraciones de flota y conserva sus resultados exportados.")
    st.subheader("6. Rutas obtenidas")
    st.dataframe(routes, use_container_width=True, hide_index=True)
    st.write("**Flota configurada**")
    st.dataframe(st.session_state["fleet"], use_container_width=True, hide_index=True)
    st.subheader("7. Rutas sobre Google Maps")
    st.caption("Google Directions API obtiene la geometría vial siguiendo el orden de paradas calculado por VRPy; el mapa se muestra con Maps JavaScript API. Las consultas pueden generar cargos.")
    if not use_google:
        st.warning("Activa 'Usar distancias viales Google Maps' y vuelve a calcular para visualizar recorridos reales por carretera.")
    elif not browser_key:
        st.info("Introduce la clave de Google Maps JavaScript API en la barra lateral para mostrar el mapa.")
    else:
        if st.button("Dibujar rutas reales en Google Maps", type="secondary"):
            try:
                with st.spinner("Consultando geometrías de carretera con Google Directions API..."):
                    shapes, failures = fetch_road_shapes(api_key, st.session_state["raw_routes"], depots, pickups)
                st.session_state["map_shapes"] = shapes
                st.session_state["map_failures"] = failures
            except Exception as exc:
                st.error(f"No se pudieron consultar las rutas de Google: {exc}")
        if st.session_state.get("map_shapes") is not None:
            for failure in st.session_state.get("map_failures", []):
                st.warning(failure)
            if st.session_state["map_shapes"]:
                components.html(google_routes_html(browser_key, st.session_state["map_shapes"], depots), height=600)
            else:
                st.warning("No se obtuvo ninguna geometría vial. Revisa la habilitación de Directions API y los puntos.")
    with st.expander("Mapa simple de puntos (alternativo)"):
        points = st.session_state["points"]
        st.map(points.rename(columns={"latitud": "lat", "longitud": "lon"})[["lat", "lon"]])
    st.subheader("8. Descargas")
    left, right = st.columns(2)
    left.download_button("Descargar Excel", excel_bytes(res, routes), file_name="resultados_vrpy.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    right.download_button("Descargar indicadores CSV", res.to_csv(index=False).encode("utf-8-sig"), file_name="indicadores_vrpy.csv", mime="text/csv")
    st.download_button("Descargar rutas CSV", routes.to_csv(index=False).encode("utf-8-sig"), file_name="rutas_vrpy.csv", mime="text/csv")
    st.warning("Interpretación: la asignación al depósito más cercano es heurística; VRPy optimiza una flota heterogénea por depósito. No hay ventanas de tiempo, restricciones de compatibilidad sanitaria ni partición de cargas. Cada camión puede realizar como máximo una ruta.")
