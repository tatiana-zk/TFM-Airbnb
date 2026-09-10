"""
Asistente de precios para anfitriones de Airbnb en Barcelona
TFM - Fase 6 CRISP-DM (Despliegue)

Ejecutar en local:   streamlit run app.py
Estructura esperada:
    app.py
    requirements.txt
    artefactos/
        modelo_xgb.pkl
        variables_finales.json
        metricas.json
        referencia.parquet
        puntos_interes.json
"""

import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk


ART = Path(__file__).parent / "artefactos"

st.set_page_config(page_title="Precio recomendado | Barcelona", layout="wide")


# ---------------------------------------------------------------- carga de artefactos
@st.cache_resource
def cargar_modelo():
    with open(ART / "modelo_xgb.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def cargar_json(nombre):
    with open(ART / nombre, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def cargar_referencia():
    return pd.read_parquet(ART / "referencia.parquet")


modelo = cargar_modelo()
VARIABLES = cargar_json("variables_finales.json")
METRICAS = cargar_json("metricas.json")
POI = cargar_json("puntos_interes.json")
ref = cargar_referencia()

RMSE_LOG = METRICAS["rmse_log_test"]


# ---------------------------------------------------------------- distancias
def distancia_km(lat1, lon1, lat2, lon2):
    """Distancia entre dos puntos. Usa la proyección UTM 31N (la misma del
    notebook) si pyproj está disponible; si no, haversine (diferencia < 0.5%)."""
    try:
        from pyproj import Transformer

        tr = Transformer.from_crs("EPSG:4326", "EPSG:25831", always_xy=True)
        x1, y1 = tr.transform(lon1, lat1)
        x2, y2 = tr.transform(lon2, lat2)
        return math.hypot(x1 - x2, y1 - y2) / 1000
    except Exception:
        R = 6371.0
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dp = p2 - p1
        dl = math.radians(lon2 - lon1)
        a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------- catálogos de la UI
BARRIOS = {
    "Sant Pere, Santa Caterina i la Ribera": "neighbourhood_agrupado_Sant_Pere_Santa_Caterina_i_la_Ribera",
    "Sants": "neighbourhood_agrupado_Sants",
    "El Barri Gòtic": "neighbourhood_agrupado_el_Barri_Gòtic",
    "El Fort Pienc": "neighbourhood_agrupado_el_Fort_Pienc",
    "El Poble Sec": "neighbourhood_agrupado_el_Poble_Sec",
    "El Raval": "neighbourhood_agrupado_el_Raval",
    "L'Antiga Esquerra de l'Eixample": "neighbourhood_agrupado_l_Antiga_Esquerra_de_l_Eixample",
    "La Barceloneta": "neighbourhood_agrupado_la_Barceloneta",
    "La Dreta de l'Eixample": "neighbourhood_agrupado_la_Dreta_de_l_Eixample",
    "La Sagrada Família": "neighbourhood_agrupado_la_Sagrada_Família",
    "La Vila de Gràcia": "neighbourhood_agrupado_la_Vila_de_Gràcia",
    "Resto de barrios": None,  # categoría de referencia
}

CENTROIDES = {
    "Sant Pere, Santa Caterina i la Ribera": (41.3865, 2.1810),
    "Sants": (41.3750, 2.1330),
    "El Barri Gòtic": (41.3830, 2.1770),
    "El Fort Pienc": (41.3950, 2.1810),
    "El Poble Sec": (41.3730, 2.1620),
    "El Raval": (41.3800, 2.1690),
    "L'Antiga Esquerra de l'Eixample": (41.3880, 2.1520),
    "La Barceloneta": (41.3800, 2.1900),
    "La Dreta de l'Eixample": (41.3930, 2.1650),
    "La Sagrada Família": (41.4050, 2.1750),
    "La Vila de Gràcia": (41.4020, 2.1560),
    "Resto de barrios": (41.3900, 2.1600),
}
TIPOS = {
    "Vivienda entera": "property_type_agrupado_Entire_rental_unit",
    "Habitación privada en vivienda": "property_type_agrupado_Private_room_in_rental_unit",
    "Habitación de hotel": "property_type_agrupado_Room_in_hotel",
    "Otros": "property_type_agrupado_Otros",
    "Categoría de referencia": None,
}

HABITACION = {
    "Alojamiento entero": None,
    "Habitación privada": "room_type_Private_room",
    "Habitación compartida": "room_type_Shared_room",
}

LICENCIA = {
    "Licencia registrada (HUT)": "licencia_estado_Licencia_registrada",
    "Sin dato": "licencia_estado_Sin_dato",
    "Exenta / otra situación": None,
}

AMENITIES = {
    "Aire acondicionado": "has_ac_True",
    "Ascensor": "has_elevator_True",
    "Zona de trabajo": "has_workstation_True",
    "Lavavajillas": "has_dishwasher_True",
    "Parking": "has_parking_True",
    "Piscina": "has_pool_True",
}

# Centro por defecto: Plaça de Catalunya
LAT_DEF, LON_DEF = 41.3870, 2.1700


# ---------------------------------------------------------------- formulario
st.title("¿A qué precio debería publicar mi anuncio?")
st.caption(
    f"Modelo XGBoost entrenado con {METRICAS['n_train']:,} anuncios de Inside Airbnb (Barcelona, marzo 2025). "
    f"R² en test: {METRICAS['r2_test']:.3f} · error mediano: {METRICAS['mediana_error_euros']:.0f} €/noche."
)

with st.sidebar:
    st.header("Datos del alojamiento")

    barrio = st.selectbox("Barrio", list(BARRIOS))
    tipo = st.selectbox("Tipo de propiedad", list(TIPOS))
    room = st.selectbox("Tipo de anuncio", list(HABITACION))

    accommodates = st.slider("Huéspedes", 1, 16, 4)
    bedrooms = st.slider("Habitaciones", 0, 8, 2)
    bathrooms = st.slider("Baños", 0.0, 5.0, 1.0, step=0.5)
    shared_bath = st.checkbox("Baño compartido")

    st.divider()
    st.subheader("Condiciones de reserva")
    minimum_nights = st.number_input("Noches mínimas", 1, 365, 2)
    maximum_nights = st.number_input("Noches máximas", 1, 1125, 365)
    availability_eoy = st.slider("Días disponibles hasta fin de año", 0, 365, 180)
    licencia = st.selectbox("Situación de licencia", list(LICENCIA))

    st.divider()
    st.subheader("Reputación")
    sin_reviews = st.checkbox("Anuncio nuevo, sin reseñas")
    reviews_ltm = st.number_input("Reseñas en los últimos 12 meses", 0, 500, 0 if sin_reviews else 20)
    rating = st.slider("Valoración global", 1.0, 5.0, 4.7, step=0.1, disabled=sin_reviews)
    rating_loc = st.slider("Valoración de ubicación", 1.0, 5.0, 4.8, step=0.1, disabled=sin_reviews)

    st.divider()
    st.subheader("Anfitrión")
    n_anuncios = st.number_input("Anuncios que gestionas", 1, 200, 1)
    antiguedad_host = st.number_input("Antigüedad como anfitrión (meses)", 0, 240, 36)
    antiguedad_anuncio = st.number_input("Antigüedad del anuncio (meses)", 0, 240, 12)
    tiene_bio = st.checkbox("Tengo biografía en el perfil", value=True)
    tiene_desc = st.checkbox("El anuncio tiene descripción", value=True)
    foto_perfil = st.checkbox("Tengo foto de perfil", value=True)
    host_local = st.checkbox("Mi ubicación es pública", value=True)

    st.divider()
    st.subheader("Ubicación exacta")
    lat_def, lon_def = CENTROIDES[barrio]
    lat = st.number_input("Latitud", value=lat_def, format="%.5f", key=f"lat_{barrio}")

    lon = st.number_input("Longitud", value=lon_def, format="%.5f", key=f"lon_{barrio}")
  

    st.divider()
    st.subheader("Equipamiento")
    amenities_sel = {nombre: st.checkbox(nombre) for nombre in AMENITIES}


# ---------------------------------------------------------------- construcción de la fila
def construir_fila(amenities_estado):
    fila = {v: 0.0 for v in VARIABLES}

    fila["accommodates"] = accommodates
    fila["bedrooms"] = bedrooms
    fila["bathrooms_num"] = bathrooms
    fila["minimum_nights"] = minimum_nights
    fila["maximum_nights"] = maximum_nights
    fila["availability_eoy"] = availability_eoy
    fila["number_of_reviews_ltm"] = 0 if sin_reviews else reviews_ltm
    fila["review_scores_rating"] = 0.0 if sin_reviews else rating
    fila["review_scores_location"] = 0.0 if sin_reviews else rating_loc
    fila["calculated_host_listings_count"] = n_anuncios
    fila["antiguedad_como_host"] = antiguedad_host
    fila["antiguedad_anuncio"] = antiguedad_anuncio

    for p in POI:
        fila[p["feature"]] = distancia_km(lat, lon, p["ylat"], p["xlong"])

    for col in (BARRIOS[barrio], TIPOS[tipo], HABITACION[room], LICENCIA[licencia]):
        if col:
            fila[col] = 1.0

    fila["shared_bathroom_True"] = float(shared_bath)
    fila["sin_reviews_True"] = float(sin_reviews)
    fila["tiene_biografia_True"] = float(tiene_bio)
    fila["tiene_descripcion_True"] = float(tiene_desc)
    fila["host_has_profile_pic_True"] = float(foto_perfil)
    fila["host_location_agrupado_Desconocido"] = float(not host_local)
    fila["es_gran_tenedor_True"] = float(n_anuncios > 5)
    fila["has_availability_True"] = float(availability_eoy > 0)

    for nombre, col in AMENITIES.items():
        fila[col] = float(amenities_estado[nombre])

    return pd.DataFrame([fila], columns=VARIABLES).astype(float)


X_input = construir_fila(amenities_sel)
pred_log = float(modelo.predict(X_input)[0])
precio = float(np.exp(pred_log))
banda_baja = float(np.exp(pred_log - RMSE_LOG))
banda_alta = float(np.exp(pred_log + RMSE_LOG))


# ---------------------------------------------------------------- salida
tab_precio, tab_expl, tab_mercado, tab_modelo = st.tabs(
    ["Precio recomendado", "Qué influye en tu precio", "Tu posición en el mercado", "Ficha del modelo"]
)

with tab_precio:
    c1, c2, c3 = st.columns(3)
    c1.metric("Precio estimado", f"{precio:,.0f} €/noche")
    c2.metric("Rango razonable", f"{banda_baja:,.0f} – {banda_alta:,.0f} €")
    c3.metric("Ingreso potencial anual", f"{precio * availability_eoy * 0.65:,.0f} €",
              help="Precio estimado × días disponibles × 65% de ocupación media.")

    st.markdown(
        "El rango se construye con el error del modelo en test "
        f"(RMSE de {RMSE_LOG:.2f} en escala logarítmica, es decir aproximadamente "
        f"×{np.exp(-RMSE_LOG):.2f} y ×{np.exp(RMSE_LOG):.2f} sobre el precio estimado)."
    )

  

    st.subheader("Simulador: ¿cuánto suma cada mejora?")
    st.caption("Cambio en el precio estimado si añades cada equipamiento manteniendo todo lo demás igual.")
    filas = []
    for nombre in AMENITIES:
        if amenities_sel[nombre]:
            continue
        estado = dict(amenities_sel)
        estado[nombre] = True
        nuevo = float(np.exp(modelo.predict(construir_fila(estado))[0]))
        filas.append({"Mejora": nombre, "Precio con la mejora (€)": round(nuevo, 1),
                      "Diferencia (€)": round(nuevo - precio, 1)})
    if filas:
        st.dataframe(pd.DataFrame(filas).sort_values("Diferencia (€)", ascending=False),
                     hide_index=True, use_container_width=True)
    else:
        st.info("Ya has marcado todo el equipamiento disponible.")

with tab_expl:
    st.subheader("Contribución de cada variable a tu precio")
    try:
        import shap
        import matplotlib.pyplot as plt

        @st.cache_resource
        def cargar_explainer():
            return shap.TreeExplainer(modelo)

        explainer = cargar_explainer()
        sv = explainer(X_input)
        fig = plt.figure()
        shap.plots.waterfall(sv[0], max_display=14, show=False)
        st.pyplot(fig, clear_figure=True)
        st.caption(
            "Las contribuciones son aditivas sobre el log-precio: una barra de +0,20 "
            "multiplica el precio por e^0,20 ≈ 1,22 (+22%)."
        )
    except Exception as e:
        st.warning(f"No se ha podido generar la explicación SHAP: {e}")

with tab_mercado:
    col_barrio = BARRIOS[barrio]
    comparables = ref.copy()
    if col_barrio:
        comparables = comparables[comparables[col_barrio] == 1]
    comparables = comparables[comparables["accommodates"].between(accommodates - 1, accommodates + 1)]

    st.subheader(f"Anuncios comparables ({len(comparables)} en la muestra de referencia)")
    if len(comparables) >= 10:
        q = comparables["price_eur"].quantile([0.25, 0.5, 0.75])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("P25 del mercado", f"{q[0.25]:,.0f} €")
        c2.metric("Mediana", f"{q[0.5]:,.0f} €")
        c3.metric("P75", f"{q[0.75]:,.0f} €")
        c4.metric("Percentil de tu precio",
                  f"{(comparables['price_eur'] < precio).mean() * 100:.0f}%")
        hist = (pd.cut(comparables["price_eur"], bins=20).value_counts().sort_index()
                )
        hist.index = hist.index.astype(str)
        st.bar_chart(hist.rename("Anuncios"))
    else:
        st.info("Pocos comparables con estos filtros. Amplía el número de huéspedes o elige 'Resto de barrios'.")
        CENTROIDES = {
        "Sant Pere, Santa Caterina i la Ribera": (41.3865, 2.1810),
        "Sants": (41.3750, 2.1330),
        "El Barri Gòtic": (41.3830, 2.1770),
        "El Fort Pienc": (41.3950, 2.1810),
        "El Poble Sec": (41.3730, 2.1620),
        "El Raval": (41.3800, 2.1690),
        "L'Antiga Esquerra de l'Eixample": (41.3880, 2.1520),
        "La Barceloneta": (41.3800, 2.1900),
        "La Dreta de l'Eixample": (41.3930, 2.1650),
        "La Sagrada Família": (41.4050, 2.1750),
        "La Vila de Gràcia": (41.4020, 2.1560),
    }

    filas = []
    for nombre, col in BARRIOS.items():
        if col is None or nombre not in CENTROIDES:
            continue
        sub = ref[ref[col] == 1]
        if len(sub) < 15:
            continue
        lat_b, lon_b = CENTROIDES[nombre]
        filas.append({"barrio": nombre, "lat": lat_b, "lon": lon_b,
                      "mediana": int(round(sub["price_eur"].median())), "n": len(sub)})

    if filas:
        mapa = pd.DataFrame(filas)
        lo, hi = mapa["mediana"].min(), mapa["mediana"].max()
        t = (mapa["mediana"] - lo) / max(hi - lo, 1)
        mapa["c"] = [[int(225 * v + 25), 70, int(225 * (1 - v) + 25), 190] for v in t]
        mapa["r"] = 200 + 500 * t
        mapa["etiqueta"] = mapa["mediana"].astype(str) + " €"
        mapa["tip"] = (mapa["barrio"] + ": " + mapa["mediana"].astype(str)
                       + " €/noche (" + mapa["n"].astype(str) + " anuncios)")

        st.subheader("Precio mediano por barrio")
        st.pydeck_chart(pdk.Deck(
            map_style="light",
            initial_view_state=pdk.ViewState(latitude=41.392, longitude=2.168, zoom=12.1),
            layers=[
                pdk.Layer("ScatterplotLayer", mapa, get_position=["lon", "lat"],
                          get_fill_color="c", get_radius="r", pickable=True),
                pdk.Layer("TextLayer", mapa, get_position=["lon", "lat"],
                          get_text="etiqueta", get_size=13, get_color=[255, 255, 255]),
                pdk.Layer("ScatterplotLayer", pd.DataFrame([{"lat": lat, "lon": lon}]),
                          get_position=["lon", "lat"], get_fill_color=[20, 20, 20],
                          get_radius=140),
            ],
            tooltip={"text": "{tip}"},
        ))
        st.caption(
            f"Círculo grande y rojo: barrio más caro. Pequeño y azul: más barato. "
            f"El número es el precio mediano por noche. Punto negro: tu alojamiento "
            f"({precio:,.0f} €/noche estimado). Muestra de referencia: {len(ref):,} anuncios "
            f"del conjunto de test (Inside Airbnb, marzo 2025)."
        )

with tab_modelo:
    st.subheader("Ficha técnica")
    st.json({
        "Algoritmo": "XGBoost Regressor (variable objetivo: log del precio)",
        "R² test": round(METRICAS["r2_test"], 3),
        "RMSE test (log)": round(METRICAS["rmse_log_test"], 3),
        "MAE (€)": round(METRICAS["mae_euros"], 1),
        "Error mediano (€)": round(METRICAS["mediana_error_euros"], 1),
        "Nº variables": len(VARIABLES),
        "Datos": "Inside Airbnb, Barcelona, marzo 2025",
    })
    st.markdown(
        "**Limitaciones.** El modelo estima el precio publicado por anuncios similares, no el precio "
        "óptimo de mercado ni la ocupación esperada. No incorpora estacionalidad ni eventos, y los "
        "resultados se degradan en alojamientos atípicos (más de 8 huéspedes o precios superiores a 500 €)."
    )
