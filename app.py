"""
Asistente de precios para anfitriones de Airbnb en Barcelona
TFM - Fase 6 CRISP-DM (Despliegue)

Ejecutar en local:   streamlit run app.py
Estructura esperada:
    app.py
    requirements.txt
    artefactos/
        modelo_xgb.json
        variables_finales.json
        metricas.json
        referencia.parquet
        puntos_interes.json
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

def eur(x):
    return f"{x:,.0f}".replace(",", ".")


ART = Path(__file__).parent / "artefactos"

st.set_page_config(
    page_title="Precio recomendado | Barcelona",
    page_icon="🏠",
    layout="wide"
)


#carga de artefactos
@st.cache_resource
def cargar_modelo():
    from xgboost import XGBRegressor
    m = XGBRegressor()
    m.load_model(str(ART / "modelo_xgb.json"))
    return m


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


#distancias
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


#catálogos de la UI
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

SHAP_LABELS = {
    "minimum_nights": "Noches mínimas",
    "maximum_nights": "Noches máximas",
    "accommodates": "Huéspedes",
    "bedrooms": "Habitaciones",
    "bathrooms_num": "Número de baños",

    "antiguedad_como_host": "Antigüedad como anfitrión (meses)",
    "antiguedad_anuncio": "Antigüedad del anuncio (meses)",
    "calculated_host_listings_count": "Anuncios gestionados por el anfitrión",

    "number_of_reviews_ltm": "Reseñas últimos 12 meses",
    "review_scores_rating": "Valoración global",
    "review_scores_location": "Valoración de ubicación",

    "dist_center_km": "Distancia al centro (km)",
    "dist_beach_km": "Distancia a la playa (km)",
    "dist_sagrada_familia_km": "Distancia a la Sagrada Familia (km)",
    "dist_park_guell_km": "Distancia al Park Güell (km)",
    "dist_camp_nou_km": "Distancia al Camp Nou (km)",

    "host_has_profile_pic_True": "Foto de perfil",
    "room_type_Private_room": "Habitación privada",
    "room_type_Shared_room": "Habitación compartida",
    "has_availability_True": "Disponibilidad activa",
    "shared_bathroom_True": "Baño compartido",
    "es_gran_tenedor_True": "Gran tenedor",
    "tiene_biografia_True": "Biografía del anfitrión",

    "licencia_estado_Licencia_registrada": "Licencia registrada",
    "licencia_estado_Sin_dato": "Licencia sin datos",

    "sin_reviews_True": "Anuncio sin reseñas",
    "tiene_descripcion_True": "Descripción del anuncio",

    "has_ac_True": "Aire acondicionado",
    "has_elevator_True": "Ascensor",
    "has_workstation_True": "Espacio de trabajo",
    "has_dishwasher_True": "Lavavajillas",
    "has_parking_True": "Parking",
    "has_pool_True": "Piscina",

    "host_location_agrupado_Desconocido": "Ubicación del anfitrión desconocida",

    "property_type_agrupado_Entire_rental_unit": "Alojamiento entero",
    "property_type_agrupado_Otros": "Otro tipo de propiedad",
    "property_type_agrupado_Private_room_in_rental_unit": "Habitación privada en alojamiento",
    "property_type_agrupado_Room_in_hotel": "Habitación de hotel",

    "neighbourhood_agrupado_Sant_Pere_Santa_Caterina_i_la_Ribera": "Barrio: Sant Pere, Santa Caterina i la Ribera",
    "neighbourhood_agrupado_Sants": "Barrio: Sants",
    "neighbourhood_agrupado_el_Barri_Gòtic": "Barrio: Barrio Gótico",
    "neighbourhood_agrupado_el_Fort_Pienc": "Barrio: Fort Pienc",
    "neighbourhood_agrupado_el_Poble_Sec": "Barrio: Poble-sec",
    "neighbourhood_agrupado_el_Raval": "Barrio: El Raval",
    "neighbourhood_agrupado_l_Antiga_Esquerra_de_l_Eixample": "Barrio: Antiga Esquerra de l'Eixample",
    "neighbourhood_agrupado_la_Barceloneta": "Barrio: La Barceloneta",
    "neighbourhood_agrupado_la_Dreta_de_l_Eixample": "Barrio: Dreta de l'Eixample",
    "neighbourhood_agrupado_la_Sagrada_Família": "Barrio: Sagrada Família",
    "neighbourhood_agrupado_la_Vila_de_Gràcia": "Barrio: Vila de Gràcia",
}
# Centro por defecto: Plaça de Catalunya
LAT_DEF, LON_DEF = 41.3870, 2.1700


#formulario
st.title("¿A qué precio debería publicar mi anuncio?")

with st.sidebar:
    st.header("Datos del alojamiento")

    barrio = st.selectbox("Barrio", list(BARRIOS))
    tipo = st.selectbox("Tipo de propiedad", list(TIPOS))
    room = st.selectbox("Tipo de anuncio", list(HABITACION))

    accommodates = st.slider("Huéspedes", 1, 16, 4)
    bedrooms = st.slider("Habitaciones", 0, 8, 2)
    bathrooms = st.slider("Baños", 0.0, 5.0, 1.0, step=0.5)
    shared_bath = st.checkbox("Baño compartido")

    with st.expander("Condiciones de reserva"):
        minimum_nights = st.number_input("Noches mínimas", 1, 365, 2)
        maximum_nights = st.number_input("Noches máximas", 1, 1125, 365)
        availability_eoy = st.slider("Días disponibles hasta fin de año", 0, 365, 180)
        licencia = st.selectbox("Situación de licencia", list(LICENCIA))

    with st.expander("Reputación"):
        sin_reviews = st.checkbox("Anuncio nuevo, sin reseñas")
        reviews_ltm = st.number_input(
            "Reseñas en los últimos 12 meses",
            0, 500,
            0 if sin_reviews else 20
        )
        rating = st.slider(
            "Valoración global",
            1.0, 5.0, 4.7,
            step=0.1,
            disabled=sin_reviews
        )
        rating_loc = st.slider(
            "Valoración de ubicación",
            1.0, 5.0, 4.8,
            step=0.1,
            disabled=sin_reviews
        )

    with st.expander("Anfitrión y ubicación"):
        n_anuncios = st.number_input("Anuncios que gestionas", 1, 200, 1)
        antiguedad_host = st.number_input(
            "Antigüedad como anfitrión (meses)",
            0, 240, 36
        )
        antiguedad_anuncio = st.number_input(
            "Antigüedad del anuncio (meses)",
            0, 240, 12
        )
        tiene_bio = st.checkbox("Tengo biografía en el perfil", value=True)
        tiene_desc = st.checkbox("El anuncio tiene descripción", value=True)
        foto_perfil = st.checkbox("Tengo foto de perfil", value=True)
        host_local = st.checkbox("Mi ubicación es pública", value=True)

        st.caption("La ubicación se ajusta automáticamente al barrio seleccionado.")

        lat_def, lon_def = CENTROIDES[barrio]

        lat = st.number_input(
            "Latitud",
            value=lat_def,
            format="%.5f",
            key=f"lat_{barrio}"
        )

        lon = st.number_input(
            "Longitud",
            value=lon_def,
            format="%.5f",
            key=f"lon_{barrio}"
        )

    with st.expander("Equipamiento", expanded=True):
        amenities_sel = {
            nombre: st.checkbox(nombre)
            for nombre in AMENITIES
        }


#construcción de la fila
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


# Comparables de mercado
col_barrio = BARRIOS[barrio]

comparables = ref.copy()

if col_barrio:
    comparables = comparables[comparables[col_barrio] == 1]

comparables = comparables[
    comparables["accommodates"].between(accommodates - 1, accommodates + 1)
]

mediana_mercado = (
    float(comparables["price_eur"].median())
    if len(comparables) >= 10
    else None
)
#salida
tab_precio, tab_expl, tab_mercado, tab_modelo = st.tabs(
    ["Precio recomendado", "Qué influye en tu precio", "Comparación de mercado", "Ficha del modelo"]
)

with tab_precio:
    c1, c2, c3 = st.columns(3)
    c1.metric("Precio recomendado", f"{precio:,.0f} €/noche", delta=(f"{precio - mediana_mercado:+,.0f} € vs. mediana de comparables"
                     if mediana_mercado is not None
                     else None))
    c2.metric("Rango razonable", f"{banda_baja:,.0f} – {banda_alta:,.0f} €")
    c3.metric("Ingreso potencial anual", f"{precio * availability_eoy * 0.65:,.0f} €",
              help="Precio estimado × días disponibles × 65% de ocupación media.")

with st.expander("¿Cómo se calcula el rango?"):
    st.markdown("El rango se construye con el error del modelo en test "
        "(RMSE de 0.30 en escala logarítmica), es decir aproximadamente "
        "×0.74 y ×1.35 sobre el precio estimado.")
  

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
        sv.feature_names = [SHAP_LABELS.get(c, c) for c in X_input.columns]
        shap.plots.waterfall(sv[0], max_display=14, show=False)
        fig.canvas.draw()
        ax = plt.gca()
        labels = ax.get_yticklabels()
        ax.set_yticklabels([
    "variables adicionales" if "other features" in x.get_text() else x.get_text()
    for x in labels
])
        st.pyplot(fig, clear_figure=True)
       

        st.caption(
            "Las contribuciones son aditivas sobre el log-precio: una barra de +0,20 "
            "multiplica el precio por e^0,20 ≈ 1,22 (+22%)."
        )
    except Exception as e:
        st.warning(f"No se ha podido generar la explicación SHAP: {e}")

with tab_mercado:
    st.subheader(f"Anuncios similares ({len(comparables)})")

    if len(comparables) >= 10:
        q = comparables["price_eur"].quantile([0.25, 0.5, 0.75])
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("P25 del mercado", f"{eur(q[0.25])} €")
        c2.metric("Mediana", f"{eur(q[0.5])} €")
        c3.metric("P75", f"{eur(q[0.75])} €")
        c4.metric(
            "Tu posición",
            f"{(comparables['price_eur'] < precio).mean() * 100:.0f}%"
        )

        import altair as alt

        conteo, bordes = np.histogram(comparables["price_eur"], bins=18)

        hist = pd.DataFrame({
            "centro": (bordes[:-1] + bordes[1:]) / 2,
            "desde": bordes[:-1],
            "hasta": bordes[1:],
            "anuncios": conteo,
        })

        barras = alt.Chart(hist).mark_bar().encode(
            x=alt.X("centro:Q", title="Precio por noche (€)"),
            y=alt.Y("anuncios:Q", title="Nº de anuncios"),
            tooltip=[
                alt.Tooltip("desde:Q", title="Desde (€)", format=".0f"),
                alt.Tooltip("hasta:Q", title="Hasta (€)", format=".0f"),
                alt.Tooltip("anuncios:Q", title="Anuncios"),
            ],
        )

        linea = alt.Chart(
            pd.DataFrame({"x": [precio]})
        ).mark_rule(strokeWidth=3).encode(x="x:Q")

        st.altair_chart(
            (barras + linea).properties(height=280),
            use_container_width=True
        )

        st.caption(
            f"La línea marca tu precio recomendado: {eur(precio)} €/noche."
        )

    else:
        st.info(
            "Pocos anuncios similares con estos filtros. "
            "Amplía el número de huéspedes o elige «Resto de barrios»."
        )
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
            f"del conjunto de test (Inside Airbnb, marzo 2026)."
        )

with tab_modelo:
    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown("""
        **¿Cómo funciona este modelo?**

        El precio se estima con un algoritmo de *gradient boosting* (XGBoost) entrenado
        con anuncios de Airbnb publicados en Barcelona. El modelo aprende las relaciones
        entre características como ubicación, capacidad, equipamiento y reputación, y
        utiliza estos patrones para estimar el precio de un nuevo alojamiento.

        **¿Qué significa el R²?**

        Indica qué proporción de la variación de precios entre anuncios consigue explicar
        el modelo. En el conjunto de test, el modelo alcanza un R² de 0,878.

        **¿Qué error cabe esperar?**

        El error mediano absoluto es de aproximadamente 23 €, lo que significa que en
        la mitad de los casos la estimación se desvía menos de esta cantidad respecto
        al precio observado.
        """)

    with c2:
        st.metric("R² en test", f"{METRICAS['r2_test']:.3f}")
        st.metric("Error mediano", f"{eur(METRICAS['mediana_error_euros'])} €")
        st.metric("Error medio (MAE)", f"{eur(METRICAS['mae_euros'])} €")
        st.metric("Anuncios de entrenamiento", f"{eur(METRICAS['n_train'])}")
        st.metric("Variables del modelo", len(VARIABLES))

    st.caption(
        "Datos: Inside Airbnb, Barcelona, marzo 2026 · "
        "RMSE en escala logarítmica: 0,300"
    )
