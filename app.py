"""
Asistente de precios para anfitriones de Airbnb en Barcelona
TFM - Fase 6 CRISP-DM (Despliegue)

Ejecutar en local:   streamlit run app.py
Estructura esperada:
    app.py
    requirements.txt
    .streamlit/config.toml
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
import altair as alt
import base64

st.set_page_config(
    page_title="Precio recomendado | Barcelona",
    page_icon="🏠",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-baseweb="tab"] {font-size: 18px; font-weight: 500;}
    [data-testid="stCaptionContainer"] {font-size: 15px;}
    [data-testid="stHeaderActionElements"] {display: none;}

    .hero {
        position: relative;
        width: 100%;
        height: 360px;
        overflow: hidden;
        border-radius: 14px;
        margin-bottom: 28px;
    }

    .hero img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }

    .hero-overlay {
        position: absolute;
        inset: 0;
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        padding: 42px;
        background: linear-gradient(
            transparent 25%,
            rgba(0,0,0,0.65)
        );
    }

    .hero-overlay h1 {
        color: white;
        font-size: 2.6rem;
        margin: 0 0 12px 0;
    }

    .hero-overlay p {
        color: white;
        font-size: 1.05rem;
        max-width: 800px;
        margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def eur(x):
    """Formatea un número con separador de miles español."""
    return f"{x:,.0f}".replace(",", "@").replace(".", ",").replace("@", ".")


ART = Path(__file__).parent / "artefactos"
st.markdown(
    f"""
    <div class="hero">
        <img src="data:image/png;base64,{base64.b64encode(
            (ART / "logo.png").read_bytes()
        ).decode()}">
        <div class="hero-overlay">
            <h1>¿A qué precio debería publicar mi anuncio?</h1>
            <p>
                Estima un precio recomendado por noche para tu alojamiento
                de Airbnb en Barcelona a partir de sus características,
                ubicación y condiciones del anuncio.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# carga de artefactos
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


# distancias
def distancia_km(lat1, lon1, lat2, lon2):
    """Distancia entre dos puntos. Usa la proyección UTM 31N (la misma del
    notebook) si pyproj está disponible; si no, haversine (diferencia < 0,5%)."""
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


# catálogos de la UI
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


# cabecera


with st.expander("¿Cómo funciona?"):
    e1, e2, e3 = st.columns(3)
    e1.markdown("**1 · Describe tu alojamiento**")
    e1.markdown("Barrio, capacidad, equipamiento y condiciones de reserva.")
    e2.markdown("**2 · Obtén un precio y un rango**")
    e2.markdown("El modelo estima el precio y un rango razonable.")
    e3.markdown("**3 · Entiende la recomendación**")
    e3.markdown("Consulta qué factores influyen y cómo te comparas con anuncios similares.")
    st.caption(
        f"Datos: Inside Airbnb (Barcelona, marzo 2026) · "
        f"R² en test: {METRICAS['r2_test']:.3f} · "
        f"Error mediano: {METRICAS['mediana_error_euros']:.0f} €/noche. "
        "La herramienta es orientativa y no sustituye el criterio del anfitrión."
    )


# formulario
with st.sidebar:
    st.header("Datos del alojamiento")

    barrio = st.selectbox("Barrio", list(BARRIOS))
    accommodates = st.slider("Huéspedes", 1, 16, 4)
    bedrooms = st.slider("Habitaciones", 0, 8, 2)
    bathrooms = st.slider("Baños", 0.0, 5.0, 1.0, step=0.5)
    shared_bath = st.checkbox("Baño compartido")

    tipo = st.selectbox("Tipo de propiedad", list(TIPOS))

    if tipo == "Vivienda entera":
        room = "Alojamiento entero"
        st.selectbox("Tipo de anuncio", ["Alojamiento entero"], disabled=True)
    elif tipo == "Habitación privada en vivienda":
        room = "Habitación privada"
        st.selectbox("Tipo de anuncio", ["Habitación privada"], disabled=True)
    elif tipo == "Habitación de hotel":
        room = None
        st.selectbox("Tipo de anuncio", ["No aplica"], disabled=True)
    else:
        room = st.selectbox("Tipo de anuncio", list(HABITACION))

    with st.expander("Condiciones de reserva"):
        minimum_nights = st.number_input("Noches mínimas", 1, 365, 2)
        maximum_nights = st.number_input("Noches máximas", 1, 1125, 365)
        availability_eoy = st.slider("Días disponibles hasta fin de año", 0, 365, 180)
        licencia = st.selectbox("Situación de licencia", list(LICENCIA))

    with st.expander("Reputación"):
        sin_reviews = st.checkbox("Anuncio nuevo, sin reseñas")
        reviews_ltm = st.number_input(
            "Reseñas en los últimos 12 meses", 0, 500, 0 if sin_reviews else 20
        )
        rating = st.slider("Valoración global", 1.0, 5.0, 4.7, step=0.1, disabled=sin_reviews)
        rating_loc = st.slider(
            "Valoración de ubicación", 1.0, 5.0, 4.8, step=0.1, disabled=sin_reviews
        )

    with st.expander("Anfitrión y ubicación"):
        n_anuncios = st.number_input("Anuncios que gestionas", 1, 200, 1)
        antiguedad_host = st.number_input("Antigüedad como anfitrión (meses)", 0, 240, 36)
        antiguedad_anuncio = st.number_input("Antigüedad del anuncio (meses)", 0, 240, 12)
        tiene_bio = st.checkbox("Tengo biografía en el perfil", value=True)
        tiene_desc = st.checkbox("El anuncio tiene descripción", value=True)
        foto_perfil = st.checkbox("Tengo foto de perfil", value=True)
        host_local = st.checkbox("Mi ubicación es pública", value=True)

        st.caption("La ubicación se ajusta automáticamente al barrio seleccionado.")
        lat_def, lon_def = CENTROIDES[barrio]
        lat = st.number_input("Latitud", value=lat_def, format="%.5f", key=f"lat_{barrio}")
        lon = st.number_input("Longitud", value=lon_def, format="%.5f", key=f"lon_{barrio}")

    with st.expander("Equipamiento", expanded=True):
        amenities_sel = {nombre: st.checkbox(nombre) for nombre in AMENITIES}


# construcción de la fila
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

    for col in (BARRIOS[barrio], TIPOS[tipo], HABITACION.get(room), LICENCIA[licencia]):
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


#comparables de mercado
col_barrio = BARRIOS[barrio]
comparables = ref.copy()
if col_barrio:
    comparables = comparables[comparables[col_barrio] == 1]
comparables = comparables[
    comparables["accommodates"].between(accommodates - 1, accommodates + 1)
]
mediana_mercado = (
    float(comparables["price_eur"].median()) if len(comparables) >= 10 else None
)


# salida
tab_precio, tab_expl, tab_mercado, tab_modelo = st.tabs(
    ["Precio recomendado", "Factores", "Mercado", "Modelo"]
)


with tab_precio:
    c1, c2, c3 = st.columns(3)

    with c1:
        with st.container(border=True):
            st.markdown("**Precio recomendado**")
            st.markdown(f"## {eur(precio)} €/noche")
            if mediana_mercado is not None:
                diferencia = precio - mediana_mercado
                st.markdown(
                    f"<span style='background:#E8F5E9; color:#2E7D32; "
                    f"padding:4px 10px; border-radius:14px; font-size:14px;'>"
                    f"↑ {eur(abs(diferencia))} € vs. mediana de comparables</span>",
                    unsafe_allow_html=True)

    with c2:
        with st.container(border=True):
            st.markdown("**Intervalo estimado**")
            st.markdown(f"## {eur(banda_baja)} – {eur(banda_alta)} €")
            st.caption("Basado en el error del modelo")

    with c3:
        with st.container(border=True):
            st.markdown("**Comparación con el mercado**")
            if mediana_mercado is not None:
                diferencia_pct = (precio / mediana_mercado - 1) * 100
                texto = (
                    f"+{diferencia_pct:.0f}%"
                    if diferencia_pct >= 0
                    else f"{diferencia_pct:.0f}%"
                )
                st.markdown(f"## {texto}")
                st.caption(
                    f"Recomendado: {eur(precio)} € · Mediana: {eur(mediana_mercado)} €"
                )
            else:
                st.markdown("## —")
                st.caption("Sin comparables suficientes")

    # Alojamiento, con mini mapa 
    with st.container(border=True):
         c1, c2 = st.columns([1.7, 1])
         with c1:
            st.markdown("**Tu alojamiento**")
            st.markdown(f"📍 **{barrio}**")
            st.caption("Barcelona")

            p1, p2, p3, p4 = st.columns(4)
            p1.markdown(f"🏠  {tipo}")
            p2.markdown(f"🛏  {bedrooms} habitaciones")
            p3.markdown(f"🛀  {bathrooms:g} baño{'s' if bathrooms != 1 else ''}")
            p4.markdown(f"👥  {accommodates} huéspedes")

         with c2:
            st.pydeck_chart(pdk.Deck(map_style="light",initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=12.5),
                                     layers=[pdk.Layer(
                            "ScatterplotLayer",
                            pd.DataFrame([{"lat": lat, "lon": lon}]),
                            get_position=["lon", "lat"],
                            get_fill_color=[193, 72, 60, 220],
                            get_radius=150,)],),height=145,)


    # Simulador en formato cascada
    # Impacto de las mejoras
    st.subheader("¿Qué impacto tienen las mejoras?")
    st.caption("Efecto acumulado de añadir cada equipamiento, uno tras otro, partiendo de tu configuración actual.")

    pendientes = [n for n in AMENITIES if not amenities_sel[n]]

    if pendientes:
        # Orden por impacto individual
        impactos = []
        for n in pendientes:
            e = {**amenities_sel, n: True}
            p = float(np.exp(modelo.predict(construir_fila(e))[0]))
            impactos.append((n, p - precio))

        orden_mejoras = [n for n, _ in sorted(impactos, key=lambda x: x[1], reverse=True)]

        # Cascada acumulada: recalcular XGBoost en cada paso
        pasos, estado, anterior = [], amenities_sel.copy(), precio

        for n in orden_mejoras:
            estado[n] = True
            nuevo = float(np.exp(modelo.predict(construir_fila(estado))[0]))
            pasos.append((n, anterior, nuevo, nuevo - anterior))
            anterior = nuevo

        precio_final = anterior
        pasos = [("Precio actual", precio, precio, 0)] + pasos
        pasos += [("Precio final", precio_final, precio_final, precio_final - precio)]

        # Datos del gráfico
        wf = pd.DataFrame(pasos, columns=["etapa", "inicio", "fin", "delta"])
        wf["tipo"] = np.where(
            wf.etapa.isin(["Precio actual", "Precio final"]),
            "base", np.where(wf.delta >= 0, "sube", "baja")
        )
        wf["orden"] = range(len(wf))

        y_min = int(np.floor((min(wf.inicio.min(), wf.fin.min()) - 15) / 10) * 10)
        y_max = int(np.ceil((max(wf.inicio.max(), wf.fin.max()) + 10) / 10) * 10)

        wf.loc[wf.tipo == "base", "inicio"] = y_min
        wf["etiqueta"] = np.where(
            wf.tipo == "base",
            wf.fin.round().astype(int).astype(str) + " €",
            wf.delta.map(lambda x: f"{x:+.0f} €")
        )
        wf["label_y"] = wf[["inicio", "fin"]].max(axis=1)

        labels = wf.etapa.tolist()
        expr = " : ".join(
            [f"datum.value == {i} ? '{v}'" for i, v in enumerate(labels)]
        ) + " : ''"

        x = alt.X(
            "orden:Q", title=None,
            scale=alt.Scale(domain=[-0.5, len(wf) - 0.5], nice=False),
            axis=alt.Axis(
                values=list(range(len(wf))),
                labelExpr=expr,
                labelAngle=-20,
                tickSize=0
            )
        )

        # Barras
        barras = alt.Chart(wf).mark_bar(size=42).encode(
            x=x,
            y=alt.Y(
                "inicio:Q",
                title="Precio por noche (€)",
                scale=alt.Scale(domain=[y_min, y_max]),
                axis=alt.Axis(
                    values=list(range(y_min, y_max + 1, 20)),
                    labelOverlap=False
                )
            ),
            y2="fin:Q",
            color=alt.Color(
                "tipo:N",
                scale=alt.Scale(
                    domain=["base", "sube", "baja"],
                    range=["#6B6560", "#C1483C", "#9B5C54"]
                ),
                legend=None
            )
        )

        # Conectores
        c = pd.DataFrame({
            "x1": wf.orden.iloc[:-1] + .18,
            "x2": wf.orden.iloc[1:].values - .18,
            "y": wf.fin.iloc[:-1].values
        })

        lineas = alt.Chart(c).mark_rule(
            color="#B8B2AD",
            strokeDash=[4, 4]
        ).encode(
            x="x1:Q", x2="x2:Q", y="y:Q"
        )

        # Etiquetas
        etiquetas = alt.Chart(wf).mark_text(
            dy=-8, fontSize=11, fontWeight="bold"
        ).encode(
            x="orden:Q",
            y="label_y:Q",
            text="etiqueta:N"
        )

        st.altair_chart(
            (barras + lineas + etiquetas).properties(height=300),
            use_container_width=True
        )

        st.caption(
            f"Del precio actual ({eur(precio)} €) al precio final "
            f"({eur(precio_final)} €) con todas las mejoras. "
            "Las diferencias reflejan asociaciones observadas en los datos, "
            "no efectos causales."
        )


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
        sv.feature_names = [SHAP_LABELS.get(c, c) for c in X_input.columns]

        shap.plots.waterfall(sv[0], max_display=12, show=False)

        fig = plt.gcf()
        fig.set_size_inches(10, 5)
        ax = fig.gca()
        ax.set_yticklabels(
            [
                "variables adicionales" if "other features" in t.get_text() else t.get_text()
                for t in ax.get_yticklabels()
            ]
        )
        ax.tick_params(labelsize=9)
        fig.tight_layout()

        izq, centro, der = st.columns([1, 10, 1])
        with centro:
            st.pyplot(fig, clear_figure=True)

        st.caption(
            "Las contribuciones son aditivas sobre el logaritmo del precio: una barra "
            "de +0,20 multiplica el precio por e^0,20 ≈ 1,22 (+22%)."
        )
    except Exception as e:
        st.warning(f"No se ha podido generar la explicación SHAP: {e}")


with tab_mercado:
    st.subheader(f"Anuncios similares ({len(comparables)})")

    if len(comparables) >= 10:
        q = comparables["price_eur"].quantile([0.25, 0.5, 0.75])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("P25 del mercado", f"{eur(q[0.25])} €")
        m2.metric("Mediana", f"{eur(q[0.5])} €")
        m3.metric("P75", f"{eur(q[0.75])} €")
        m4.metric(
            "Comparables por debajo",
            f"{(comparables['price_eur'] < precio).mean() * 100:.0f}%",
        )

        conteo, bordes = np.histogram(comparables["price_eur"], bins=15)
        hist = pd.DataFrame(
            {"desde": bordes[:-1], "hasta": bordes[1:], "anuncios": conteo}
        )

        barras_h = alt.Chart(hist).mark_bar(color="#9BB7C4").encode(
            x=alt.X("desde:Q", bin="binned", title="Precio por noche (€)"),
            x2="hasta:Q",
            y=alt.Y("anuncios:Q", title="Nº de anuncios"),
            tooltip=[
                alt.Tooltip("desde:Q", title="Desde (€)", format=".0f"),
                alt.Tooltip("hasta:Q", title="Hasta (€)", format=".0f"),
                alt.Tooltip("anuncios:Q", title="Anuncios"),
            ],
        )

        linea = (
            alt.Chart(pd.DataFrame({"x": [precio]}))
            .mark_rule(color="#C1483C", strokeWidth=3)
            .encode(x="x:Q")
        )

        st.altair_chart(
            (barras_h + linea).properties(height=260), use_container_width=True
        )
        st.caption(f"La línea marca tu precio recomendado: {eur(precio)} €/noche.")
    else:
        st.info(
            "Pocos anuncios similares con estos filtros. "
            "Amplía el número de huéspedes o elige «Resto de barrios»."
        )

    filas_mapa = []
    for nombre, col in BARRIOS.items():
        if col is None or nombre not in CENTROIDES:
            continue
        sub = ref[ref[col] == 1]
        if len(sub) < 15:
            continue
        lat_b, lon_b = CENTROIDES[nombre]
        filas_mapa.append(
            {
                "barrio": nombre,
                "lat": lat_b,
                "lon": lon_b,
                "mediana": int(round(sub["price_eur"].median())),
                "n": len(sub),
            }
        )

    if filas_mapa:
        mapa = pd.DataFrame(filas_mapa)
        lo, hi = mapa["mediana"].min(), mapa["mediana"].max()
        t = (mapa["mediana"] - lo) / max(hi - lo, 1)
        mapa["c"] = [[int(225 * v + 25), 70, int(225 * (1 - v) + 25), 190] for v in t]
        mapa["r"] = 200 + 500 * t
        mapa["etiqueta"] = mapa["mediana"].astype(str) + " €"
        mapa["tip"] = (
            mapa["barrio"]
            + ": "
            + mapa["mediana"].astype(str)
            + " €/noche ("
            + mapa["n"].astype(str)
            + " anuncios)"
        )

        st.subheader("Precio mediano por barrio")
        st.pydeck_chart(
            pdk.Deck(
                map_style="light",
                initial_view_state=pdk.ViewState(
                    latitude=41.392, longitude=2.168, zoom=12.1
                ),
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        mapa,
                        get_position=["lon", "lat"],
                        get_fill_color="c",
                        get_radius="r",
                        pickable=True,
                    ),
                    pdk.Layer(
                        "TextLayer",
                        mapa,
                        get_position=["lon", "lat"],
                        get_text="etiqueta",
                        get_size=13,
                        get_color=[255, 255, 255],
                    ),
                    pdk.Layer(
                        "ScatterplotLayer",
                        pd.DataFrame([{"lat": lat, "lon": lon}]),
                        get_position=["lon", "lat"],
                        get_fill_color=[20, 20, 20],
                        get_radius=140,
                    ),
                ],
                tooltip={"text": "{tip}"},
            )
        )
        st.caption(
            f"Círculo grande y rojo: barrio más caro. Pequeño y azul: más barato. "
            f"El número es el precio mediano por noche. Punto negro: tu alojamiento "
            f"({eur(precio)} €/noche estimado). Muestra de referencia: {eur(len(ref))} "
            f"anuncios del conjunto de test (Inside Airbnb, marzo 2026)."
        )


with tab_modelo:
    d1, d2 = st.columns([3, 2])

    with d1:
        st.markdown(
            """
        **¿Cómo funciona este modelo?**

        El precio se estima con un algoritmo de *gradient boosting* (XGBoost) entrenado
        con anuncios de Airbnb publicados en Barcelona. El modelo aprende las relaciones
        entre características como ubicación, capacidad, equipamiento y reputación, y
        utiliza estos patrones para estimar el precio de un nuevo alojamiento.

        **¿Qué significa el R²?**

        Indica qué proporción de la variación del precio (en escala logarítmica, que es
        la variable sobre la que se entrena el modelo) consigue explicar. En el conjunto
        de test alcanza un R² de 0,878, un valor alto para datos de mercado real.

        **¿Qué error cabe esperar?**

        El error mediano absoluto es de aproximadamente 23 €: en la mitad de los casos
        la estimación se desvía menos de esa cantidad respecto al precio observado. La
        diferencia con el error medio (50 €) indica que las desviaciones grandes se
        concentran en un número reducido de alojamientos caros.
        """
        )

    with d2:
        st.metric("R² en test", f"{METRICAS['r2_test']:.3f}")
        st.metric("Error mediano", f"{eur(METRICAS['mediana_error_euros'])} €")
        st.metric("Error medio (MAE)", f"{eur(METRICAS['mae_euros'])} €")
        st.metric("Anuncios de entrenamiento", f"{eur(METRICAS['n_train'])}")
        st.metric("Variables del modelo", len(VARIABLES))

    st.caption(
        "Datos: Inside Airbnb, Barcelona, marzo 2026 · "
        "RMSE en escala logarítmica: 0,300"
    )
