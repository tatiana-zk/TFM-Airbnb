"""
Asistente de precios para anfitriones de Airbnb en Barcelona
TFM - Despliegue

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
        valores_por_defecto.json
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
    '<div class="tfm-label">TRABAJO FIN DE MÁSTER · 2026</div>',
    unsafe_allow_html=True,
)

_logo = ART / "logo.png"
_img_html = (
    f'<img src="data:image/png;base64,{base64.b64encode(_logo.read_bytes()).decode()}">'
    if _logo.exists() else ""
)
st.markdown(
    f"""
    <div class="hero">
        {_img_html}
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
DEF = cargar_json("valores_por_defecto.json")  # mediana de X_train = anuncio típico

MEDIANA_ANTIG_DIAS = float(DEF.get("antiguedad_anuncio", 1262.5))

# Límites coherentes con los datos de entrenamiento (Inside Airbnb, 21-mar-2026)
LIM = {
    "hab_max": 6,
    "banos_max": 4.0,
}


def slider_seguro(label, lo, hi, valor, step=1, **kw):
    """Slider que no falla cuando lo == hi y recorta el valor al rango."""
    valor = min(max(valor, lo), hi)
    if lo >= hi:
        st.markdown(f"**{label}:** {lo:g}")
        if kw.get("help"):
            st.caption(kw["help"])
        return lo
    return st.slider(label, lo, hi, valor, step=step, **kw)


def base(col, lo, hi, fallback, tipo=float):
    """Valor del caso base (mediana de entrenamiento), recortado al rango del widget."""
    v = DEF.get(col, fallback)
    return tipo(min(max(v, lo), hi))


def base_bool(col, fallback=False):
    """Para dummies la mediana equivale a la moda: >= 0,5 significa 'Sí'."""
    return bool(DEF.get(col, float(fallback)) >= 0.5)

RMSE_LOG = METRICAS["rmse_log_test"]


# distancias
@st.cache_resource
def cargar_transformer():
    from pyproj import Transformer

    return Transformer.from_crs("EPSG:4326", "EPSG:25831", always_xy=True)


def distancia_km(lat1, lon1, lat2, lon2):
    """Distancia entre dos puntos. Usa la proyección UTM 31N (la misma del
    notebook) si pyproj está disponible; si no, haversine (diferencia < 0,5%)."""
    try:
        tr = cargar_transformer()
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


HABITACION = {
    "Alojamiento entero": None,
    "Habitación privada": "room_type_Private_room",
    "Habitación compartida": "room_type_Shared_room",
}

LICENCIA = {
    "Licencia registrada": "licencia_estado_Licencia_registrada",
    "No indicada en el anuncio": "licencia_estado_Sin_dato",
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
    "antiguedad_anuncio": "Días desde la primera reseña",
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
        f"R² del modelo XGBoost en datos de prueba: {METRICAS['r2_test']:.3f} · "
        f"Error mediano: {METRICAS['mediana_error_euros']:.0f} €/noche. "
        "La herramienta es orientativa y no sustituye el criterio del anfitrión."
    )


# formulario

with st.sidebar:
    st.header("Tu alojamiento")
    st.caption(
        "Rellena las características principales de tu alojamiento. "
    "Puedes ajustar más detalles en 'Opciones avanzadas'.")

    barrio = st.selectbox("Barrio", list(BARRIOS.keys()))

    tipo_habitacion = st.selectbox("Tipo de habitación", list(HABITACION.keys()))

    huespedes_max = {
    "Alojamiento entero": 8,
    "Habitación privada": 3,
    "Habitación compartida": 8,}[tipo_habitacion]
    accommodates = slider_seguro(
    "Huéspedes", 1, huespedes_max,
    min(int(base("accommodates", 1, huespedes_max, 4, int)), huespedes_max))

    hab_max = 1 if tipo_habitacion != "Alojamiento entero" else min(LIM["hab_max"], accommodates)
    bedrooms = slider_seguro("Habitaciones", 1, hab_max,
    min(int(base("bedrooms", 1, hab_max, 2, int)), hab_max))

    banos_max = float(min(LIM["banos_max"], bedrooms + 1))
    bathrooms = slider_seguro(
    "Baños", 0.5, banos_max,
    min(base("bathrooms_num", 0.5, banos_max, 1.0), banos_max),
    step=0.5)

    shared_bath = st.checkbox(
    "Baño compartido",
    value=False if tipo_habitacion == "Alojamiento entero" else base_bool("shared_bathroom_True", False),
    disabled=tipo_habitacion == "Alojamiento entero",
    help="Indica si el baño es compartido con otros huéspedes.",)
    

    minimum_nights = slider_seguro(
        "Noches mínimas", 1, 30,
        int(base("minimum_nights", 1, 30, 2, int)))

    anuncio_nuevo = st.checkbox(
    "Anuncio nuevo",
    value=False,
    help="Indica si el alojamiento es nuevo y todavía no tiene reseñas."
)
    rating = st.slider(
    "Valoración global", 1.0, 5.0,
    base("review_scores_rating", 1.0, 5.0, 4.61),
    step=0.01,
    disabled=anuncio_nuevo,
    help="Valoración media del alojamiento por parte de los huéspedes.",
)

    with st.expander("⚙️ Opciones avanzadas (opcional)"):
        st.markdown("**Licencia**")
    
        licencia = st.selectbox("Situación de licencia",list(LICENCIA),)

        st.markdown("**Servicios**")

        amenities_sel = {
            label: st.checkbox(
                label,
                value=base_bool(col, False)
            )
            for label, col in AMENITIES.items()
        }




# construcción de la fila
def construir_fila(amenities_estado):
    fila = {v: float(DEF.get(v, 0.0)) for v in VARIABLES}

    for v in VARIABLES:
        if v.startswith(("room_type_",
        "property_type_agrupado_",
        "neighbourhood_agrupado_",
        "licencia_estado_",)):
            fila[v] = 0.0
    PROPIEDAD = {
        "Alojamiento entero": "property_type_agrupado_Entire_rental_unit",
        "Habitación privada": "property_type_agrupado_Private_room_in_rental_unit",
        "Habitación compartida": None,
    }
    if PROPIEDAD[tipo_habitacion]:
        fila[PROPIEDAD[tipo_habitacion]] = 1.0


    if HABITACION[tipo_habitacion]:
        fila[HABITACION[tipo_habitacion]] = 1.0

    fila["accommodates"] = accommodates
    fila["bedrooms"] = bedrooms
    fila["bathrooms_num"] = bathrooms
    fila["minimum_nights"] = minimum_nights
    

    fila["maximum_nights"] = float(DEF.get("maximum_nights", 365))
    fila["availability_eoy"] = float(DEF.get("availability_eoy", 207))
    fila["number_of_reviews_ltm"] = float(DEF.get("number_of_reviews_ltm", 3))
    fila["review_scores_rating"] = rating
    fila["review_scores_location"] = float(DEF.get("review_scores_location", 4.78))

    fila["calculated_host_listings_count"] = float(DEF.get("calculated_host_listings_count", 12))

    fila["antiguedad_como_host"] = float(
        DEF.get("antiguedad_como_host", 98.61)
    )
    fila["antiguedad_anuncio"] = MEDIANA_ANTIG_DIAS

    # Barrio y tipo de habitación
    for col in (BARRIOS[barrio], HABITACION[tipo_habitacion], LICENCIA[licencia]):
        if col:
            fila[col] = 1.0

    # Baño compartido y anuncio nuevo
    fila["shared_bathroom_True"] = float(shared_bath)
    fila["sin_reviews_True"] = float(anuncio_nuevo)
    if anuncio_nuevo:
        fila["number_of_reviews_ltm"] = 0.0
        fila["review_scores_rating"] = 0.0
        fila["review_scores_location"] = 0.0
        fila["sin_reviews_True"] = 1.0
        fila["antiguedad_anuncio"] = MEDIANA_ANTIG_DIAS
    else:
        fila["number_of_reviews_ltm"] = float(DEF.get("number_of_reviews_ltm", 3))
        fila["review_scores_rating"] = rating
        fila["review_scores_location"] = float(DEF.get("review_scores_location", 4.78))
        fila["sin_reviews_True"] = 0.0
        fila["antiguedad_anuncio"] = MEDIANA_ANTIG_DIAS

    # Variables ocultas
    fila["tiene_biografia_True"] = float(DEF.get("tiene_biografia_True", 1))
    fila["tiene_descripcion_True"] = float(DEF.get("tiene_descripcion_True", 1))
    fila["host_has_profile_pic_True"] = float(DEF.get("host_has_profile_pic_True", 1))
    fila["host_location_agrupado_Desconocido"] = float(DEF.get("host_location_agrupado_Desconocido", 0))
    fila["es_gran_tenedor_True"] = float(DEF.get("es_gran_tenedor_True", 1))
    
    fila["has_availability_True"] = float(DEF.get("has_availability_True", 1))

    # Distancias derivadas del barrio
    lat, lon = CENTROIDES[barrio]

    for p in POI:
        fila[p["feature"]] = distancia_km(
            lat, lon, p["ylat"], p["xlong"]
        )
        

    

    for nombre, col in AMENITIES.items():
        fila[col] = float(amenities_estado[nombre])

    return pd.DataFrame([fila], columns=VARIABLES).astype(float)


X_input = construir_fila(amenities_sel)
P01, P99 = ref["price_eur"].quantile([0.01, 0.99])
pred_log = float(modelo.predict(X_input)[0])
precio = float(np.exp(pred_log))
banda_baja = float(np.exp(pred_log - RMSE_LOG))
banda_alta = float(np.exp(pred_log + RMSE_LOG))


#comparables de mercado

col_barrio = BARRIOS[barrio]
comparables_base = ref.copy()

if col_barrio:
    comparables_base = comparables_base[
        comparables_base[col_barrio] == 1
    ]
else:
    cols_b = [c for c in BARRIOS.values() if c and c in comparables_base.columns]
    comparables_base = comparables_base[
        comparables_base[cols_b].sum(axis=1) == 0
    ]

acc_ref = min(accommodates, 9.5)
comparables_base = comparables_base[
    comparables_base["accommodates"].between(acc_ref - 1, acc_ref + 1)
]

# Primero: mismo tipo de habitación
col_room = HABITACION[tipo_habitacion]

if col_room:
    comparables = comparables_base[
        comparables_base[col_room] == 1
    ]
else:
    comparables = comparables_base[
        (comparables_base["room_type_Private_room"] == 0)
        & (comparables_base["room_type_Shared_room"] == 0)
    ]

# Si hay menos de 10, se amplía a todos los tipos de habitación
if len(comparables) < 10:
    comparables = comparables_base.copy()

mediana_mercado = (
    float(comparables["price_eur"].median())
    if len(comparables) >= 10 else None
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
            if not (P01 <= precio <= P99):
                st.caption("⚠️ Precio fuera del rango habitual del mercado: tómalo con cautela.")
            if mediana_mercado is not None:
                diferencia = precio - mediana_mercado
                arriba = diferencia >= 0
                bg, fg, flecha = ("#E8F5E9", "#2E7D32", "↑") if arriba else ("#FDECEA", "#C62828", "↓")
                st.markdown(
                    f"<span style='background:{bg}; color:{fg}; "
                    f"padding:4px 10px; border-radius:14px; font-size:14px;'>"
                    f"{flecha} {eur(abs(diferencia))} € vs. mediana de comparables</span>",
                    unsafe_allow_html=True)

    with c2:
        with st.container(border=True):
            st.markdown("**Intervalo estimado**")
            st.markdown(f"## {eur(banda_baja)} – {eur(banda_alta)} €")
            st.caption("Rango orientativo basado en el error típico del modelo.")

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
            p1.markdown(f"🏠  {tipo_habitacion}")
            p2.markdown(f"🛏  {bedrooms} habitaci{'ón' if bedrooms == 1 else 'ones'}")
            p3.markdown(f"🛀  {bathrooms:g} baño{'s' if bathrooms != 1 else ''}")
            p4.markdown(f"👥  {accommodates} huésped{'' if accommodates == 1 else 'es'}")

         with c2:
            lat, lon = CENTROIDES[barrio]
            st.pydeck_chart(pdk.Deck(map_style="light",initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=12.5),
                                     layers=[pdk.Layer(
                            "ScatterplotLayer",
                            pd.DataFrame([{"lat": lat, "lon": lon}]),
                            get_position=["lon", "lat"],
                            get_fill_color=[193, 72, 60, 220],
                            get_radius=150,)],),height=145,)



with tab_expl:
    #SHAP

    try:
                    import shap
                    @st.cache_resource
                    def cargar_explainer():
                        return shap.TreeExplainer(modelo)
                    explainer = cargar_explainer()
                    sv = explainer(X_input)
                    vals = sv.values[0]
                    names = [SHAP_LABELS.get(c, c) for c in X_input.columns]
                    data = pd.DataFrame({
                        "variable": names,
                        "valor": X_input.iloc[0].values,
                        "shap": vals
                    })
                    data["efecto_pct"] = (np.exp(data["shap"]) - 1) * 100
                    data["abs"] = data["shap"].abs()
                    data = data.nlargest(10, "abs").sort_values("efecto_pct")

                    def valor_texto(row):
                        v = row["valor"]
                        variable = row["variable"]

                        if variable == "Lavavajillas":
                            return "Lavavajillas: Sí" if v == 1 else "Lavavajillas: No"

                        if variable == "Baño compartido":
                            return "Baño compartido: Sí" if v == 1 else "Baño compartido: No"

                        if variable == "Habitación privada":
                            return "Habitación privada: Sí" if v == 1 else "Habitación privada: No"

                        if variable == "Aire acondicionado":
                            return "Aire acondicionado: Sí" if v == 1 else "Aire acondicionado: No"
                        if variable == "Ascensor":
                            return "Ascensor: Sí" if v == 1 else "Ascensor: No"

                        if variable == "Licencia registrada":
                            return "Licencia registrada: Sí" if v == 1 else "Licencia registrada: No"

                        if variable == "Anuncios gestionados por el anfitrión":
                            return f"Anuncios gestionados por el anfitrión: {v:.0f}"

                        if variable == "Noches mínimas":
                            return f"Noches mínimas: {v:.0f}"
                        if variable == "Huéspedes":
                            return f"Huéspedes: {v:.0f}"
                        if variable == "Habitaciones":
                            return f"Habitaciones: {v:.0f}"
                        if variable == "Número de baños":
                            return f"Baños: {v:g}"
                        if variable == "Reseñas últimos 12 meses":
                            return f"Reseñas últimos 12 meses: {v:.0f}"
                        if "Distancia" in variable:
                            return f"{variable.replace(' (km)', '')}: {v:.1f} km"

                        return variable

                    data["label"] = data.apply(valor_texto, axis=1)
                    data["efecto"] = np.where(data["efecto_pct"] >= 0, "Aumenta", "Reduce")

                    st.markdown(f"### Precio recomendado: **{eur(precio)} €/noche**")

                    st.markdown(
                    "🟢 **Aumenta la estimación** · 🔴 **Reduce la estimación**"
                )

                    st.caption("Cada barra representa la contribución aproximada de esa característica a esta predicción respecto al valor base del modelo")

                    chart = (
                        alt.Chart(data)
                        .mark_bar()
                        .encode(
                            y=alt.Y(
                                "label:N",
                                sort=data["label"].tolist(),
                                title=None,
                                axis=alt.Axis(labelLimit=500, labelFontSize=13)
                            ),
                            x=alt.X("efecto_pct:Q",title="Efecto aproximado sobre el precio (%)",axis=alt.Axis(format="+.0f")),
                            color=alt.Color(
                                "efecto:N",
                                scale=alt.Scale(
                                    domain=["Aumenta", "Reduce"],
                                    range=["#2E8B57", "#C84A3D"]
                                ),
                                legend=alt.Legend(title=None)
                            ),
                            tooltip=[
                                alt.Tooltip("label:N", title="Característica"),
                                alt.Tooltip("efecto_pct:Q", title="Efecto aproximado", format="+.1f"),
                                alt.Tooltip("efecto:N", title="Efecto")
                            ]
                        )
                    )

                    cero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
                        color="#555555"
                    ).encode(x="x:Q")

                    st.altair_chart(
                        (chart + cero).properties(height=330),
                        width="stretch"
                    )

                    st.markdown("#### 💡 ¿Qué significa esto para ti?")
                    positivos = data[data["shap"] > 0].sort_values("shap", ascending=False)
                    negativos = data[data["shap"] < 0].sort_values("shap")

                    if len(positivos) and len(negativos):
                        st.info(f'La característica que más contribuye a un precio recomendado más alto es '
            f'**{positivos.iloc[0]["label"]}** (**+{positivos.iloc[0]["efecto_pct"]:.0f}%**). '
            f'La que más contribuye a reducirlo es '
            f'**{negativos.iloc[0]["label"]}** (**{negativos.iloc[0]["efecto_pct"]:.0f}%**).'
        )
                    st.caption("Las contribuciones SHAP muestran asociaciones aprendidas por el modelo "
            "y no deben interpretarse como efectos causales.")
    except Exception as e:
        st.warning(f"No se ha podido generar la explicación SHAP: {e}")
                

with tab_mercado:
    st.subheader(f"Anuncios similares ({len(comparables)})")

    if len(comparables) >= 10:
        q = comparables["price_eur"].quantile([0.25, 0.5, 0.75])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Percentil 25",
            f"{eur(q[0.25])} €",
            help="El 25 % de los anuncios similares cuesta menos que este precio. Es la parte baja del mercado.",)

        m2.metric("Mediana",
            f"{eur(q[0.5])} €",
            help="La mitad de los anuncios similares cuesta menos y la otra mitad más.",)

        m3.metric("Percentil 75",
            f"{eur(q[0.75])} €",
            help="El 75 % de los anuncios similares cuesta menos que este precio. Solo el 25 % más caro lo supera.",)
        m4.metric("Comparables por debajo",
    f"{(comparables['price_eur'] < precio).mean() * 100:.0f}%",
    help="Anuncios similares con un precio inferior al recomendado.")
    
                

        
##histogram
        conteo, bordes = np.histogram(comparables["price_eur"], bins=15)
        hist = pd.DataFrame({
        "desde": bordes[:-1],
        "hasta": bordes[1:],
        "anuncios": conteo
    })

        y_top = int(conteo.max())
        mediana_c = float(q[0.5])
        

        barras_h = alt.Chart(hist).mark_bar(color="#9BB7C4").encode(
            x=alt.X(
                "desde:Q",
                bin="binned",
                title="Precio por noche (€)",
                axis=alt.Axis(tickCount=10, format=".0f", labelFontSize=12)
            ),
            x2="hasta:Q",
            y=alt.Y(
                "anuncios:Q",
                title="Nº de anuncios",
                scale=alt.Scale(domain=[0, y_top * 1.25]),
                axis=alt.Axis(tickMinStep=1, format=".0f")
            ),
            tooltip=[
                alt.Tooltip("desde:Q", title="Desde (€)", format=".0f"),
                alt.Tooltip("hasta:Q", title="Hasta (€)", format=".0f"),
                alt.Tooltip("anuncios:Q", title="Anuncios")
            ]
        )
        ref_lineas = pd.DataFrame({
            "x": [precio, mediana_c],
            "tipo": ["Tu precio", "Mediana"],
            "texto": [
                f"Tu precio: {eur(precio)} €",
                f"Mediana: {eur(mediana_c)} €"
            ],
            "y": [y_top * 1.18, y_top * 1.06]
        })
        linea_precio = alt.Chart(ref_lineas[ref_lineas.tipo == "Tu precio"]).mark_rule(
            color="#C1483C",strokeWidth=3).encode(x="x:Q")
        linea_mediana = alt.Chart(ref_lineas[ref_lineas.tipo == "Mediana"]).mark_rule(
                                                                                color="#555555",
                                                                                strokeWidth=2,
                                                                                strokeDash=[6, 4]
                                                                            ).encode(x="x:Q")

        x_max = float(bordes[-1])

        def etiqueta_linea(fila, color):
            a_la_derecha = fila["x"] < bordes[0] + 0.75 * (x_max - bordes[0])
            return alt.Chart(pd.DataFrame([fila])).mark_text(
                align="left" if a_la_derecha else "right",
                dx=8 if a_la_derecha else -8,
                fontSize=14,
                fontWeight="bold",
                color=color
            ).encode(
                x="x:Q",
                y="y:Q",
                text="texto:N"
            )

        et_precio = etiqueta_linea(ref_lineas.iloc[0].to_dict(), "#C1483C")
        et_mediana = etiqueta_linea(ref_lineas.iloc[1].to_dict(), "#555555")

        st.altair_chart((barras_h + linea_mediana + linea_precio + et_mediana + et_precio).properties(height=300), width="stretch")
    else:
        st.info("Pocos anuncios similares con estos filtros.")

        

##mapa

    filas_mapa = []

    for nombre, col in BARRIOS.items():
        if col is None or nombre not in CENTROIDES:
            continue

        sub = ref[ref[col] == 1]
        if len(sub) < 15:
            continue

        lat_b, lon_b = CENTROIDES[nombre]
        filas_mapa.append({
            "barrio": nombre,
            "lat": lat_b,
            "lon": lon_b,
            "mediana": int(round(sub["price_eur"].median())),
            "n": len(sub)
        })

    if filas_mapa:
        mapa = pd.DataFrame(filas_mapa)
        lo, hi = mapa["mediana"].min(), mapa["mediana"].max()
        t = (mapa["mediana"] - lo) / max(hi - lo, 1)

        mapa["c"] = [
            [int(60 + 170*v), int(110 - 50*v), int(200 - 150*v), 205]
            for v in t
        ]
        mapa["r"] =   200 + 500 * t
        mapa["etiqueta"] = mapa["mediana"].astype(str) + " €"
        mapa["tip"] = (
            mapa["barrio"] + ": " + mapa["mediana"].astype(str)
            + " €/noche (" + mapa["n"].astype(str) + " anuncios)"
        )

        es_tuyo = mapa["barrio"] == barrio
        mapa["borde"] = [
            [25, 25, 25, 255] if x else [255, 255, 255, 230]
            for x in es_tuyo
        ]
        mapa["ancho_borde"] = np.where(es_tuyo, 70, 25)

        st.subheader("Precio mediano por barrio")

        capas = [
            pdk.Layer(
                "ScatterplotLayer", mapa,
                get_position=["lon", "lat"],
                get_fill_color="c",
                get_radius="r",
                stroked=True,
                get_line_color="borde",
                get_line_width="ancho_borde",
                pickable=True
            ),
            pdk.Layer(
                "TextLayer", mapa,
                get_position=["lon", "lat"],
                get_text="etiqueta",
                get_size=14,
                get_color=[255, 255, 255, 255],
                get_text_anchor="'middle'",
                get_alignment_baseline="'center'",
                font_weight=700,
                font_settings={"sdf": True},
                outline_width=3,
                outline_color=[0, 0, 0, 140]
            )
        ]

        if es_tuyo.any():
            pass
        else:
            lat_u, lon_u = CENTROIDES[barrio]
            capas.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    pd.DataFrame([{"lat": lat_u, "lon": lon_u}]),
                    get_position=["lon", "lat"],
                    get_fill_color=[255, 255, 255, 255],
                    get_line_color=[25, 25, 25, 255],
                    stroked=True,
                    get_line_width=40,
                    get_radius=90
                )
            )

        st.pydeck_chart(
            pdk.Deck(
                map_style="light",
                initial_view_state=pdk.ViewState(
                    latitude=41.391, longitude=2.165, zoom=12.3
                ),
                layers=capas,
                tooltip={"text": "{tip}"}
            )
        )

        st.caption(
            "Azul: barrios más baratos · Rojo: más caros. "
            "El número es el precio mediano por noche y el borde oscuro marca "
            "**tu barrio**. Pasa el ratón por un círculo para ver el número "
            "de anuncios." )

# modelo

with tab_modelo:
    st.markdown("### Sobre este proyecto")
    st.markdown(
        "**Estimación del precio de alojamientos de Airbnb en Barcelona mediante análisis de datos públicos**"
    )
    st.markdown(
        "Nuestro Trabajo de Fin de Máster analiza los factores que influyen en el precio "
        "de los alojamientos de Airbnb en Barcelona y desarrolla un modelo de aprendizaje "
        "automático capaz de estimar su precio por noche a partir de sus principales características."
    )

    st.markdown("### El dashboard")
    st.markdown(
        "Para llevar los resultados del proyecto a una herramienta práctica, hemos desarrollado "
        "este dashboard interactivo. Su objetivo es ofrecer una **estimación orientativa del precio "
        "de un alojamiento** de forma sencilla, sin necesidad de utilizar código."
    )
    st.markdown(
        "El usuario introduce las características principales de su alojamiento y obtiene un "
        "**precio recomendado por noche**. Además, puede consultar los factores que influyen "
        "en la estimación y compararla con alojamientos similares del mercado."
    )

    st.markdown("### ¿Cómo funciona este modelo?")
    st.markdown("El precio se estima mediante un algoritmo de **gradient boosting (XGBoost)**, "
    "entrenado por el **equipo del proyecto** con información de marzo de 2026 obtenida de **Inside Airbnb**. "
    "El modelo utiliza las características del alojamiento para generar una estimación "
    "del precio por noche.")
# Sobre este proyecto

    st.divider()

   

    st.markdown(
        """
        **Autores**

        Jennifer Arroyo Becerra · Natalia Kechkina Korotchenkova ·
        Fátima Guamán Tumbaco · Tatiana Zakharchenko ·
        Natalia Hernández Martín
        """
    )

    st.caption("Universidad Complutense de Madrid · Madrid · 2026")
