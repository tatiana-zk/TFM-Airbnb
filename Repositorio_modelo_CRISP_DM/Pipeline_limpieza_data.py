# -*- coding: utf-8 -*-
"""

@author: 
Grupo 1 TFM Airbnb
Arroyo Becerra Jennifer
Guaman Tumbaco Fatima
Hernandez Martin Natalia
Kechkina Korotchenkova Natalia
Zakharchenko Tatiana

"""
# Pipeline para limpiar y procesar la data en raw de Listings de Airbnb de Barcelona
def preparar_limpieza(data):

    # 1. LIMPIEZA BÁSICA

    data = data.drop(columns=[
        'listing_url','picture_url','host_url','host_profile_url','host_picture_url'
    ], errors='ignore')

    # Booleanas
    bools = ['host_is_superhost','host_has_profile_pic','host_identity_verified','has_availability']
    data[bools] = data[bools].replace({"t": 1, "f": 0}).astype("bool")

    # Fechas
    cols_fechas = [
        'last_scraped','price_quote_checkin_date','price_quote_checkout_date',
        'calendar_last_scraped','first_review','last_review'
    ]
    data[cols_fechas] = data[cols_fechas].apply(
        lambda col: pd.to_datetime(col, format="%Y-%m-%d", errors="coerce")
    )

    #Categoricas
    cols_categoricas = [ 'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'property_type',
    'room_type']
    data[cols_categoricas] = data[cols_categoricas].astype("category")

    # Strings
    cols_string = [
        'amenities','source','name','host_name','description','host_location',
        'host_about','bathrooms_text','price','price_quote_raw','license'
    ]
    data[cols_string] = data[cols_string].astype("string")

    data = data.dropna(axis=1, how='all')

    # 2. TRANSFORMACIONES Y CREACION DE NUEVAS VARIABLES

    data['price'] = data['price'].replace('[\\$,]', '', regex=True).astype(float)
    data = data.dropna(subset=['price'])

    data["bathrooms_num"] = data["bathrooms_text"].str.extract(r"([\d.]+)").astype(float)
    data["shared_bathroom"] = data["bathrooms_text"].str.contains("shared", case=False, na=False)

    # Gran tenedor
    host_prop = (
        data.assign(coord=data["latitude"].astype(str) + "_" + data["longitude"].astype(str))
            .groupby("host_id")["coord"].nunique()
            .rename("n_propiedades_distintas")
    )
    data = data.merge(host_prop, on="host_id", how="left")
    data["es_gran_tenedor"] = data["n_propiedades_distintas"] >= 5

    data['tiene_biografia'] = data["host_about"].notna()

    # Licencias
    def clasificar_licencia(val):
        if pd.isna(val):
            return 'Sin dato'
        v = val.upper()
        if re.search(r'(HUTB|HB-|AJ\d|ES[A-Z]{2,4}\d{5,})', v):
            return 'Licencia registrada'
        if 'EXEMPT' in v:
            return 'Exento'
        return 'Otro'

    data['licencia_estado'] = data['license'].apply(clasificar_licencia).astype('category')

    # Agrupar barrios
    umbral = 150
    conteo = data["neighbourhood_cleansed"].value_counts()
    data["neighbourhood_agrupado"] = (
        data["neighbourhood_cleansed"]
        .apply(lambda x: x if conteo[x] >= umbral else "Otros")
        .astype("category")
    )

    # Antigüedades
    data['antiguedad_como_user'] = data['hosts_time_as_user_years']*12 + data['hosts_time_as_user_months']
    data['antiguedad_como_host'] = data['hosts_time_as_host_years']*12 + data['hosts_time_as_host_months']
    data['sin_reviews'] = data['reviews_per_month'].isna()
    data['host_location'].fillna('Desconocido', inplace=True)
    data['antiguedad_anuncio'] = (data['last_scraped'] - data['first_review']).dt.days

    # Agrupar categorías raras
    def agrupar_raras(serie, umbral_pct=1.0):
        freq = serie.value_counts(normalize=True)*100
        comunes = freq[freq >= umbral_pct].index
        return serie.apply(lambda x: x if x in comunes else 'Otros')

    data['host_location_agrupado'] = agrupar_raras(data['host_location']).astype('category')
    data['property_type_agrupado'] = agrupar_raras(data['property_type']).astype('category')

    data['tiene_descripcion'] = data['description'].notna()

    # 3. IMPUTACIONES
    #Imputacion por moda
    data[['review_scores_location', 'review_scores_rating', 'reviews_per_month']] = data[['review_scores_location', 'review_scores_rating', 'reviews_per_month']].fillna(0)
    #Imputacion por mediana
    data['antiguedad_anuncio'] = data['antiguedad_anuncio'].fillna(data['antiguedad_anuncio'].median())
    #rellenar nulos por desconocidos o ceros
    data['host_name'] = data['host_name'].fillna('Desconocido')

    for col in ['host_has_profile_pic','host_identity_verified','has_availability']:
        data[col] = data[col].fillna(data[col].mode()[0])

    # Imputacion por Iterative imputers
    cols_iter = ['bedrooms','bathrooms_num','accommodates']
    data[cols_iter] = IterativeImputer(max_iter=10, random_state=42).fit_transform(data[cols_iter])

    cols_chain = ["host_listings_count","antiguedad_como_user","antiguedad_como_host",
                  "maximum_nights","minimum_nights"]
    data[cols_chain] = skl_imp.IterativeImputer(max_iter=20, random_state=0).fit_transform(data[cols_chain])



    # 4. GESTION DE OUTLIERS

    cols_wins = [
        'host_listings_count','calculated_host_listings_count','n_propiedades_distintas',
        'number_of_reviews','number_of_reviews_ltm','number_of_reviews_l30d',
        'reviews_per_month','accommodates'
    ]
    for col in cols_wins:
        q1, q3 = data[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lim_inf = max(q1 - 1.5*iqr, data[col].min())
        lim_sup = q3 + 1.5*iqr
        data[col] = data[col].clip(lim_inf, lim_sup)

    data.loc[data['maximum_nights'] > 100_000, 'maximum_nights'] = 1125


    # 5. FEATURE ENGINEERINGS
    # CREACION DE NUEVAS VARIABLES DESDE AMENITIES

    amen = {
        'has_ac': 'Air conditioning',
        'has_elevator': 'Elevator',
        'has_workstation': 'Dedicated workspace',
        'has_dishwasher': 'Dishwasher',
        'has_balcony': 'balcony',
        'has_parking': 'parking',
        'has_pool': 'Pool',
        'has_self_checkin': 'Self check-in'
    }
    for col, patt in amen.items():
        data[col] = data['amenities'].str.contains(patt, case=False, na=False)
    data['price_log'] = np.log(data['price'])
    data = data.dropna(subset=['price_log'])

    # CREACION DE VARIABLES DE DISTANCIAS A POIs

    geo = gpd.read_file("geo_barcelona.geojson").to_crs(25831)
    geo.index = [
    "dist_center_km",
    "dist_beach_km",
    "dist_sagrada_familia_km",
    "dist_park_guell_km",
    "dist_camp_nou_km"
    ]
    gdata = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data["longitude"], data["latitude"]),
        crs="EPSG:4326"
    ).to_crs(25831)

    for nombre, punto in geo.geometry.items():
      gdata[nombre] = gdata.geometry.distance(punto)/1000

    # 6. PRESELECCION DE VARIABLES

    cols_num = ['id','accommodates','bedrooms','number_of_reviews_ltm',
    'availability_eoy','review_scores_rating','review_scores_location','calculated_host_listings_count',
    'bathrooms_num','antiguedad_anuncio','antiguedad_como_host','maximum_nights',
    'minimum_nights','price_log','dist_center_km','dist_beach_km',
    'dist_sagrada_familia_km','dist_park_guell_km','dist_camp_nou_km']


    cols_cat = [
        'host_has_profile_pic','host_identity_verified','room_type','has_availability',
        'shared_bathroom','es_gran_tenedor','tiene_biografia','licencia_estado',
        'neighbourhood_agrupado','sin_reviews','host_location_agrupado',
        'property_type_agrupado','tiene_descripcion','has_ac','has_elevator',
        'has_workstation','has_dishwasher','has_balcony','has_parking',
        'has_pool','has_self_checkin'
    ]
    cols_mantener = cols_num + cols_cat
    data = gdata[cols_mantener].copy()

    # 7. TRANSFORMACION DE VARIABLES CATEGORICAS A DUMMIES
    data = pd.get_dummies(data, columns=cols_cat, drop_first=True)

    # 8. LIMPIEZA DE NOMBRES DE VARIABLES
    data.columns = (
        data.columns
        .str.replace(' ', '_')
        .str.replace("'", '_')
        .str.replace('-', '_')
        .str.replace(',', '_')
        .str.replace(r'[^\w]', '', regex=True)
        .str.replace('__+', '_', regex=True)
        .str.strip('_')
    )

    # 9. TRANSFORMACION DE BOOLEANAS A INT
    lista_bool = data.select_dtypes(include=['bool','boolean']).columns
    data[lista_bool] = data[lista_bool].astype(int)

    return data

