# Pipeline de seleccion de variables segun Selec_feats para la data limpia

def seleccionar_variables(data):
    #partir data para luego entrenar con patsy y obtener las variables mas importantes
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(['price_log', 'id'], axis=1),
        data['price_log'],
        test_size=0.2,
        random_state=50
    )

    data_train = X_train.join(y_train)

    #  Fórmula completa, con todas las variables
    from statsmodels.formula.api import ols
    form_completo = ols_formula(data_train, 'price_log')

    # Patsy matrices 
    y, X_patsy = patsy.dmatrices(form_completo, data_train, return_type='dataframe')

    #  Selección con LassoLarsIC 
    from sklearn import linear_model

    reg = linear_model.LassoLarsIC(criterion='bic')
    reg.fit(X_patsy, y)

    selec_feats = X_patsy.loc[:, (reg.coef_ != 0).ravel().tolist()]
    vars_seleccionadas = selec_feats.columns.tolist()

    # Variables finales  
    var_eliminar = ['aleatorio1', 'aleatorio2']
    variables_seleccion= [c for c in vars_seleccionadas if c not in var_eliminar]


    # OUTPUT
    return  variables_seleccion

