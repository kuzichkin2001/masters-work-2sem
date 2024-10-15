# Insert into `AnnNeuronBaseSplit` class in `masters_work.ipynb`.

@staticmethod
def jac_model(model_dir, var_names, prm_file, normalize=True):

    with open(prm_file, 'r') as file:
        prm = json.loads(file.read())
    
    model = tf.keras.models.load_model(model_dir)

    dim_p = prm['dim_p']
    dim_u = prm['dim_u']
    stab_d = prm['stab_d']
    stab_u = prm['stab_u']
   
    inp_p = Input(shape=(dim_p,))
    inp_u = Input(shape=(dim_u,))

    if normalize:
        norm_p = Normalizator(prm['loc_p'], prm['scl_p'])
        norm_u = Normalizator(prm['loc_u'], prm['scl_u'])
    else:
        norm_p = NullNormalizator()
        norm_u = NullNormalizator()

    pp = norm_p(inp_p)
    u1 = norm_u(inp_u)

    models = [AnnNeuronBaseSplit._jac_row(
        nvar, dim_p, dim_u, stab_d, stab_u,
        model.get_layer(name=name))
        for nvar, name in enumerate(var_names)]

    jac = [m([pp, u1]) for m in models]

    jac = tf.stack(jac, axis=1)

    model_jac = Model(inputs=[inp_p, inp_u], outputs=jac,
                      name='sherman_jac')
    model_jac.compile()
    return model_jac

@staticmethod
def _jac_row(nvar, dim_p, dim_u, stab_d, stab_u, model):

    inp_p = Input(shape=(dim_p,), name='inp_p')
    inp_u = Input(shape=(dim_u,), name='inp_u')

    dense_prm = model.get_layer(name='dense_prm')
    dense_hid = model.get_layer(name='dense_hid')
    activat_hid = model.get_layer(name='activat_hid')
    dense_out = model.get_layer(name='dense_out')
    add = model.get_layer(name='add')
    con = model.get_layer(name='con')

    u0 = inp_u[:, nvar:nvar+1]
    inx = list(range(dim_u))
    del inx[nvar]
    pu = tf.gather(inp_u, inx, axis=1)

    px = dense_prm(con([inp_p, pu]))  # это g(.)
    g_prime = 1 - px * px  # это g'(.)
    ux = dense_hid(u0)
    ux = add([ux, px])
    ux = activat_hid(ux)  # это f(.)
    f_prime = 1 - ux * ux  # это f'(.)

    bi = dense_out.weights[0][:, 0]
    # ai = dense_hid.weights[0][0, :]
    ai = dense_hid.weights[0]
    db = f_prime * bi
    ddb = g_prime * db
    A = dense_prm.weights[0][1:, :]

    jac_ii = tf.reshape(stab_u + stab_d * tf.matmul(db, ai, transpose_b=True), (-1,))

    jac_row = [stab_d * tf.linalg.matvec(ddb, row) for row in A]
    jac_row.insert(nvar, jac_ii)

    jac_out = tf.stack(jac_row, axis=1)

    return Model(inputs=[inp_p, inp_u],
                 outputs=jac_out,
                 name=model.name + '_jac')