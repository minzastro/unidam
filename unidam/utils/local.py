import numpy as np
from scipy.stats import norm, truncnorm
from unidam.skewnorm_boosted import skewnorm_boosted as skewnorm
from unidam.utils.trunc_t import trunc_t
from unidam.utils.trunc_revexpon import trunc_revexpon


def get_param(fit, par):
    """
    Convert parameters from DB to distribution parameters.
    """
    if fit == 'S':
        return skewnorm, [par[2], par[0], par[1]]
    elif fit == 'G':
        return norm, [par[0], par[1]]
    elif fit == 'T':
        sigma = np.abs(par[1])
        alpha = (par[2] - par[0]) / par[1]
        beta = (par[3] - par[0]) / par[1]
        return truncnorm, [alpha, beta, par[0], sigma]
    elif fit == 'P':
        sigma = np.abs(par[1])
        alpha = (par[3] - par[0]) / par[1]
        beta = (par[4] - par[0]) / par[1]
        return trunc_t, [par[2], alpha, beta, par[0], sigma]
    elif fit == 'L':
        sigma = np.abs(par[1])
        if par[0] < par[2]:
            par[0] = par[2] - 1e-3
        elif par[0] > par[3]:
            par[0] = par[3] + 1e-3
        alpha = (par[2] - par[0]) / sigma
        beta = (par[3] - par[0]) / sigma
        return trunc_revexpon, [alpha, beta, par[0], sigma]
    else:
        raise ValueError('Unknown fit type: %s' % fit)
    return None


def get_catalog_list(conn=None, with_filter=False, catalog_list=None,
                     with_color=False, with_text_name=False):
    """
    Get all active catalogs.
    """
    from sqlconnection import SQLConnection
    if conn is None:
        xconn = SQLConnection('z', database='sage_gap')
    else:
        if type(conn) == SQLConnection:
            xconn = conn
        else:
            xconn = SQLConnection('z', connection=conn)
    if with_filter:
        extra = ", coalesce(standard_filter, '') as filter"
    else:
        extra = ''
    if with_color:
        extra = '%s, plot_color' % extra
    if with_text_name:
        extra = '%s, text_name' % extra
    if catalog_list is not None:
        catalogs = catalog_list.split(',')
        catalogs = ["'%s'" % cat for cat in catalogs]
        filter = 'and c.name in (%s)' % ','.join(catalogs)
    else:
        filter = ''
    return xconn.exec_all("""select name %s
                               from catalogs c
                              where c.active
                                 %s""" % (extra, filter))


def get_catalog_list_extended(conn=None, as_dict=True, text_key=False):
    """
    Get all active catalogs.
    """
    from sqlconnection import SQLConnection
    if conn is None:
        xconn = SQLConnection('z', database='sage_gap')
    else:
        xconn = SQLConnection('z', connection=conn)
    result = xconn.exec_all("""select name, text_name, bib_ref, bib_tex,
                                      coalesce(standard_filter, '') as filter,
                                      plot_color
                               from catalogs c
                              where c.active
                              order by name""")
    if as_dict:
        if text_key:
            return {row[1]: row for row in result}
        else:
            return {row[0]: row[1:] for row in result}
    else:
        return result
