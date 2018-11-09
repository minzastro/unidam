#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 13:31:04 2018

@author: mints
"""


from sqlconnection import SQLConnection

conn = SQLConnection('convert', database='sage_gap')

for row in conn.exec_dict('select * from catalogs', return_all=True):
    #keep = []
    #for key in ['id', 'ra', 'de', 'l', 'b',
    #            't', 'logg', 'feh',
    #            'dt', 'dlogg', 'dfeh']:
    #    #if row[key] is not None:
    #    #    keep.append("'" + row[key] + "'")

    data = f"""keep=
[mapping]
# Mapping of columns. Columns will be renamed according to
# this mapping.
id={row['id']}
T={row['t']}
logg={row['logg']}
feh={row['feh']}
# uncertainties columns. If the format is !value
# than value is taken as an uncertainty.
dT={row['dt']}
dlogg={row['dlogg']}
dfeh={row['dfeh']}

[galactic]
longitude={row['l']}
lattitude={row['b']}"""
    with open(f'{row["name"]}.conf', 'w') as file:
        file.write(data)
    print(row['name'])