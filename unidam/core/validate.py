#!/usr/bin/env python3
import sys
import numpy as np
from astropy.io import fits
from astropy.table import Table

main = Table.read(sys.argv[1])
test = Table.read(sys.argv[2])

for row in test:
    main_row = main[main['id'] == row['id']]
    main_row = main_row[main_row['uspdf_priority'] == row['uspdf_priority']][0]
    for column in test.colnames:
        if column not in main.colnames:
            continue
        if np.any(row[column] != main_row[column]):
            print(column, row[column], main_row[column], row[column] - main_row[column])
