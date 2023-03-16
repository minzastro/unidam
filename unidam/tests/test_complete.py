import unittest
import numpy as np
from astropy.table import Table
from ..core.unidam_main import UniDAMTool
from unidam.core.model_fitter import model_fitter as mf


class IntegrationTest(unittest.TestCase):

    def test_run_complete(self):
        de = UniDAMTool(config_filename="test_data/unidam_parallax.conf")
        data = Table.read("test_data/Bensby.fits")
        de.id_column = 'id'  # Default ID column.
        mf.use_model_weight = True
        i = 0
        idtype = data.columns[de.id_column].dtype
        final = de.get_table(data, idtype)
        for xrow in data:
            result = de.process_star(xrow, dump=False)
            if result is None:
                continue
            elif isinstance(result, dict):
                continue
            for new_row in result:
                final.add_row(new_row)
            i += 1
        final.meta.update(de.config)
        reference = Table.read('test_data/Bensby_output.fits')
        print(reference.meta)
        print(final.meta)
        for key, value in reference.meta.items():
            assert key.lower() in final.meta
            assert value == final.meta[key.lower()]
        assert len(reference) == len(final)
        assert np.all(reference == final)
