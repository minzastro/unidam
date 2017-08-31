# UniDAM
A Unified tool to estimate Distances, Ages, and Masses (UniDAM) is
described in https://arxiv.org/abs/1705.00963

What you will need to run UniDAM (apart from UniDAM itself) is the following:

1) Model file (can be downloaded from https://github.com/minzastro/unidam/releases/download/1.0/parsec_models.fits). Path to the model file should be indicated in the configuration file (see example in https://github.com/minzastro/unidam/blob/master/unidam/iso/unidam_pdf.conf)

2) Data file. Should be a fits or votable with the following columns:

| Value   | Units       |
|---------|-------------|
| T       | K           |
| logg    | log(cm*s-2) |
| feh     | dex         |
| dT      | K           |
| dlogg   | $log(cm*s^{-2})$ |
| dfeh    | dex         |
| Jmag    | mag         |
| e_Jmag  | mag         |
| Hmag    | mag         |
| e_Hmag  | mag         |
| Kmag    | mag         |
| e_Kmag  | mag         |
| W1mag   | mag         |
| e_W1mag | mag         |
| W2mag   | mag         |
| e_W2mag | mag         |

Magnitudes are for 2MASS and AllWISE. If you do not want to bother with AllWISE, you can leave the corresponding columns empty. Other columns are possible, but they will be ignored.

Then you have to call

python2 unidam_runner.py -i input_file -o output_file -c config_file(e.g. unidam_pdf.conf)

you can add -p flag for parallel execution, number of processes is set by OMP_NUM_THREADS shell variable.
