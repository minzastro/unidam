# UniDAM
A Unified tool to estimate Distances, Ages, and Masses (UniDAM) is
described in https://arxiv.org/abs/1705.00963 and https://arxiv.org/abs/1804.06578

## Installation
You can install UniDAM by running

```
python3 setup.py build
python3 setup.py install [--user]
```

### What you will need to run UniDAM (apart from UniDAM itself) is the following:

1. Model file (can be downloaded from https://github.com/minzastro/unidam/releases/download/2.0/PARSEC.fits for v2.0 or
https://github.com/minzastro/unidam/releases/download/1.0/parsec_models.fits for v1.0).
Path to the model file should be indicated in the configuration file (see example in https://github.com/minzastro/unidam/blob/master/unidam/iso/unidam_pdf.conf)
You can also be brave and prepare a model file yourself, using e.g. https://github.molgen.mpg.de/MPS-SAGE/parsec_isochrones
2. Data file. Should be a fits or votable, as described below

## Data file format
Data file is a fits or votable file with the following columns:

| Value   | Units       |
|---------|-------------|
| id      |             |
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

If you know the parallax (e.g. from Gaia), you can add the following columns (column names can be different, but this should be reflected in the configuration file):

| Value         | Units       |
|---------------|-------------|
| parallax      | arcsec      |
| parallax_error| arcsec      |
| extinction    | mag         |
| extinction_error| mag       |

Magnitudes are for 2MASS and AllWISE. If you do not want to bother with AllWISE, you can leave the corresponding columns empty.
Extinction value is a soft upper limit of A_K value, with extinction prior being flat below the limit and decreasing exponentially (with width of the given extinction_error) above that.

Other columns are possible, but they will be ignored.

## Configuration file format
Configuration file is a config-like file with key=value settings.
An example of the config file is given below:

```INI
[general]
# Path to the model file
model_file=PARSEC.fits

# Store stacked full PDF data in distance modulus, log(age) and 
# 2D (distance modulus vs log(age)) PDFs
dump_pdf=True

# Folder name for debug dump data (activated by -d switch, default=dump)
dump_prefix=dump_gaia

# Distance prior values are:
# 0: no prior,
# 1: volume prior (default)
# 2: decreasing exponential density prior (experimental)
distance_prior=1

# These columns are passed to output without changes.
keep_columns=

# Maximum difference between model and observed values in sigmas (default=4)
# Setting higher values can decrease the calculations dramatically with little effect on results
# Setting lower values speed things up, but has impact on the outcome.
max_param_err=4.

# Wether to allow extinction to be negative (see discussion in sec 4.4 of Mints&Hekker 2017).
allow_negative_extinction=False

# This refers to columns in the model_file. 
# Input file columns should have the following columns: 
# For each column A from model_columns 
# 1) a column A with a value
# 2) a column dA with the uncertainty
model_columns=T,logg,feh

# For each band column A there should be 
# 1) a column Amag with a value
# 2) a column e_Amag with the uncertainty
band_columns=J,H,K,W1,W2

# These columns will be used to derive output parameters
# Fitted column names are taken from the model file.
# any of [distance_modulus,distance,extinction,parallax] can also be used.
fitted_columns=stage,age,mass,distance_modulus,distance,extinction,parallax

# If parallax is known, parallax and extinction sections below are required.
parallax_known=True

[parallax]
column=parallax
err_column=parallax_error

[extinction]
column=extinction
err_column=extinction_error

```


## Running UniDAM

### Batch mode
You have to call
```Shell
python3 unidam_runner.py -i input_file -o output_file -c config_file(e.g. unidam_pdf.conf)
```
you can add `-p` flag for parallel execution, number of processes is set by `OMP_NUM_THREADS` shell variable.

### Debug mode
You have to call
```Shell
python3 unidam_runner.py -d --id IDs -i input_file -o output_file -c config_file(e.g. unidam_pdf.conf)
```
where `IDs` is a comma-separated list of IDs from the input file. In this case two files will be created for each ID:
 * `{dump_prefix}/dump_{ID}.dat` -- an ASCII table with parameters for all models that fit.
 * `{dump_prefix}/dump_{ID}.json` -- JSON file with extended data for each solution.

`dump_prefix` is taken from the config file in this case (default = `dump`)

## Output file format
| Column name                                    |  Units  | Description                                                               |
|------------------------------------------------|:-------:|---------------------------------------------------------------------------|
| id                                             |         | Unique ID of the star from the input data                                 |
| stage                                          |         | Stage number (I, II or III)                                               |
| uspdf\_priority                                |         | Priority order of a given USPDF (starting from 0)                         |
| uspdf\_weight                                  |         | Weight of a given USPDF $V_m$                                             |
| total\_uspdfs                                  |         | Number of USPDF with $V_m > 0.03$                                         |
| p\_best                                        |         | Probability for a best-fitting model  |
| p\_sed                                         |         | p-value from $\chi^2$ SED fit     |
| quality                                        |         | Quality flag (see below)                              |
| distance_modulus_smooth | mag | smoothing used in distance modulus PDF calculations |
| extinction_smooth | mag | smoothing used in extinction PDF calculations |
| extinction_zero |   | fraction of the PDF with zero extinction |
| **For every *param* in the fitted_columns list** |||
| param_mean                    |      | Mean value of param PDF                                               |
| param_err                     |         | Standard deviation of param PDF                           |
| param_mode                      |         | Mode of param PDF                                   |
| param_median                   |         | Median of param PDF                                   |
| param_fit                      |      |  Letter indicating the fitted functon                 |
| param_par                                 |  | Parameters of the fitted function |
| **Distance modulus - logarithm of age relation:**            || |
| dm\_age\_slope                                 |         | Slope of the relation                                                     |
| dm\_age\_intercept                             |         | Intercept of the relation                                                 |
| dm\_age\_scatter                               |         | Scatter of the relation                                                   |
| dm\_age\_mad                                  |         | Median absolute deviation of the relation                                                   |
| **Distance modulus - logarithm of mass relation:** |         |                                                                           |
| dm\_mass\_slope                                |         | Slope of the relation                                                     |
| dm\_mass\_intercept                            |         | Intercept of the relation                                                 |
| dm\_mass\_scatter                              |         | Scatter of the relation                                                   |
| dm\_mass\_mad                                  |         | Median absolute deviation of the relation                                                   |
