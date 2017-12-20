#!/bin/bash
./unidam_runner.py -c unidam_pdf.conf -i ~/data/prepared/$1.fits -d --id="$2"
python plot_parts.py -i "$2"
feh dump/dump_$2age.png