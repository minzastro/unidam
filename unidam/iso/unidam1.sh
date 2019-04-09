#!/bin/bash
./unidam_runner.py -c unidam_mini.conf -i ~/data/prepared/$1.fits -d --id="$2"
python3 ../tools/plot_parts.py -i "$2" --dev -w A
feh dump/dump_$2age.png