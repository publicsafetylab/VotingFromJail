#!/bin/bash

# Run data prep, balance and modeling for registered voters, p > 0.75.
cd matched_bookings || exit
python3 prep_data.py -c score_weighted -r -t 0.75 -xc
python3 balance_iterator.py -c score_weighted -r -t 0.75 -xc
python3 model_balance.py -c score_weighted -r -t 0.75 -xc
python3 model_turnout.py -c score_weighted -r -t 0.75 -xc
python3 model_turnout_placebo.py -c score_weighted -r -t 0.75 -xc
python3 model_turnout_heterogeneous.py -c score_weighted -r -t 0.75 -xc
python3 model_turnout_heterogeneous.py -c score_weighted -r -t 0.75 -xc -xr

# Run data prep, balance and modeling for registered voters, p > 0.95.
python3 prep_data.py -c score_weighted -r -t 0.95 -xc
python3 balance_iterator.py -c score_weighted -r -t 0.95 -xc
python3 model_balance.py -c score_weighted -r -t 0.95 -xc
python3 model_turnout.py -c score_weighted -r -t 0.95 -xc
python3 model_turnout_placebo.py -c score_weighted -r -t 0.95 -xc
python3 model_turnout_heterogeneous.py -c score_weighted -r -t 0.95 -xc
python3 model_turnout_heterogeneous.py -c score_weighted -r -t 0.95 -xc -xr

# Run data prep, balance and modeling for all voters, p > 0.95.
python3 prep_data.py -c score_weighted -t 0.95 -xc
cd ../full_bookings || exit
python3 prep_data.py -c score_weighted -t 0.95 -xc
python3 balance_iterator.py -c score_weighted -t 0.95 -xc
python3 model_balance.py -c score_weighted -t 0.95 -xc
python3 model_match_in.py -c score_weighted -t 0.95 -xc
python3 model_turnout.py -c score_weighted -t 0.95 -xc
