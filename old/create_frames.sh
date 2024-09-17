#!/usr/bin/env bash

dir='slabs/ML3_20x20_ol_ac_hexane'

for i in {2..16}
do
    echo $i
    python npl_slab.py > /dev/null 2>&1
    mv NPLs_ML3_20x20_ol_ac.lmp ${dir}/NPLs_ML3_20x20_ol_ac.${i}.lmp
done
