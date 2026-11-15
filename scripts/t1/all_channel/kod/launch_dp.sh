#!/bin/bash
W='/path/to/work/t1_hakem/hbn_tum_kanal'
mkdir -p ~/t1hakem_dp
cp "$W/kod/dp_dft.py" ~/t1hakem_dp/
cd ~/t1hakem_dp
PY=/usr/bin/python3
[ -x "$PY" ] || PY=python3
$PY -c 'import numpy; print("np", numpy.__version__)'
nohup setsid $PY dp_dft.py "$W/data/dp_sonuc.json" > dp.log 2>&1 &
echo started
