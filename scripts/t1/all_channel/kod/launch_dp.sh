#!/bin/bash
W='/mnt/c/Users/prof.dr.f.sami/AppData/Local/Temp/claude/C--Users-prof-dr-f-sami-Desktop-BABAM--claude-worktrees-tbg-fotonik-bariyeri-chatimi-35a068/9b6a327a-26b2-469d-978b-55fdc4c53201/scratchpad/t1_hakem/hbn_tum_kanal'
mkdir -p ~/t1hakem_dp
cp "$W/kod/dp_dft.py" ~/t1hakem_dp/
cd ~/t1hakem_dp
PY=/usr/bin/python3
[ -x "$PY" ] || PY=python3
$PY -c 'import numpy; print("np", numpy.__version__)'
nohup setsid $PY dp_dft.py "$W/data/dp_sonuc.json" > dp.log 2>&1 &
echo started
