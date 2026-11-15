#!/bin/bash
W='/mnt/c/Users/prof.dr.f.sami/AppData/Local/Temp/claude/C--Users-prof-dr-f-sami-Desktop-BABAM--claude-worktrees-tbg-fotonik-bariyeri-chatimi-35a068/9b6a327a-26b2-469d-978b-55fdc4c53201/scratchpad/t1_hakem/hbn_tum_kanal'
cp "$W/kod/dp_dft.py" ~/t1hakem_dp/
cd ~/t1hakem_dp
rm -rf ux+1
/usr/bin/python3 dp_dft.py "$W/data/dp_sonuc.json" 2>&1 | tee dp_fg.log
for t in e0 ux+1 ux-1 sxy+1; do cp ~/t1hakem_dp/$t/nscf.out "$W/data/${t}_nscf.out" 2>/dev/null; done
