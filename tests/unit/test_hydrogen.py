import tbmodels

hr_file  = "/mnt/c/Users/Hanada Tatsuki/Desktop/sol2liq/data/input/H/wanner90_hr.dat"
win_file = "/mnt/c/Users/Hanada Tatsuki/Desktop/sol2liq/data/input/H/wanner90.win"

model = tbmodels.Model.from_wannier_files(hr_file=hr_file, win_file=win_file)

