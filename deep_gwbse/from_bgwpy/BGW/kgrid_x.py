import warnings
# TODO: implement python version of kgrid.x

def write_espresso(filename):
    with open(filename, "w") as f:
        f.write("K_POINTS crystal\n")
        f.write(f"kgrid_x.py is NOT AVAILABLE YET!!!")
        f.write(f"It will be coming soon!!!")
        f.write(f"--------------")
        f.write(f"To generate right k-points")
        f.write(f"please make sure <BerkeleyGW/bin/kgrid.x> is in your PATH!!!.")
    print('Warning: Make sure <BerkeleyGW/bin/kgrid.x> in your PATH!!! (intergrated kgrid.x version will be coming soon!)')

def write_log(filename):
    with open(filename, "w") as f:
        f.write(f"kgrid_x.py is NOT AVAILABLE YET!!!")
        f.write(f"It will be coming soon!!!")
        f.write(f"--------------")
        f.write(f"To generate right k-points")
        f.write(f"please make sure <BerkeleyGW/bin/kgrid.x> is in your PATH!!!.")

def kgrid_x_main(fi:str, fo:str, flog:str):
    write_espresso(fo)
    write_log(flog)
