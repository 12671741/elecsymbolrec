import numpy as np

width=1280/2
height=720/2
kernal = np.ones((4,4),np.uint8)
flip=1
croph=16
cropw=16


str10=["ground","DC votage","DC votage","current meter","AC source","resistor","resistor","inductor","capacitor","diode"]

str=["ground","ground","ground","ground","DC votage meter","DC votage",\
    "DC votage","DC votage","DC votage","current meter","AC source",\
    "resistor","resistor","resistor","resistor","inductor","inductor",\
    "capacitor","diode","diode",'wire','wire','wire','wire','wire','blank']
