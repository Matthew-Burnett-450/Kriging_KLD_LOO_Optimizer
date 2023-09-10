from Ok_Uk_Module import *
import numpy as np
from matplotlib import pyplot as plt


###Options###
#colum_defs
xi_col=0
yi_col=1
zi_col=5
#resolution
resolution=10
#kriging config
Params=[]

Variogram='power'

    #plotting
saveplot=True


loadeddata=np.loadtxt('data\surface_roughness.csv',delimiter=',',skiprows=1)
points=loadeddata[:,[xi_col,yi_col]]
zi=loadeddata[:,zi_col]

krige=UniversalKriging(points,zi,Variogram=Variogram,trendfunc='cubic')
krige.AutoKrige(step=resolution)
krige.Plot(f'Side SA_{Variogram}_Variogram_OK r2={round(krige.LOOr2,2)}_interaction',xtitle='Power',ytitle='Speed',saveplot=saveplot,address=f'figs\Side_SA_{Variogram}_UK.png',extent=[0, 900, 0, 5000])
print(krige.LOOr2)
print(krige.params)