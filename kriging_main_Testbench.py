from Ok_Uk_Module import *
import numpy as np
from matplotlib import pyplot as plt


###Options###
#colum_defs
xi_col=0
yi_col=1
zi_col=4
#resolution
resolution=30
#kriging config
Params=[]
saveplot=True


loadeddata=np.loadtxt('data\surface_roughness.csv',delimiter=',',skiprows=1)
points=loadeddata[:,[xi_col,yi_col]]
zi=loadeddata[:,zi_col]
start=time.time()
krige=OrdinaryKrigning(points,zi,Variogram=PowerVariogram())
krige.AutoKrige(step=resolution)
end=time.time()
krige.Plot(f'Side SA_Power_Variogram_OK r2={round(krige.LOOr2,2)} aspect={krige.anisotropy_factor}',xtitle='Power',ytitle='Speed',saveplot=saveplot,address=f'figs',extent=[0, 900, 0, 5000])
print(krige.LOOr2)
print(krige.params)
print(f"total: {end-start}")