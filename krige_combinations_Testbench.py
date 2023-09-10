from Ok_Uk_Module import *
import numpy as np
from matplotlib import pyplot as plt


###Options###
#colum_defs
xi_col=0
yi_col=1
zi_col=4
#resolution
resolution=100
#kriging config
Params=[]
combs=[('gaussian', 'quadratic'),
 ('gaussian', 'cubic'),
 ('gaussian', 'interaction'),
 ('gaussian', 'hyperbolic'),
 ('gaussian', 'inverse'),
 ('gaussian', 'interaction_squared'),
 ('spherical', 'quadratic'),
 ('spherical', 'cubic'),
 ('spherical', 'interaction'),
 ('spherical', 'hyperbolic'),
 ('spherical', 'inverse'),
 ('spherical', 'interaction_squared'),
 ('exponential', 'quadratic'),
 ('exponential', 'cubic'),
 ('exponential', 'interaction'),
 ('exponential', 'hyperbolic'),
 ('exponential', 'inverse'),
 ('exponential', 'interaction_squared'),
 ('linear', 'quadratic'),
 ('linear', 'cubic'),
 ('linear', 'interaction'),
 ('linear', 'hyperbolic'),
 ('linear', 'inverse'),
 ('linear', 'interaction_squared'),
 ('power', 'quadratic'),
 ('power', 'cubic'),
 ('power', 'interaction'),
 ('power', 'hyperbolic'),
 ('power', 'inverse'),
 ('power', 'interaction_squared')]

for i in combs:

    Variogram=i[0]
    trendfunc=i[1]
    #plotting
    saveplot=True


    loadeddata=np.loadtxt('data\surface_roughness.csv',delimiter=',',skiprows=1)
    points=loadeddata[:,[xi_col,yi_col]]
    zi=loadeddata[:,zi_col]

    krige=UniversalKriging(points,zi,Variogram=Variogram,trendfunc=trendfunc)
    krige.AutoKrige(step=resolution)
    krige.Plot(f'Top SA_{Variogram}_Variogram_UK r2={krige.LOOr2}',xtitle='Power',ytitle='Speed',saveplot=saveplot,address=f'figs\Top_SA_{Variogram}_{trendfunc}_UK.png',extent=[0, 900, 0, 5000])
    print(krige.LOOr2)
    print(krige.params)
    
