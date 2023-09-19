import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import linalg
import time
from Variograms_Trendfuncs import *


class OrdinaryKrigning:
    def __init__(self,Points,Zvals,Variogram=GaussianVariogram()):
        self.points=Points
        self.zvals=Zvals
        self.variogram=Variogram
        #immutable instance of the Z coordanites
        self.zvals_org = np.copy(self.zvals)

        self.vVariogram=Variogram



    def ManualParamSet(self,C,a,nugget,anisotropy_factor):
        self.a=a
        self.anisotropy_factor=anisotropy_factor
        self.C=C
        self.nugget=nugget



    def Matrixsetup(self):
        # Compute the pairwise distance matrix

        #_______Make this go brrrrrrr______#
        distances = numba_dist_matrix(self.points)
        #__________________________________#
        self.vVariogram.set_a_C(self.a,self.C)
        result=self.vVariogram(distances)

        # Add a row of ones at the bottom and a column of ones at the right
        result = np.pad(result, ((0, 1), (0, 1)), mode='constant', constant_values=1)

        # Replace the bottom-right corner with 0
        result[-1, -1] = 0

        result = result + np.eye(result.shape[0]) * self.nugget

        self.result=result



    def SinglePoint(self,Xo,Yo,training_points=None):
        if training_points is None:
            training_points = self.points


        distances_to_point0 = numba_distances_to_point0(self.points, Xo, Yo)

        #_____can probably be removed but this is more safe, if run time affect to large can be removed_______#
        #self.vVariogram.set_a_C(self.a,self.C)
        #____________________________________________________________________________________________________#

        vectorb=self.vVariogram(distances_to_point0)

        vectorb = np.append(vectorb, 1)

        lamd=linalg.solve(self.result,vectorb)

        lamd=np.delete(lamd,-1)

        zout=np.dot(lamd,self.zvals.T)
        return zout
        
#___________to be implemented_____________________
#interp grid equiveland of single point StdDev
    def SinglePointStdDev(self,Xo,Yo,training_points=None):
        if training_points is None:
            training_points = self.points    
        self.std_dev = np.sqrt(self.C - np.dot(lamd, vectorb[:-1]))
        
        point0 = np.array([Xo, Yo])

        distances_to_point0 = np.sqrt((training_points[:, 0] - Xo) ** 2 + (training_points[:, 1] - Yo) ** 2 / self.anisotropy_factor ** 2)

        vectorb = self.vVariogram(distances_to_point0, self.a,self.C)

        vectorb = np.append(vectorb, 1)

        lamd=linalg.solve(self.result,vectorb,assume_a='sym')

        lamd=np.delete(lamd,-1)

        std_dev = np.sqrt(self.C - np.dot(lamd, vectorb[:-1]))
        
        return std_dev
    
    def interpgrid(self, xmin, xmax, ymin, ymax, step):
        start=time.time()
        SingleGuess = np.vectorize(self.SinglePoint,otypes=[float])

        x_range = np.arange(xmin, xmax, step)
        y_range = np.arange(ymin, ymax, step)

        # Create a grid of points
        X, Y = np.meshgrid(x_range, y_range)

        # Stack the points into a 2D array
        points_to_estimate = np.column_stack((X.flatten(), Y.flatten()))

        z = SingleGuess(points_to_estimate[:,0],points_to_estimate[:,1])

        z = z.reshape(X.shape)
        end=time.time()

        print(f"Interp grid time {end-start}")
        return z
    
    

    def AutoOptimize(self, InitialParams=None):
        globaltime=[]
        if InitialParams is None:
            InitialParams = [np.var(self.zvals), np.max(np.sqrt(np.sum((self.points[:, None, :] - self.points[None, :, :]) ** 2, axis=-1)))/2, .001, 1]

        self.ManualParamSet(*InitialParams)
        self.params=InitialParams 
        
        def calc_r_squared_LOO(params):   
            start2=time.time() 
            predictions = []
            for i in range(len(self.points)):
                model = OrdinaryKrigning(np.delete(self.points, i, axis=0), np.delete(self.zvals, i),Variogram=self.variogram)
                model.ManualParamSet(*params)
                model.Matrixsetup()
                estimate = model.SinglePoint(*self.points[i])
                predictions.append(estimate)
            correlation_matrix = np.corrcoef(predictions, self.zvals)
            correlation_xy = correlation_matrix[0,1]
            self.LOOr2 = correlation_xy**2
            end2=time.time()
            globaltime.append(end2-start2)
            return 1 - self.LOOr2  # We subtract from 1 because we want to minimize the function

        
        # Perform the optimization
        result = minimize(calc_r_squared_LOO, self.params,method='L-BFGS-B',options={'maxiter':3})
        C_opt, a_opt, nugget_opt, anisotropy_factor_opt = result.x
        self.a=a_opt
        self.anisotropy_factor=anisotropy_factor_opt
        self.C=C_opt
        self.nugget=nugget_opt
        print(sum(globaltime))
    


    def AutoKrige(self,step=1,bounds=None):
        t0 = time.time()

        self.AutoOptimize()
        self.Matrixsetup()

        if bounds == None:
            self.zarray=self.interpgrid(xmax=np.max(self.points[:,0]),xmin=np.min(self.points[:,0]),ymax=np.max(self.points[:,1]),ymin=np.min(self.points[:,1]),step=step)
        else:
            xmax,xmin,ymax,ymin=bounds
            self.zarray=self.interpgrid(xmax=xmax,xmin=xmin,ymax=ymax,ymin=ymin,step=step)

        t1 = time.time()
        self.exetime = t1-t0
        return self.zarray
    


    def Plot(self,title='Insert_Title',xtitle='',ytitle='',saveplot=False,address='',extent=[]):
        try:
            self.zarray
        except NameError:
            print('zmap not generated')
            quit()  
        plt.imshow(self.zarray,aspect='auto',extent=extent,origin='lower')
        plt.scatter(self.points[:,0],self.points[:,1],marker='^',s=30,c=self.zvals_org)
        plt.clim(0,np.max(self.zvals_org))
        plt.colorbar()
        plt.title(title)
        plt.xlabel(xtitle)
        plt.ylabel(ytitle)
        if saveplot==True:
            plt.savefig(address)
        plt.show()
        plt.close()






class UniversalKriging(OrdinaryKrigning):
    def __init__(self, Points, Zvals, Variogram='gaussian', trendfunc='cubic'):
        super().__init__(Points, Zvals, Variogram)
        self.trend_func_setting=trendfunc

        #define trend function for UK
        trend_functions = {
            'quadratic': {
                'func': lambda a0,a1,a2,a3,a4,x, y: a0 + a1*x + a2*y + a3*x**2 + a4*y**2,
                'matrix': lambda: np.column_stack((np.ones(len(self.points)), self.points[:, 0], self.points[:, 1], self.points[:, 0]**2, self.points[:, 1]**2))
            },
            'cubic': {
                'func': lambda a0,a1,a2,a3,a4,a5,a6,x, y: a0 + a1*x + a2*y + a3*x**2 + a4*y**2 + a5*x**3 + a6*y**3,
                'matrix': lambda: np.column_stack((np.ones(len(self.points)), self.points[:, 0], self.points[:, 1], self.points[:, 0]**2, self.points[:, 1]**2, self.points[:, 0]**3, self.points[:, 1]**3))
            },
            'interaction': {
                'func': lambda a0,a1,a2,a3,x, y: a0 + a1*x + a2*y + a3*x*y,
                'matrix': lambda: np.column_stack((np.ones(len(self.points)), self.points[:, 0], self.points[:, 1], self.points[:, 0] * self.points[:, 1]))
            },
            'hyperbolic': {
                'func': lambda a0,a1,x, y: a0 + a1*np.sqrt(x**2 + y**2),
                'matrix': lambda: np.column_stack((np.ones(len(self.points)), np.sqrt(self.points[:, 0]**2 + self.points[:, 1]**2)))
            },
            'inverse': {
                'func': lambda a0,a1,a2,x, y: a0 + a1/x + a2/y,
                'matrix': lambda: np.column_stack((np.ones(len(self.points)), 1/self.points[:, 0], 1/self.points[:, 1]))
            },
            'interaction_squared': {
            'func': lambda a0,a1,a2,a3,a4,a5,a6,x, y: a0 + a1*x + a2*y + a3*x**2 + a4*y**2 + a5*x*y + a6*(x**2)*(y**2),
            'matrix': lambda: np.column_stack((np.ones(len(self.points)), self.points[:, 0], self.points[:, 1], self.points[:, 0]**2, self.points[:, 1]**2, self.points[:, 0]*self.points[:, 1], (self.points[:, 0]**2)*(self.points[:, 1]**2)))
            },
            'sincos_2nd' : {
            'func': lambda a0,a1,a2,a3,a4,a5,a6,x, y: a0 + a1*np.sin(x) + a2*np.cos(x) + a3*np.sin(y) + a4*np.cos(y) + a5*np.sin(x)**2 + a6*np.cos(y)**2,
            'matrix': lambda: np.column_stack((np.ones(len(self.points)), np.sin(self.points[:, 0]), np.cos(self.points[:, 0]), np.sin(self.points[:, 1]), np.cos(self.points[:, 1]), np.sin(self.points[:, 0])**2, np.cos(self.points[:, 1])**2))
            },
        }
        #error handling
        if trendfunc not in trend_functions:
            quit()
        #assign var for calc_trend step
        self.trend_func = trend_functions[trendfunc]['func']
        self.design_matrix_func = trend_functions[trendfunc]['matrix']
        self.trend_function = np.vectorize(self.trend_func, excluded=['trend_params'])

    def calc_trend_coefficients(self):
        # Define the residuals function
        residuals = np.vectorize(lambda trend_params: self.trend_function(trend_params, self.points[:, 0], self.points[:, 1] / self.anisotropy_factor) - self.zvals)

        # Prepare the design matrix based on the selected trend function
        X = self.design_matrix_func()

        # Estimate the trend parameters
        self.trend_params, _, _, _ = np.linalg.lstsq(X, self.zvals, rcond=None)

        # Subtract the estimated trend from the data
        self.zvals -= self.trend_function(*self.trend_params, self.points[:, 0], self.points[:, 1] / self.anisotropy_factor)


#___________to be implemented_____________________
#interp grid equiveland of single point StdDev
#in UK need to add back trend function i think? math here is weird

    def SinglePoint(self, Xo, Yo, training_points=None):
        # First, add back the trend to the actual observations
        zvals_original = self.zvals + self.trend_function(*self.trend_params, self.points[:, 0], self.points[:, 1])

        # Perform kriging on detrended data and add the trend back
        return super().SinglePoint(Xo, Yo, training_points) + self.trend_function(*self.trend_params, Xo, Yo)




    def interpgrid(self, xmin, xmax, ymin, ymax, step):
        x_range = np.arange(xmin, xmax, step)
        y_range = np.arange(ymin, ymax, step)

        # Create a grid of points
        X, Y = np.meshgrid(x_range, y_range)

        # Stack the points into a 2D array
        points_to_estimate = np.column_stack((X.flatten(), Y.flatten()))

        # Initialize an empty array for the results
        z = np.empty(points_to_estimate.shape[0])

        # For each point in the grid
        for i, point in enumerate(points_to_estimate):
            # Get the kriging estimate for the point
            z[i] = self.SinglePoint(*point)


        # Reshape the results to match the shape of the original grid
        self.zarray = z.reshape(X.shape)

        return self.zarray



    def AutoOptimize(self, InitialParams=None):
        if InitialParams is None:
            InitialParams = [np.var(self.zvals), np.max(np.sqrt(np.sum((self.points[:, None, :] - self.points[None, :, :]) ** 2, axis=-1)))/2, .001, 1]

        self.ManualParamSet(*InitialParams)
        self.params=InitialParams 
#_______________to Do_____________________
#add a quicker option LOO is too slow for some things
#probably a traditional R2 funciton
#needs to ouput a residual
        def calc_r_squared(params):    
            C, a, nugget, anisotropy_factor = params
            predictions = []
            for i in range(len(self.points)):
                model = UniversalKriging(np.delete(self.points, i, axis=0), np.delete(self.zvals, i),trendfunc=self.trend_func_setting,Variogram=self.variogram)
                model.ManualParamSet(*params)
                model.calc_trend_coefficients()
                model.Matrixsetup()
                estimate = model.SinglePoint(*self.points[i])
                predictions.append(estimate)
            correlation_matrix = np.corrcoef(predictions, self.zvals_org)
            correlation_xy = correlation_matrix[0,1]
            self.LOOr2 = correlation_xy**2
            
            return 1 - self.LOOr2  # We subtract from 1 because we want to minimize the function

        # Perform the optimization
        result = minimize(calc_r_squared, self.params,method='L-BFGS-B')
        C_opt, a_opt, nugget_opt, anisotropy_factor_opt = result.x
        self.a=a_opt
        self.anisotropy_factor=anisotropy_factor_opt
        self.C=C_opt
        self.nugget=nugget_opt



    def AutoKrige(self,step=1,bounds=None):
        t0 = time.time()

        self.AutoOptimize()
        self.calc_trend_coefficients()
        self.Matrixsetup()

        if bounds == None:
            self.zarray=self.interpgrid(xmax=np.max(self.points[:,0]),xmin=np.min(self.points[:,0]),ymax=np.max(self.points[:,1]),ymin=np.min(self.points[:,1]),step=step)
        else:
            xmin,xmax,ymin,ymax=bounds
            self.zarray=self.interpgrid(xmax=xmax,xmin=xmin,ymax=ymax,ymin=ymin,step=step)
            
        t1 = time.time()
        self.exetime = t1-t0
        return self.zarray
            

