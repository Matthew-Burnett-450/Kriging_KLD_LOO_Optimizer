from Ok_Uk_Module import *
from scipy.special import kl_div
from scipy.interpolate import griddata

class SpacialSensitivityAnalysisOK(OrdinaryKrigning):


    def __init__(self, Points, Zvals, Variogram='gaussian',DiverganceModel='KLD', radius=10):
        super().__init__(Points, Zvals, Variogram)
        self.DivModel=DiverganceModel
        self.radius=radius

        if self.DivModel == 'KLD':
            def DivModel(p, q):
                m = 0.5 * (p + q)
                kld=np.mean(0.5 * np.sum(kl_div(p, m)) + 0.5 * np.sum(kl_div(q, m)))
                return kld
        self.DivModel = np.vectorize(DivModel,otypes=[np.float64])
    def remove_neighbors(self, index, radius):
#_______________To do_______________________
#catch if r is too big and deletes all points and if deletes too many return warning
        point = self.points[index]
        distances = np.linalg.norm(self.points - point, axis=1)
        neighbors_indices = np.where(distances <= radius)[0]
        return np.delete(self.points, neighbors_indices, axis=0), np.delete(self.zvals, neighbors_indices)

    def DiverganceLOO(self,step=10):

        #set up the initial kriging model
        params=[self.C, self.a, self.nugget, self.anisotropy_factor]
        self.divscores=[]
        #LOO cross validation loop
        for i in range(len(self.points)):
            new_points, new_zvals = self.remove_neighbors(i, self.radius)
            print('Iterations: ',i+1,'/',len(self.points)+1)
            model = OrdinaryKrigning(new_points,new_zvals,Variogram=self.variogram)
            estimate = model.AutoKrige(step=step,bounds=[np.max(self.points[:,0]),np.min(self.points[:,0]),np.max(self.points[:,1]),np.min(self.points[:,1])])
            self.divscores.append((np.mean(self.DivModel(np.abs(self.zarray/np.sum(self.zarray)),np.abs(estimate/np.sum(estimate))))))
        np.save('div_scores.npy', self.divscores)
        return self.divscores
    

    def plot_div(self,resolution=1000,saveplot=False,powerscale=1):
            plt.style.use('ggplot')
            #check if divscores file exists if so load it if not save it
            try:
                self.divscores=np.load('div_scores.npy')
            except:
                ValueError('div_scores.npy does not exist')

            #interpolate the divergance scores in 3d space with x and y as the input and divscores as the output
            x=np.linspace(np.min(self.points[:,0]),np.max(self.points[:,0]),resolution)
            y=np.linspace(np.min(self.points[:,1]),np.max(self.points[:,1]),resolution)
            X,Y=np.meshgrid(x,y)
            Z=griddata(self.points,self.divscores,(X,Y),method='linear')
            
            #plot the interpolated divergance scores
            im = plt.imshow(Z**powerscale, cmap='YlOrRd', interpolation='bilinear', origin='lower',aspect='auto', extent=[np.min(self.points[:,0]),np.max(self.points[:,0]),np.min(self.points[:,1]),np.max(self.points[:,1])])
            plt.scatter(self.points[:,0],self.points[:,1],c='k',s=15)
            #plot configuration
            plt.title('LOO Divergance Values')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.colorbar(im)
            plt.grid(color='black', linestyle='-', linewidth=0.5,alpha=.25)
            #set x and y lims to 0  
            plt.xlim(0)
            plt.ylim(0)

            plt.show()
            plt.close()
            if saveplot==True:
                plt.savefig('div_scores.png')
            plt.show()
            plt.close()
            return Z
        
class SpacialSensitivityAnalysisUK(UniversalKriging):


    def __init__(self, Points, Zvals, Variogram='gaussian',DiverganceModel='KLD',trendfunc='linear',radius=10):
        super().__init__(Points, Zvals, Variogram,trendfunc)
        self.DivModel=DiverganceModel
        self.radius=radius
        if self.DivModel == 'KLD':
            def DivModel(p, q):
                m = 0.5 * (p + q)
                kld=np.mean(0.5 * np.sum(kl_div(p, m)) + 0.5 * np.sum(kl_div(q, m)))
                return kld
        self.DivModel = np.vectorize(DivModel,otypes=[np.float64])

    def remove_neighbors(self, index, radius):
#_______________To do_______________________
#catch if r is too big and deletes all points and if deletes too many return warning
        point = self.points[index]
        distances = np.linalg.norm(self.points - point, axis=1)
        neighbors_indices = np.where(distances <= radius)[0]
        return np.delete(self.points, neighbors_indices, axis=0), np.delete(self.zvals, neighbors_indices)
    def DiverganceLOO(self,step=10,manualbounds=None):

        #set up the initial kriging model
        params=[self.C, self.a, self.nugget, self.anisotropy_factor]
        self.divscores=[]
        if manualbounds==None:
            bounds=[np.min(self.points[:,0]),np.max(self.points[:,0]),np.min(self.points[:,1]),np.max(self.points[:,1])]
        else:
            bounds=manualbounds
        #LOO cross validation loop
        for i in range(len(self.points)):
            print('Iterations: ',i+1,'/',len(self.points)+1)
            new_points, new_zvals = self.remove_neighbors(i, self.radius)
            model = UniversalKriging(new_points, new_zvals,Variogram=self.variogram)
            estimate = model.AutoKrige(step=step,bounds=bounds)
            self.divscores.append((np.mean(self.DivModel(np.abs(self.zarray/np.sum(self.zarray)),np.abs(estimate/np.sum(estimate))))))
        np.save('div_scores.npy', self.divscores)
        return self.divscores
    

    def plot_div(self,resolution=1000,saveplot=False,powerscale=1):
            plt.style.use('ggplot')
            #check if divscores file exists if so load it if not save it
            try:
                self.divscores=np.load('div_scores.npy')
            except:
                ValueError('div_scores.npy does not exist')

            #interpolate the divergance scores in 3d space with x and y as the input and divscores as the output
            x=np.linspace(np.min(self.points[:,0]),np.max(self.points[:,0]),resolution)
            y=np.linspace(np.min(self.points[:,1]),np.max(self.points[:,1]),resolution)
            X,Y=np.meshgrid(x,y)
            Z=griddata(self.points,self.divscores,(X,Y),method='linear')
            
            #plot the interpolated divergance scores
            im = plt.imshow(Z**powerscale, cmap='YlOrRd', interpolation='bilinear', origin='lower',aspect='auto', extent=[np.min(self.points[:,0]),np.max(self.points[:,0]),np.min(self.points[:,1]),np.max(self.points[:,1])])
            plt.scatter(self.points[:,0],self.points[:,1],c='k',s=15)
            #plot configuration
            plt.title('LOO Divergance Values')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.colorbar(im)
            plt.grid(color='black', linestyle='-', linewidth=0.5,alpha=.25)
            #set x and y lims to 0  
            plt.xlim(0)
            plt.ylim(0)

            plt.show()
            plt.close()
            if saveplot==True:
                plt.savefig('div_scores.png')
            plt.show()
            plt.close()
            return Z
         


