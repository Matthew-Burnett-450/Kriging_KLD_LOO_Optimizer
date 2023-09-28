import numpy as np
from noise import snoise2
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from Ok_Uk_Module import *
import concurrent.futures
from KLDclasses import *
from Variograms_Trendfuncs import *
error=0
# Set the scale of the terrain features
scale = 0.025
np.random.seed(69) # Seed for reproducibility

#Simulated terrain test
def generate_terrain(width, height, scale, octaves, persistence, sigma,z):
    # Initialize the terrain
    terrain = np.zeros((width, height))

    # Generate the fractal noise
    for i in range(width):
        for j in range(height):
            frequency = scale
            amplitude = 1.0
            for _ in range(octaves):
                terrain[i][j] += snoise2(i * frequency, j * frequency) * amplitude
                frequency *= 2  # Double the frequency at each octave
                amplitude *= persistence  # Reduce the amplitude by the persistence at each octave

    # Apply Gaussian blur
    terrain = gaussian_filter(terrain, sigma=sigma)

    return np.abs(terrain)*z

def simulate_array_with_error(original_array, error):
    epsilon = 0
    simulated_array = original_array + epsilon
    return simulated_array

def sample_points(terrain, num_points):
    # Get the width and height of the terrain
    width, height = terrain.shape

    # Generate random x and y coordinates
    x_coords = np.random.randint(0, width, num_points)
    y_coords = np.random.randint(0, height, num_points)

    # Sample the z values from the terrain
    z_values = terrain[x_coords, y_coords]
    z_values = simulate_array_with_error(z_values, error)

    # Add the four corners
    corners_x = np.array([0, 0, width-1, width-1])
    corners_y = np.array([0, height-1, 0, height-1])
    corners_z = terrain[corners_x, corners_y]

    # Concatenate the random points and the corners
    final_x_coords = np.concatenate([x_coords, corners_x])
    final_y_coords = np.concatenate([y_coords, corners_y])
    final_z_values = np.concatenate([z_values, corners_z])

    # Return the coordinates and z values as numpy arrays
    return np.array([final_x_coords, final_y_coords]).T, final_z_values


def find_filtered_centroid(array_2d, max_x, max_y):
    # Get array shape
    width, height = array_2d.shape
    
    # Compute dx and dy
    dx = max_x / width
    dy = max_y / height
    
    # Filter out values below the 90th percentile
    threshold = np.percentile(array_2d, 80)
    array_2d[array_2d < threshold] = 0
    
    # Find unweighted centroid in array coordinates
    indices = np.argwhere(array_2d > 0)
    if indices.shape[0] == 0:
        return None  # Handle the case where all values are zero
    centroid_array_coords = indices.mean(axis=0)
    
    # Convert to real-world coordinates
    centroid_real_coords = centroid_array_coords * [dx, dy]
    
    return centroid_real_coords




def run_test(disp=False,num_points=1,width=30,height=30):
    start=time.time()
    terrain = generate_terrain(width=width, height=height, scale=np.random.uniform(low=.01,high=.02), octaves=np.random.randint(low=3,high=8), persistence=.1, sigma=1,z=10)
    points,zpoints=sample_points(terrain=terrain,num_points=num_points)    

    kld=SpacialSensitivityAnalysisOK(points,zpoints,Variogram=ExponentialVariogram(),radius=1)
    zmap=kld.AutoKrige(step=1,bounds=[width,0,height,0])
    kld.DiverganceLOO(step=1,manualbounds=[width,0,height,0])
    MAE=np.mean(np.abs(terrain-zmap.T))
    print(MAE)
    end=time.time()
    print(f"total: {end-start}")
    if disp == True:
        # Create a figure with two subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 6))


        vmin = 0
        vmax =10
        
        # Display the first terrain in the first subplot
        im1=ax1.imshow(zmap, cmap='terrain', vmin=vmin, vmax=vmax,origin='lower')
        ax1.set_title('Krige Guess')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        # Display the second terrain in the second subplot
        im2 = ax2.imshow(terrain.T, cmap='terrain',vmin=vmin, vmax=vmax,origin='lower')
        ax2.scatter(points[:,0],points[:,1],facecolors='none', edgecolors='r',s=1)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('True Terrain')


        # Show the figure
        x=np.linspace(np.min(kld.points[:,0]),np.max(kld.points[:,0]),200)
        y=np.linspace(np.min(kld.points[:,1]),np.max(kld.points[:,1]),200)
        X,Y=np.meshgrid(x,y)
        Z=griddata(kld.points,kld.divscores,(X,Y),method='linear')
        

        #plot the interpolated divergance scores
        im = ax3.imshow(Z, cmap='YlOrRd', interpolation='bilinear', origin='lower', extent=[np.min(kld.points[:,0]),np.max(kld.points[:,0]),np.min(kld.points[:,1]),np.max(kld.points[:,1])])
        ax3.scatter(kld.points[:,0],kld.points[:,1],c='k',s=15)
        ax3.scatter(find_filtered_centroid(Z,width,height)[0],find_filtered_centroid(Z,width,height)[1],c='green',s=5)
        #plot configuration
        ax3.set_title('LOO Divergance Values')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        fig.colorbar(im, ax=ax3)

        #ax4 make true diffrence between terain and kriged guess
        vmin = np.min([np.abs(terrain.T-zmap)])
        vmax = np.max([np.abs(terrain.T-zmap)])
        im4=ax4.imshow(np.abs(terrain.T-zmap)**2, cmap='Blues',vmin=vmin, vmax=vmax,origin='lower')
        ax4.set_title('True Error ^2')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        fig.colorbar(im4, ax=ax4)
        plt.xlabel(f'True MAE = {MAE}')
        plt.suptitle(f'Random n={len(zpoints)}')
        plt.savefig(f'./kldsimfigs/initial.png')
        plt.show()
    return terrain,points,zpoints,find_filtered_centroid(Z,width,height)

    

#Terrain test

"""n=2


t0 = time.time()

for i in range(n):
    t_0 = time.time()
    run_test(disp=True,num_points=1)
    t_1 = time.time()
    print(t_1-t_0)

t1 = time.time()
totalexetime = t1-t0

print(totalexetime)
print(f'avg time _ {totalexetime/n}')"""
#Iterative terrain test



def run_next(inpoints,inzpoints,terrain,nextpoint,i,disp=False,width=30,height=30):
    points=np.vstack([inpoints,(np.round(nextpoint[0]).astype(int),np.round(nextpoint[1]).astype(int))])
    zpoints=np.append(inzpoints,(terrain[np.round(nextpoint[0]-.5).astype(int),np.round(nextpoint[1]-.5).astype(int)]))
    kld=SpacialSensitivityAnalysisOK(points,zpoints,Variogram=ExponentialVariogram(),radius=1)
    zmap=kld.AutoKrige(step=1,bounds=[width,0,height,0])
    kld.DiverganceLOO(step=1,manualbounds=[width,0,height,0])
    MAE=np.mean(np.abs(terrain-zmap.T))
    print(MAE)

    x=np.linspace(np.min(kld.points[:,0]),np.max(kld.points[:,0]),200)
    y=np.linspace(np.min(kld.points[:,1]),np.max(kld.points[:,1]),200)
    X,Y=np.meshgrid(x,y)
    Z=griddata(kld.points,kld.divscores,(X,Y),method='linear')
    
    if disp == True:
        # Create a figure with two subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 6))


        vmin = 0
        vmax = 10
        
        # Display the first terrain in the first subplot
        im1=ax1.imshow(zmap, cmap='terrain', vmin=vmin, vmax=vmax,origin='lower')
        ax1.set_title('Krige Guess')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        # Display the second terrain in the second subplot
        im2 = ax2.imshow(terrain.T, cmap='terrain',vmin=vmin, vmax=vmax,origin='lower')
        ax2.scatter(points[:,0],points[:,1],facecolors='none', edgecolors='r',s=1)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('True Terrain')


        # Show the figure
        x=np.linspace(np.min(kld.points[:,0]),np.max(kld.points[:,0]),200)
        y=np.linspace(np.min(kld.points[:,1]),np.max(kld.points[:,1]),200)
        X,Y=np.meshgrid(x,y)
        Z=griddata(kld.points,kld.divscores,(X,Y),method='linear')
        
        #plot the interpolated divergance scores
        im3 = ax3.imshow(Z, cmap='YlOrRd', interpolation='bilinear', origin='lower', extent=[np.min(kld.points[:,0]),np.max(kld.points[:,0]),np.min(kld.points[:,1]),np.max(kld.points[:,1])])
        ax3.scatter(kld.points[:,0],kld.points[:,1],c='k',s=15)
        ax3.scatter(find_filtered_centroid(Z,width,height)[0],find_filtered_centroid(Z,width,height)[1],c='green',s=5)
        #plot configuration
        ax3.set_title('LOO Divergance Values')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')


        #ax4 make true diffrence between terain and kriged guess
        vmin = np.min([np.abs(terrain.T-zmap)])
        vmax = np.max([np.abs(terrain.T-zmap)])
        im4=ax4.imshow(np.abs(terrain.T-zmap), cmap='Blues',vmin=vmin, vmax=vmax,origin='lower')
        ax4.set_title('True Error ^2')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        fig.colorbar(im1, ax=ax1)
        fig.colorbar(im2, ax=ax2)
        fig.colorbar(im3, ax=ax3)
        fig.colorbar(im4, ax=ax4)
        plt.suptitle(f'Iteration {i+1}, n={len(zpoints)}')
        plt.xlabel(f'True MAE = {MAE}')
        plt.savefig(f'./kldsimfigs/Iteration {i+1}.png')
        #plt.show()
        plt.close()

    return terrain,points,zpoints,np.array(find_filtered_centroid(Z,width,height))
    
terrain,outpoints,zpoints,nextpoints=run_test(disp=True,num_points=1,width=100,height=100)
nsteps=5+5
for i in range(nsteps):
    
    if i ==nsteps-1:
        terrain,outpoints,zpoints,nextpoints=run_next(outpoints,zpoints,terrain,nextpoints,i,disp=True,width=100,height=100)
    else:
        terrain,outpoints,zpoints,nextpoints=run_next(outpoints,zpoints,terrain,nextpoints,i,disp=True,width=100,height=100)

terrain,outpoints,zpoints,nextpoints=run_test(disp=True,num_points=6+5,width=100,height=100)