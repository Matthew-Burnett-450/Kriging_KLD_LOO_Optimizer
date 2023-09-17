import numpy as np
from noise import snoise2
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from Ok_Uk_Module import *
import concurrent.futures
from KLDclasses import *

error=0
# Set the scale of the terrain features
scale = 0.025

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
    epsilon = np.random.uniform(-error, error, size=original_array.shape)
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

def run_test(disp=False):
    terrain = generate_terrain(width=10, height=10, scale=np.random.uniform(low=.01,high=.02), octaves=np.random.randint(low=15,high=25), persistence=.1, sigma=2,z=10)
    points,zpoints=sample_points(terrain=terrain,num_points=10)    

    kld=SpacialSensitivityAnalysisUK(points,zpoints,Variogram=ExponentialVariogram,trendfunc='cubic',radius=1)
    zmap=kld.AutoKrige(step=1,bounds=[0,10,0,10])
    kld.DiverganceLOO(step=1,manualbounds=[0,10,0,10])
    mse=np.mean((terrain-zmap.T)**2)
    print(mse)

    if disp == True:
        # Create a figure with two subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 6))


        vmin = np.min([zmap, terrain])
        vmax = np.max([zmap, terrain])
        
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
        im = ax3.imshow(Z**.25, cmap='YlOrRd', interpolation='bilinear', origin='lower', extent=[np.min(kld.points[:,0]),np.max(kld.points[:,0]),np.min(kld.points[:,1]),np.max(kld.points[:,1])])
        ax3.scatter(kld.points[:,0],kld.points[:,1],c='k',s=15)
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
        


        plt.show()
    

#Terrain test

n=1

t0 = time.time()

for i in range(n):
    t_0 = time.time()
    run_test(disp=True)
    t_1 = time.time()
    print(t_1-t_0)

t1 = time.time()
totalexetime = t1-t0

print(totalexetime)
print(f'avg time _ {totalexetime/n}')