# Import libraries
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt 
import numpy as np 


# Creating radii and angles
r = np.linspace(0.125, 1.0, 100) 
a = np.linspace(0, 2 * np.pi, 
				100,
				endpoint = False) 

# Repeating all angles for every radius 
a = np.repeat(a[..., np.newaxis], 100, axis = 1) 

# Creating dataset
x = np.append(0, (r * np.cos(a))) 
y = np.append(0, (r * np.sin(a))) 
z = (np.sin(x ** 4) + np.cos(y ** 4)) 

# Creating figure
fig = plt.figure(figsize =(16, 9)) 
ax = plt.axes(projection ='3d') 

# Creating color map
my_cmap = plt.get_cmap('hot')

# Creating plot
trisurf = ax.plot_trisurf(x, y, z,
						cmap = my_cmap,
						linewidth = 0.2, 
						antialiased = True,
						edgecolor = 'grey') 
fig.colorbar(trisurf, ax = ax, shrink = 0.5, aspect = 5)
ax.set_title('Tri-Surface plot')

# Adding labels
ax.set_xlabel('X-axis', fontweight ='bold') 
ax.set_ylabel('Y-axis', fontweight ='bold') 
ax.set_zlabel('Z-axis', fontweight ='bold')
	
# show plot
print(x.shape, y.shape, z.shape)
