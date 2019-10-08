import matplotlib.pyplot as plt

def visualize_activity_centers(raster, points):
    points_x = [p[0] for p in points]
    points_y = [p[1] for p in points]
    
    # imshow coordinate (0, 0) is at the center of raster grid (0, 0)
    points_x_plot = [p - 0.5 for p in points_x]
    points_y_plot = [p - 0.5 for p in points_y]
    plt.imshow(raster)
    plt.scatter(points_x_plot, points_y_plot, c='k', s=3)
    
    # cut off image at raster boundaries (in imshow coordinates)
    # imshow origin is bottom left; raster axis is top left
    plt.gca().axis([-0.5, raster.shape[0]-0.5, raster.shape[1]-0.5, -0.5])
    plt.show()

def visualize_trap_layout(raster, grid_loc):
    grid_x_plot = [p[0] - 0.5 for p in grid_loc]
    grid_y_plot = [p[1] - 0.5 for p in grid_loc]

    plt.imshow(raster)
    plt.scatter(grid_x_plot, grid_y_plot, c='k', marker='+')
    plt.show()