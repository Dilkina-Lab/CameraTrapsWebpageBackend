# Camera Traps Tool

## Current Functionality
<ul>
<li>Add and remove markers (new markers you add are a different color)</li> 
<li>Load CSV file of camera trap locations with detections data → different color markers based on number of detections</li>
<li>Markers when you click say how many detections there are at the location and show the location ID</li>
<li>Save markers to CSV (click save locations and it will create a file called markers.csv with the locations)</li>
<li>View activity centers of animals </li>
<li>Load a tiff file and display a density map based on activity center locations</li>
</ul>

## CSV File Formatting
The CSV file should be formatted with an id, longitude, latitude, total detections, and unique detections. This CSV file is created based on trap locations. The code is in the amritagupta/scrpy/src/Generate Landscape Rasters notebook. 

## Tiff File Formatting
The tiff file is created in the amritagupta/scrpy/src/Generate Landscape Rasters notebook. We create a grid array where each grid cell has a number which is the maximum count of activity centers in that cell. Based on this array, we create a tiff file to represent the activity centers.  

## Steps to run web app
cd into the bottle-webapps folder and type python runapp.py

## Future Steps
<ul>
<li>Tell user the accuracy of the placements</li>
<li>Add the density map overlay in a better way</li>
<li>Zoom to extent of markers</li>
</ul>


## Density Map Overlay 
I tried to do the density map in a few ways. 
<ul>
<li>Currently, you upload the tiff file to the web page and use a function called parseGeoraster which creates a GeoRasterLayer of the raster that you add to the map as a layer.</li> 
<li>I also tried to use the function ScalarField.fromGeoTIFF but it did not load anything.</li>
<li>I also tried to save the tiff file as a png and then overlay the image on the map, but when adding the image as an overlay I was not able to remove the background color so the map was not visible under the image.</li>
</ul>

## Zoom to the markers
I tried to zoom to the extent of the markers in a few ways. 
<ul>
<li>I tried to create a feature group of all the markers and then fit the map to the bounds of the feature group.</li> 
<li>I also tried to use the function latLngBounds to get the bounds of all the markers and fit to that.</li>
<li>In the function fitMarkers(), I tried to get the lat long for each marker and then use the functoin latLngBounds and fit the bounds to that.</li>
Note: When you upload the GeoTiff file and add that as a layer, it does zoom the bounds of the layer with this line of code: mymap.fitBounds(layer.getBounds()). 
</ul>