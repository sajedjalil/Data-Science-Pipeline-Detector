import gdal
import numpy as np

im = gdal.Open('../input/sixteen_band/6070_2_3_M.tif')
img = im.ReadAsArray()
(bands,x,y) = np.shape(img)
driver = gdal.GetDriverByName('GTiff')

dst_datatype = gdal.GDT_UInt16
dst_ds = driver.Create("test.tif",y,x,3,dst_datatype)

#True Color (Red, Green, Blue)
# dst_ds.GetRasterBand(1).WriteArray(img[4,:,:])
# dst_ds.GetRasterBand(2).WriteArray(img[2,:,:])
# dst_ds.GetRasterBand(3).WriteArray(img[1,:,:])

#See Vegetation (NIR1,Green,Blue)
dst_ds.GetRasterBand(1).WriteArray(img[3,:,:])
dst_ds.GetRasterBand(2).WriteArray(img[5,:,:])
dst_ds.GetRasterBand(3).WriteArray(img[0,:,:])

#See output by running this from terminal
#gdal_translate -of JPEG -scale -co worldfile=yes test.tif test1.jpg