
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from zipfile import ZipFile
with ZipFile('submission.zip','w') as zip:           
  zip.write('../input/resnet101v2-vgg16-model/ResNet101v2.h5', arcname='Saved_Model.h5') 