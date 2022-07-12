#
# Added clean and OCR features over the images.
#
import numpy as np
from PIL import Image
import pytesseract

# sudo apt-get install tesseract-ocr
# pip install pytesseract

threshold=100

def discretize(a, threshold):
    return np.uint8((a > threshold)*255)

image_id = 101
dirty_image_path = "../train/%d.png" % image_id
clean_image_path = "../train_cleaned/%d.png" % image_id

dirty = Image.open(dirty_image_path)
clean = Image.open(clean_image_path)

dirty.save("dirty.png")
clean.save("clean.png")

clean_array = np.asarray(clean)
dirty_array = np.asarray(dirty)
    
discretized_array = discretize(dirty_array, threshold)
Image.fromarray(discretized_array).save("discretized.png")

text_image = pytesseract.image_to_string(Image.open("discretized.png"))

html = """<html>
	<body>
	<h1>Thresholding</h1>
	<p>This is a very simple attempt to clean up an image by thresholding the pixel value at 50. (Under 50 goes to 0, above 50 goes to 255.)</p>
	<h2>Dirty image</h2>
	<img src="dirty.png">
        <p>%s</p>
	<h2>Cleaned up by thresholding</h2>
	<img src="discretized.png">
	<h2>Original clean image</h2>
	<img src="clean.png">
	</body>
	</html>
""" % text_image

with open("/var/www/output.html", "w") as output_file:
	output_file.write(html)

