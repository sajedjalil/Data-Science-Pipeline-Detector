# We've included a sample of 1000 images and trainLabels.csv in the scripts environment for this competition
# See https://www.kaggle.com/benhamner/diabetic-retinopathy-detection/sample-images for a couple sample images

from subprocess import check_output

# Print a list of the input files to the log
print(check_output(["ls", "../input"]).decode("utf8"))
