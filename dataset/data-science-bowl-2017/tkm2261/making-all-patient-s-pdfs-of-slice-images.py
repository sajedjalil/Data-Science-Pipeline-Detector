import os
import glob
import numpy as np
import dicom
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from logging import StreamHandler, DEBUG, Formatter, getLogger

logger = getLogger(__name__)

DATA_PATH = '../input/'
STAGE1_FOLDER = DATA_PATH + 'sample_images/'
PDF_FOLDER = '../stage1/'

def get_3d_data(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    img = np.stack([s.pixel_array for s in slices])

    return img


def make_pdf():

    plt.switch_backend('agg')
    logger.info("Start")
    logger.info("Due to time limit, making only 5 pdfs")
    for folder in glob.glob(STAGE1_FOLDER + '*')[:5]:
        patient_id = os.path.basename(folder)
        logger.info("Saving pdf in %s" % (patient_id))
        img = get_3d_data(folder)
        with PdfPages('%s.pdf' % patient_id) as pdf:
            for i in range(img.shape[0]):
                plt.figure()
                plt.imshow(img[i], cmap=plt.cm.gray)
                pdf.savefig()
                plt.close()


if __name__ == '__main__':

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    make_pdf()
