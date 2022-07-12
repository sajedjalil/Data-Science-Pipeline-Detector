# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import sys, os, glob
import shutil
import time, _thread
import multiprocessing as mp
import gc

import argparse

import numpy as np
import pandas as pd
import matplotlib as plt


from IPython.display import HTML
from base64 import b64encode


os.system("mkdir -p /kaggle/working/scripts")
os.system("mkdir -p /kaggle/working/facedetect/single")
os.system("mkdir -p /kaggle/working/facedetect/diff")
os.system("mkdir -p /kaggle/working/facedetect/videodiff")

os.system("mkdir -p /kaggle/working/test_videos/single")
os.system("mkdir -p /kaggle/working/test_videos/videodiff")

os.environ["GST_PLUGIN_PATH_1_0"]='/opt/conda/lib/gstreamer-1.0:/usr/lib/x86_64-linux-gnu/gstreamer-1.0'
os.environ["GST_PLUGIN_PATH"]='/opt/conda/lib/gstreamer-1.0:/usr/lib/x86_64-linux-gnu/gstreamer-1.0'
os.environ["PATH"]='/opt/conda/bin:/usr/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/sbin:/bin'
os.environ["LD_LIBRARY_PATH"]='/opt/conda/lib:/usr/local/lib:/opt/conda/lib/gstreamer-1.0:/usr/lib:/usr/lib/x86_64-linux-gnu'
os.environ['FREI0R_PATH']='/usr/lib/frei0r-1'
os.environ["GI_TYPELIB_PATH"]='/opt/conda/lib/girepository-1.0:/usr/lib/girepository-1.0'
os.system("set PYTHONPATH=/kaggle/lib/kaggle:/kaggle/lib:/usr/lib/python3:/kaggle/input/deepfake-detection-challenge")
os.system("set PATH=/opt/conda/bin:/usr/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/sbin:/bin")
os.system("set LD_LIBRARY_PATH=/opt/conda/lib:/usr/local/lib:/opt/conda/lib/gstreamer-1.0:/usr/lib:/usr/lib/x86_64-linux-gnu")
os.system("set GST_PLUGIN_PATH_1_0=/opt/conda/lib/gstreamer-1.0:/usr/lib/x86_64-linux-gnu/gstreamer-1.0")
os.system("set GST_PLUGIN_PATH=/opt/conda/lib/gstreamer-1.0:/usr/lib/x86_64-linux-gnu/gstreamer-1.0")
os.system("set FREI0R_PATH=/usr/lib/frei0r-1")
try:
    def install_packages():
        shutil.copyfile("/kaggle/input/video-tools/scripts/install-additional-packages.sh", "/kaggle/working/scripts/install-additional-packages.sh")
        os.system("chmod ugo+rx /kaggle/working/scripts/install-additional-packages.sh")
        os.system("sh -c /kaggle/working/scripts/install-additional-packages.sh")
        os.system("set GI_TYPELIB_PATH=/opt/conda/lib/girepository-1.0:/usr/lib/girepository-1.0")
        os.system("rm -R ~/.cache/gstreamer-1.0;gst-inspect-1.0")

    class StaticSupportMethods:
        @staticmethod
        def playvid(filepath, width=640):
            vid = open(filepath,'rb').read()
            filename, ext = os.path.splitext(filepath)
            ext = ext.replace(".", "")
            data_url = "data:video/{0};base64,".format(ext) +b64encode(vid).decode()
            return HTML("""
                        <video width={2} controls>
                            <source src="{0}" type="video/{1}"
                        </video>
                        """.format(data_url, ext, width))

        @staticmethod
        def get_meta_from_json(filepath, start=None, stop = None):
            df = pd.read_json(filepath)
            df = df.T
            df['label'] = df['label'].map({'FAKE': 1, 'REAL': 0})
            if start is not None:
                if stop is not None:
                    return df[start:stop]
                else:
                    return df[start:]
            elif stop is not None:
                return df[:stop]
            else:
                return df

        @staticmethod
        def fakes_with_original_mapping(filepath, start=None, stop = None):
            df = StaticSupportMethods.get_meta_from_json(filepath, start, stop)
            fakes = df[df.original.notnull()]
            originals = df[df.original.isnull()]
            return fakes[fakes.original.isin(originals.index)]

        @staticmethod
        def originals_in_dataset(filepath, start=None, stop = None):
            df = StaticSupportMethods.get_meta_from_json(filepath, start, stop)
            return df[df.original.isnull()]

        @staticmethod
        def mapped_in_dataset(filepath, start=None, stop = None):
            df = StaticSupportMethods.get_meta_from_json(filepath, start, stop)
            originals = df[df.original.isnull()]
            originals.original = originals.index
            fakes_ = df[df.original.notnull()]
            fakes = fakes_[fakes_.original.isin(originals.index)]
            return pd.concat([fakes,originals])

    class SingleVideoHeadExtract(object):
        def __init__(self, filepath, outdir, width=1920, height=1080, face_width=256, face_height=256, **kwargs):

            self.completed = False          

            dirname,filename = os.path.split(filepath)
            basename, ext = os.path.splitext(filename)
            output_file = basename+".avi"
            output_filepath = os.path.join(outdir,"single",output_file)
            PIPELINE_DEF = "filesrc name=vidsrc location={0} ! decodebin ! videoconvert ! deinterlace ! videoconvert qos=true ! " \
                        "video/x-raw,width={1},height={2},format=RGB,framerate=30000/1001 ! tee name=t allow-not-linked=true pull-mode=single has-chain=true " \
                        "t. ! facedetect display=false updates=every_frame profile=/opt/conda/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml " \
                        "t. ! videocrop name=crop ! videoscale add-borders=false ! video/x-raw,width={3},height={4},framerate=30000/1001 ! " \
                        "videoconvert ! video/x-raw,format=(string)I420,framerate=30000/1001 ! " \
                        "x264enc ! video/x-h264,profile=(string)baseline,framerate=30000/1001 ! " \
                        "avimux ! filesink location={5}".format(filepath,width,height,face_width,face_height,output_filepath)

            print("gst-launch-1.0 -e -m {0}".format(PIPELINE_DEF))


            self.pipeline = Gst.parse_launch(PIPELINE_DEF)

            self.file = filepath
            self.width = width
            self.height = height
            self.coords = {'x':0,'y':0,'w':0,'h':0}
            self.crop = self.pipeline.get_by_name('crop')
            # Set up a pipeline bus watch to catch errors.
            self.bus = self.pipeline.get_bus()
            self.bus.add_signal_watch()
            self.bus.connect('message', self.on_message)
            self.bus.connect('message::eos', self.on_eos)
            self.bus.connect('message::error', self.on_error)
            GObject.threads_init()
            self.mainloop = GLib.MainLoop()
            try:
                # self.playmode = True
                self.pipeline.set_state(Gst.State.READY)
                self.pipeline.set_state(Gst.State.PLAYING)
                self.mainloop.run()
                # while self.playmode:
                #    time.sleep(.1)
                # time.sleep(.1)
            except Exception as e:
                print("Error: ", e)
            finally:
                self.bus.remove_signal_watch()
                # del self.pipeline
                self.mainloop.quit()


        def on_eos(self,bus,message):
            self.pipeline.set_state(Gst.State.NULL)
            # self.playmode = False
            # self.completed = True
            self.bus.remove_signal_watch()
            # del self.pipeline
            self.mainloop.quit()


        def on_error(self,bus,message):
            self.pipeline.set_state(Gst.State.NULL)
            # self.playmode = False
            err, debug = message.parse_error()
            print("Error: {} {}".format(err, debug))
            self.bus.remove_signal_watch()
            # del self.pipeline
            self.mainloop.quit()


        def on_message(self,bus,message):
            t = message.type
            st = message.get_structure()
            face = None
            if st is not None:
                faces = st.get_value("faces")
                if faces is not None and len(faces)>0:
                    face = faces[0]
            if face is not None:
                x,y,w,h = int(face["x"]),int(face["y"]),int(face["width"]),int(face["height"])
                if self.coords['x'] != x or self.coords['y'] != y or self.coords['w'] != w or self.coords['h'] != h:
                    self.coords = {'x':x,'y':y,'w':w,'h':h}
                    left,top,right,bottom = (x,y,self.width - (x + w),self.height - (y + h))
                    # print("videocrop fqce: l={0},t={1},r={2},b={3}".format(left,top,right,bottom))
                    self.crop.set_property('left', left)
                    self.crop.set_property('top', top)
                    self.crop.set_property('right', right)
                    self.crop.set_property('bottom',bottom)


    class ProcessMethods:
        @staticmethod
        def ProcessSingleVideo(filepath=None, output_dir=None ):
            proc = SingleVideoHeadExtract(filepath, output_dir) #TMP_FACE_DETECT_FOLDER
            del proc

        @staticmethod
        def VideoDiff(filepath= None, output_directory=None):
            dirname,filename = os.path.split(filepath)
            output_path = os.path.join(output_directory,filename)
            cmdargs = "-e filesrc location={0} ! 'video/x-h264,framerate=30000/1001' ! decodebin ! videoconvert ! 'video/x-raw,format=I420,famerate=30000/1001'! videodiff ! videoconvert ! 'video/x-raw,format=(string)I420,framerate=30000/1001' ! x264enc ! 'video/x-h264,profile=(string)baseline,framerate=30000/1001' ! avimux ! filesink location={1}".format(filepath,output_path)
            ret = os.system("gst-launch-1.0 {0}".format(cmdargs))

        @staticmethod
        def DifferenceVideos(input_0 = None, input_1=None, data_dir=None, output_directory=None):
            basename_0, _ = os.path.splitext(input_0)
            basename_1, _ = os.path.splitext(input_1)
            filepath_0 = os.path.join(data_dir,basename_0+".avi")
            filepath_1 = os.path.join(data_dir,basename_1+".avi")
            output_filename = basename_0+"_"+basename_1+".avi"
            output_filepath = os.path.join(output_directory,output_filename)
            cmdargs = "-i {0} -i {1} -filter_complex \"blend=all_mode='difference'\" {2}".format(filepath_0,filepath_1,output_filepath)
            ret = os.system("ffmpeg {0}".format(cmdargs))


    def process_videos():            
        OPENCV_DATA_FOLDER='/opt/conda/share/OpenCV'
        FACE_DETECTION_FOLDER = os.path.join(OPENCV_DATA_FOLDER,'haarcascades')
        FRONTAL_DETECTION_FOLDER = os.path.join(OPENCV_DATA_FOLDER, 'lbpcascades')

        DATA_FOLDER = '/kaggle/input/deepfake-detection-challenge'
        TRAIN_SAMPLE_FOLDER = 'train_sample_videos'
        TEST_FOLDER = 'test_videos'
        WORKING_FOLDER = '/kaggle/working'
        TRAIN_SAMPLE_DIRECTORY = os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER)
        TEST_VIDEOS_DIRECTORY = os.path.join(DATA_FOLDER,TEST_FOLDER)
        TMP_FACE_DETECT_FOLDER = os.path.join(WORKING_FOLDER,'facedetect')
        TMP_FACE_VIDEODIFF_FOLDER = os.path.join(TMP_FACE_DETECT_FOLDER,"videodiff")
        TMP_FACE_DIFF_FOLDER = os.path.join(TMP_FACE_DETECT_FOLDER,"diff")
        TMP_TEST_VIDEOS_FOLDER = os.path.join(WORKING_FOLDER, TEST_FOLDER)
        TMP_TEST_VIDEODIFF_FOLDER = os.path.join(TMP_TEST_VIDEOS_FOLDER, "videodiff")

        df = StaticSupportMethods.get_meta_from_json(os.path.join(TRAIN_SAMPLE_DIRECTORY, "metadata.json"))

        Gst.init(None)

        print("Populating {0}".format(TMP_FACE_DETECT_FOLDER))
        videos = glob.glob(os.path.join(TRAIN_SAMPLE_DIRECTORY,'*.mp4'), recursive=False)
        pool = mp.Pool(processes=mp.cpu_count())
        # _ = [pool.apply(ProcessMethods.ProcessSingleVideo, args=(os.path.join(TRAIN_SAMPLE_DIRECTORY,index),TMP_FACE_DETECT_FOLDER)) for index, _ in df.iterrows()]
        _ = [pool.apply(ProcessMethods.ProcessSingleVideo, args=(f,TMP_FACE_DETECT_FOLDER)) for f in videos]
        del pool
        del videos
        gc.collect()

        print("Populating {0}".format(TMP_FACE_VIDEODIFF_FOLDER))
        FACES_DIR = os.path.join(TMP_FACE_DETECT_FOLDER,"single")
        face_videos = glob.glob(os.path.join(FACES_DIR,'*.avi'), recursive=False)
        pool = mp.Pool(processes=mp.cpu_count())
        _ = [pool.apply(ProcessMethods.VideoDiff, args=(f,TMP_FACE_VIDEODIFF_FOLDER)) for f in face_videos]
        del pool
        del face_videos
        gc.collect()

        meta_f2o_df = StaticSupportMethods.fakes_with_original_mapping(os.path.join(TRAIN_SAMPLE_DIRECTORY, "metadata.json"))

        meta_orig_df = StaticSupportMethods.originals_in_dataset(os.path.join(TRAIN_SAMPLE_DIRECTORY, "metadata.json"))

        print("Populating <fake> {0}".format(TMP_FACE_DIFF_FOLDER))
        pool = mp.Pool(processes=mp.cpu_count())
        _ = [pool.apply(ProcessMethods.DifferenceVideos, args=(index,row['original'],FACES_DIR, TMP_FACE_DIFF_FOLDER)) for index, row in meta_f2o_df.iterrows()]
        del pool
        gc.collect()

        print("Populating <real> {0}".format(TMP_FACE_DIFF_FOLDER))
        pool = mp.Pool(processes=mp.cpu_count())
        _ = [pool.apply(ProcessMethods.DifferenceVideos, args=(index,index,FACES_DIR, TMP_FACE_DIFF_FOLDER)) for index, _ in meta_orig_df.iterrows()]
        del pool
        gc.collect()

        print("Populating {0}".format(TMP_TEST_VIDEOS_FOLDER))
        test_videos = glob.glob(os.path.join(TEST_VIDEOS_DIRECTORY,'*.mp4'), recursive=False)
        pool = mp.Pool(processes=mp.cpu_count())
        _ = [pool.apply(ProcessMethods.ProcessSingleVideo, args=(f,TMP_TEST_VIDEOS_FOLDER)) for f in test_videos]
        del pool
        del test_videos
        gc.collect()

        print("Populating {0}".format(TMP_TEST_VIDEODIFF_FOLDER))
        TEST_FACES_DIR = os.path.join(TMP_TEST_VIDEOS_FOLDER,"single")
        test_face_videos = glob.glob(os.path.join(TEST_FACES_DIR,'*.avi'), recursive=False)
        pool = mp.Pool(processes=mp.cpu_count())
        _ = [pool.apply(ProcessMethods.VideoDiff, args=(f,TMP_TEST_VIDEODIFF_FOLDER)) for f in test_face_videos]
        del pool
        del test_face_videos
        gc.collect()


except:
        print("Error importing tensorflow-datasets")
        

install_packages()

import gi
gi.require_version("GIRepository", "2.0")
from gi.repository import GIRepository
gi.require_version('GLib', '2.0')
gi.require_version('GObject', '2.0')
gi.require_version('Gst','1.0')
gi.require_version('GstApp','1.0')
gi.require_version('GstAudio','1.0')
gi.require_version('GstVideo','1.0')
from gi.repository import GObject, GLib, Gst, GstApp, GstVideo, GstAudio

        
def main():
    # print command line arguments
    parser = argparse.ArgumentParser(description='Installl additonal packages.')
    parser.add_argument('--process', default=False, help='Generate Procesed file')
    args = parser.parse_args()
        
    
    if args.process:
        print("Doing process videos ...")
        process_videos()
    
    

if __name__ == "__main__":
    # execute only if run as a script
    main()