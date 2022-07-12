# version 21
# funcionalidad para capturar, procesar frames de videos
# calcular probabilidad de manipulacion y generar submmit
# bibliotecas
import os
import sys
import glob
# import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import model_from_yaml
import tensorflow.keras
from tensorflow.keras import models
import cv2
import time
# import itertools
sys.path.insert(0, "/kaggle/input/mtcmtflib/tf-mtcnn-master")
from mtcnn import MTCNN

# definiciones log
# logging.basicConfig(filename='logs/final_preproceso_submmit_.log', level=logging.ERROR,
#                     format='%(asctime)s %(levelname)s %(name)s %(message)s')
# logger = logging.getLogger(__name__)

# definiciones de funciones globales


# cambia tamaño
def escala(im_local, no_filas, no_columnas):
    """

    :param im_local:
    :param no_filas:
    :param no_columnas:
    :return:
    """
    no_filas_original = len(im_local)
    no_columnas_original = len(im_local[0])
    return [[im_local[int(no_filas_original * fila_local / no_filas)]
             [int(no_columnas_original * columna_local / no_columnas)]
             for columna_local in range(no_columnas)] for fila_local in range(no_filas)]


def normaliza_array(array_local, min_local_normalizado, max_local_normalizado):
    """

    :param array_local:
    :param min_local_normalizado:
    :param max_local_normalizado:
    :return:
    """
    if array_local.size == 0:
        return array_local
    try:
        # control de error que emerge en caso de np.max
        # --> "zero-size array to reduction operation maximum which has no identity"
        array_local_min = np.min(array_local)
        intervalo = np.max(array_local) - array_local_min + (np.max(array_local) - array_local_min == 0)
    except Exception as ee1:
        print('Error controlado en la funcion de normalizacion')
        print(ee1)
        intervalo = 1
        array_local_min = 0
        # logger.error(str(ee1), exc_info=True)
    array_normalizado = np.multiply(
        np.add(array_local, - array_local_min),
        (max_local_normalizado - min_local_normalizado) / intervalo, dtype=np.dtype('float32'))
    return np.add(array_normalizado, min_local_normalizado)


def filtro_banda(im, hipass=10, lowpass=245):
    """

    :param im:
    :param hipass:
    :param lowpass:
    :return:
    """
    return np.clip(im, hipass, lowpass)


def pareto(prob_array_local):
    prob_np = np.asarray(prob_array_local, dtype=np.float32)
    largo_prob_np = len(prob_np)
    mediana_prob_np = np.median(prob_np)
    factor_pareto = 0.8
    if mediana_prob_np >= 0.5:
        prob_np = np.sort(prob_np)[int(largo_prob_np * (1 - factor_pareto)): largo_prob_np]
    else:
        prob_np = np.sort(prob_np)[0: int(largo_prob_np *  factor_pareto) + 1]
    return np.mean(prob_np)

                    
def procesa_mp4(video_path):
    """

    :param video_path:
    :return:
    """
    prob_final = 0.5
    try:
        # variables locales y metadata
        video_f = video_path
        # captura de imagenes
        inicio_local = time.time()
        video_cap = cv2.VideoCapture(video_f)
        # no_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # alto_frame = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # ancho_frame = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # index_frames = np.linspace(0, frames_cap_por_video, num=frames_cap_por_video, endpoint=False, dtype=np.int32)
        # columnas = []
        # wchrom_signal = []
        # no_capturas = 0
        no_capt_cara_detectada = 0
        prob_array = []
        # captura frame por frame
        capture_success = True
        while capture_success and time.time() <= inicio_local + tiempo_por_video:
            capture_success, frame_source = video_cap.read()
            if capture_success:
                # convierte a RGB
                frame_RGB = cv2.cvtColor(frame_source, cv2.COLOR_BGR2RGB)
                # detecta cara(s) en frame usando mtcnn-tf
                deteccion = modelo.detect(frame_RGB)
                nu_caras = len(deteccion[1][:])
                if nu_caras > 0:
                    # consideraremos solo la primera cara detectada con mayor seguridad
                    indice_cara = 0
                    confianza = deteccion[1][indice_cara]
                    if confianza >= umbral_confianza_es_cara:
                        y_sup_izq = int(deteccion[0][indice_cara][0])
                        x_sup_izq = int(deteccion[0][indice_cara][1])
                        y_inf_der = int(deteccion[0][indice_cara][2])
                        x_inf_der = int(deteccion[0][indice_cara][3])
                        # estas lineas comentadas corresponden a la extraccion de la ROI dentro de la cara
                        # alto_rectangulo = int((y_inf_der - y_sup_izq) / 3)
                        # ancho_rectangulo = int((x_inf_der - x_sup_izq) / 4)
                        # y_sup_izq_roi = y_sup_izq + int(alto_rectangulo)
                        # x_sup_izq_roi = x_sup_izq + int(ancho_rectangulo)
                        # cara_roi = frame_RGB[y_sup_izq_roi: y_sup_izq_roi + alto_rectangulo,
                        #             x_sup_izq_roi: x_sup_izq_roi + 2 * ancho_rectangulo]
                        # roi_filtrado = filtro_chrominance(cara_roi, file_name, all_targets3, sufix, no_capturas_procesadas)
                        alto_rectangulo = int(y_inf_der - y_sup_izq)
                        ancho_rectangulo = int(x_inf_der - x_sup_izq)
                        cara_roi = frame_RGB[y_sup_izq: y_sup_izq + alto_rectangulo,
                                   x_sup_izq: x_sup_izq + ancho_rectangulo]
                        if np.shape(cara_roi)[0] == 0 or np.shape(cara_roi)[1] == 0:
                            continue
                        # image = cv2.resize(cara_roi, (128, 128), interpolation=cv2.INTER_LINEAR)
                        image = escala(cara_roi, 128, 128)
                        image = np.expand_dims(image, axis=0)
                        image = np.divide(image, 255.)
                        result_prob_real = classifier.predict(image)
                        prob_manipulado = 1 - result_prob_real[0, 0]
                        prob_array.append(prob_manipulado)
        if len(prob_array) > 0:
            return np.mean(prob_array)
        else:
            return 0.65
    except Exception as ee2:
        print('Error controlado en la iteracion de video')
        print(ee2)
        # logger.error(str(ee2), exc_info=True)
    return prob_final


# ___sin___esta___pieza__clave__el__código___puede___parecer__,____pero....___
# __pero __estrictamente___no__sería__un___PROGRAMA___PRINCIPAL_en__python____
if __name__ == '__main__':
    try:
        # paths globales
        video_source_path = '/kaggle/input/deepfake-detection-challenge/test_videos/'
        models_folder = '/kaggle/input/saulbest3/'
        model_detecta_caras_path = '/kaggle/input/mtcmtflib/tf-mtcnn-master/mtcnn.pb'
        # alto_chr_map = 64  # 64 subregiones = 128 filas, cada fila representa una subregion
        # ancho_chr_map = 128  # 128 frames por video, cada columna representa un frame
        # alto_roi = 8 * 2  # 8 es el alto de la ROI
        # ancho_roi = 8 * 2  # 8 es el ancho de la ROI
        # alto_subregion = 2  # alto de cada subregion
        # ancho_subregion = 2  # ancho de cada subregion
        # resultado_alto_chr_map_1 = np.empty(shape=(alto_chr_map, 1), dtype=np.dtype('float32'))
        # resultado_alto_chr_map_0 = np.empty(shape=(alto_chr_map, 0), dtype=np.dtype('float32'))
        tiempo_por_video = 7.9
        umbral_confianza_es_cara = 0.95
        horas_maximo = 8.92
        segundos_maximo = horas_maximo * 3600
        # carga detector de caras (modelo)
        modelo = MTCNN(model_detecta_caras_path)

        # el tiempo será controlado
        inicio = time.time()

        # carga modelo(s) y weights
        try:
            classifier = models.load_model(models_folder + 'F7.1.3_Load_mejor_modelo_CNN_03-0.1597-0.94.hdf5')
            #yaml_file_color = open(models_folder + 'modelo_chrom_map.yaml')
            #classifier_color_model_yaml = yaml_file_color.read()
            #yaml_file_color.close()
            #classifier = model_from_yaml(classifier_color_model_yaml)
            #classifier.load_weights(models_folder + 'filtros_chrom_map.hdf5')
            print('modelo y filtros cargados')
        except Exception as e:
            print('Error en la carga del modelo')
            print(e)
            classifier = None
            # logger.error(str(e), exc_info=True)

        # compila modelo(s)
        try:
            for layer in classifier.layers:
                layer.trainable = False
            classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
            print('modelo compilado')
        except Exception as e:
            print('error en la compilación del modelo')
            print(e)
            # logger.error(str(e), exc_info=True)

        # extrae ubicación de cada archivo de video
        try:
            file_loc = sorted(glob.glob(video_source_path + '*.mp4'))
            no_videos = len(file_loc)
            print('rutas de archivos de video almacenadas y ordenadas')
        except Exception as e:
            print('Error en extracción de la ubicación de cada archivo')
            print(e)
            file_loc = []
            no_videos = 0
            # logger.error(str(e), exc_info=True)

        # recorre cada video
        video_frame_data = []
        try:
            [video_frame_data.append(tuple((ruta[len(video_source_path):], procesa_mp4(ruta))))
            for ruta in file_loc if time.time() - inicio < segundos_maximo]

        except Exception as ee:
            print("Error controlado en el recorrido de videos, captura de frames y prediccion")
            print(ee)

        # genera submission.csv, controla predicciones no alzanzadas a realizar llevando a 0.5
        try:
            videos_no_procesados = no_videos - len(video_frame_data)
            if videos_no_procesados > 0:
                for ruta in file_loc[-videos_no_procesados:]:
                    video_frame_data = np.append(video_frame_data, [[ruta[len(video_source_path):], 0.5]], axis=0)
            video_frame_data = np.asarray(video_frame_data)
            submission_final = pd.DataFrame({"filename": video_frame_data[:][:, 0], "label": video_frame_data[:][:, 1]})
            submission_final.to_csv("submission.csv", index=False)
        except Exception as e:
            print('Error en la generación de archivo de submisión CSV')
            print(e)
            # logger.error(str(e), exc_info=True)

        # cálculo de tiempo total
        duracion = int(time.time() - inicio)
        minutos = int(duracion / 60)
        segundos = duracion % 60
        print('todo terminado, duración (mm:ss):', minutos, segundos)
    except Exception as ee:
        print("Error controlado en la ejecución del bloque principal ___'___main_____'____")
        print(ee)
        # logger.error(str(ee), exc_info=True)
