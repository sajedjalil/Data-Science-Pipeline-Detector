#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, dicom

# abspath_datasets_dir     = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'dataset_kaggle', 'sample')
abspath_datasets_dir     = '/Volumes/TRANSCEND_G/data-science-bowl-2017/dataset_dcm/stage1'
abspath_output_csv       = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.path.basename(__file__).replace('.py', '.csv'))
list_abspath_patient_dir = []

for str_name_file_and_directory in os.listdir(abspath_datasets_dir):
    abspath_file_and_directory = os.path.join(abspath_datasets_dir, str_name_file_and_directory)
    if os.path.isfile(abspath_file_and_directory) == False:
        list_abspath_patient_dir.append(abspath_file_and_directory)

file_output_csv   = open(abspath_output_csv, 'w')
file_output_csv.write('patient_id,image_length_x_mm,image_length_y_mm,image_length_z_mm,image_pixel_x_mm,image_pixel_y_mm,image_slice_z_mm\n')
file_output_csv.close()

for abspath_patient_dir in list_abspath_patient_dir:
    list_image_length_x_mm = []
    list_image_length_y_mm = []
    list_image_pixel_x_mm  = []
    list_image_pixel_y_mm  = []
    list_position_z        = []
    for str_name_dcm_file in os.listdir(abspath_patient_dir):
        if '.dcm' in str_name_dcm_file.lower():
            dicom_dcm_file  = dicom.read_file(os.path.join(abspath_patient_dir, str_name_dcm_file))
            list_image_length_x_mm.append(dicom_dcm_file.Columns * dicom_dcm_file.PixelSpacing[1])
            list_image_length_y_mm.append(dicom_dcm_file.Rows    * dicom_dcm_file.PixelSpacing[0])
            list_image_pixel_x_mm.append( dicom_dcm_file.PixelSpacing[1])
            list_image_pixel_y_mm.append( dicom_dcm_file.PixelSpacing[0])
            list_position_z.append(dicom_dcm_file[0x0020, 0x0032][2])

    image_length_x_mm = max(list_image_length_x_mm)
    image_length_y_mm = max(list_image_length_y_mm)
    image_length_z_mm = max(list_position_z) - min(list_position_z)
    image_pixel_x_mm  = min(list_image_pixel_x_mm)
    image_pixel_y_mm  = min(list_image_pixel_y_mm)

    list_position_z.sort()
    list_image_slice_z_mm = []
    for i in range(0, len(list_position_z) - 1):
        list_image_slice_z_mm.append(list_position_z[i+1] - list_position_z[i])

    image_slice_z_mm  = min(list_image_slice_z_mm)

    file_output_csv   = open(abspath_output_csv, 'a')
    file_output_csv.write('%s,%s,%s,%s,%s,%s,%s\n' % (os.path.basename(abspath_patient_dir), image_length_x_mm, image_length_y_mm, image_length_z_mm, image_pixel_x_mm, image_pixel_y_mm, image_slice_z_mm))
    file_output_csv.close()
