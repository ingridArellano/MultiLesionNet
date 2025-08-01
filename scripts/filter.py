#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import gc
import argparse
import numpy as np
import SimpleITK as sitk

from time import time

from CTLungSeg.utils import read_image, normalize
from CTLungSeg.utils import write_volume
from CTLungSeg.method import median_filter, std_filter, gauss_smooth
from CTLungSeg.method import adaptive_histogram_equalization, adjust_gamma


def parse_args():
    description = 'Image labeling'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--input',
                        dest='filename',
                        required=True,
                        type=str,
                        action='store',
                        help='Input filename')
    parser.add_argument('--output',
                        dest='output',
                        required=True,
                        type=str,
                        action='store',
                        help='output name label')

    args = parser.parse_args()
    return args


def main(volume, output_base):
    
    # Aplicar los filtros y normalizar
    equalized = normalize(adaptive_histogram_equalization(image=volume, radius=5))
    median = normalize(median_filter(img=volume, radius=3))
    std = normalize(std_filter(image=volume, radius=3))
    gamma = normalize(adjust_gamma(image=volume, gamma=1.5))
    gauss = normalize(gauss_smooth(image=volume))

    # Guardar cada filtro individual como archivo .nii.gz
    write_volume(equalized, output_base + "_equalized.nii.gz")
    write_volume(median, output_base + "_median.nii.gz")
    write_volume(std, output_base + "_std.nii.gz")
    write_volume(gamma, output_base + "_gamma.nii.gz")
    write_volume(gauss, output_base + "_gauss.nii.gz")


if __name__ == '__main__' :

    start = time()
    #load parameters
    args = parse_args()
    volume = read_image(filename=args.filename)

    main(volume, args.output)
    
    stop = time()
    print('Process ended after {0:.3f} seconds'.format(stop - start))
