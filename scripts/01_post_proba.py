#!/usr/bin/env python

"""
@author: Jordan Graesser

This script applies post-processing to class-conditional posterior probabilities
in order to stabilize land cover transitions over time.
"""

import os
import sys
import time
import shutil
import argparse
import fnmatch

import utils

from mpglue.errors import logger
from mpglue import vrt_builder
from mpglue import raster_tools
from mpglue.classification._moving_window import moving_window

import mtlchmm

import numpy as np
from osgeo import gdal


class PostProbaProcess(object):

    """
    Args:
        input_dir (str): The input base directory.
            *This directory should contain sub-directories with /zone_##/probs.
        mapping_zones (list): A list of mapping zones to process.
        transition_prior (float): The HMM transition prior.
        n_jobs (int): The number of HMM parallel jobs.
    """

    def __init__(self, input_dir, mapping_zones, transition_prior, n_jobs):

        self.input_dir = input_dir
        self.mapping_zones = mapping_zones
        self.transition_prior = transition_prior
        self.n_jobs = n_jobs

        if not self.mapping_zones:
            self.mapping_zones = utils.MAPPING_ZONES

        self.temp_dir = os.path.join(self.input_dir, 'temp')
        self.hmm_dir = os.path.join(self.input_dir, 'hmm_probs')
        self.hmm_map_dir = os.path.join(self.input_dir, 'hmm_maps')

        if not os.path.isdir(self.temp_dir):
            os.makedirs(self.temp_dir)

        self.block_size = 2400
        self.pad = 1

        # Create a dictionary for composting.
        self.comp_dict = dict()

        self.proba_composites = dict()

    def run(self):

        ######################
        # STEP 1
        # Mosaic mapping zones
        ######################
        self.composite()

        ################
        # STEP 2
        # Fill line gaps
        ################
        self.fill_gaps()

        #############################
        # STEP 3
        # Run HMM to stabilize class-
        #   conditional probabilities
        #############################
        self.hmm()

        ###################
        # STEP 4
        # Rank the adjusted
        #   probabilities.
        ###################
        self.rank_probabilities()

    def rank_probabilities(self):

        """Ranks HMM-adjusted probabilities for each year"""

        if not os.path.isdir(self.hmm_map_dir):
            os.makedirs(self.hmm_map_dir)

        for year in utils.YEAR_LIST:

            logger.info('  Get the adjusted classes for year {YEAR} ...'.format(YEAR=year))

            # The HMM probabilities
            hmm_probs = os.path.join(self.hmm_dir,
                                     'fill_{YEAR}_hmm.tif'.format(YEAR=year))

            hmm_map = os.path.join(self.hmm_map_dir,
                                   '{YEAR}.tif'.format(YEAR=year))

            with raster_tools.ropen(hmm_probs) as p_info:

                o_info = p_info.copy()

                o_info.update_info(storage='byte',
                                   bands=1)

                out_rst = raster_tools.create_raster(hmm_map,
                                                     o_info)

                out_rst.get_band(1)

                for i in range(0, p_info.rows, self.block_size):

                    n_rows = raster_tools.n_rows_cols(i, self.block_size, p_info.rows)

                    for j in range(0, p_info.cols, self.block_size):

                        n_cols = raster_tools.n_rows_cols(j, self.block_size, p_info.cols)

                        # Open the probabilities
                        proba_array = p_info.read(bands2open=self.n_jobs,
                                                  i=i,
                                                  j=j,
                                                  rows=n_rows,
                                                  cols=n_cols,
                                                  d_type='float32')

                        if proba_array.max() > 0:

                            # Find the probability layer
                            #   with the largest value.
                            classes2write = np.argmax(proba_array, axis=0) + 1

                            # Mask background
                            no_data_idx = np.where(proba_array.max(axis=0) == 0)

                            if np.any(no_data_idx):
                                classes2write[no_data_idx] = 0

                            # Write to file.
                            out_rst.write_array(classes2write, i=i, j=j)

            out_rst.close_band()
            out_rst.close_file()

            del out_rst, p_info

        logger.info('  Finished\n')

    def hmm(self):

        """Apply Hidden Markov Model"""

        logger.info('  Running Hidden Markov Model ...')

        hmm_model = mtlchmm.MTLCHMM(self.lc_probabilities)

        hmm_model.fit(method='forward-backward',
                      transition_prior=self.transition_prior,
                      n_jobs=self.n_jobs)

        if not os.path.isdir(self.hmm_dir):
            os.makedirs(self.hmm_dir)

        hmm_list = fnmatch.filter(os.listdir(self.temp_dir), '*hmm.tif')

        # Move the HMM files
        for hmm_file in hmm_list:

            hmm_old = os.path.join(self.temp_dir, hmm_file)
            hmm_new = os.path.join(self.hmm_dir, hmm_file)

            shutil.move(hmm_old, hmm_new)

        logger.info('  Finished\n')

    def fill_gaps(self):

        """Fills 'bad pixel' gaps"""

        weights = np.array([[.01, .1, .3, .1, .01],
                            [.1, .5, .95, .5, .1],
                            [.3, .95, 1, .95, .3],
                            [.1, .5, .95, .5, .1],
                            [.01, .1, .3, .1, .01]], dtype='float32')

        self.lc_probabilities = list()

        for year in sorted(self.comp_dict.keys()):

            logger.info('  Filling gaps for year {YEAR} ...'.format(YEAR=year.upper()))

            year_fill = os.path.join(self.temp_dir,
                                     'proba_composite_fill_{YEAR}.tif'.format(YEAR=year))

            self.lc_probabilities.append(year_fill)

            if os.path.isfile(year_fill):
                continue

            with raster_tools.ropen(self.proba_composites[year]) as p_info:

                out_rst = raster_tools.create_raster(year_fill,
                                                     p_info)

                for i in range(0, p_info.rows, self.block_size):

                    ii = i - self.pad if i - self.pad >= 0 else i
                    blkr = self.block_size + self.pad if i - ii == 0 else self.block_size + self.pad + 1

                    n_rows_write = raster_tools.n_rows_cols(i, self.block_size, p_info.rows)
                    n_rows = raster_tools.n_rows_cols(ii, blkr, p_info.rows)

                    for j in range(0, p_info.cols, self.block_size):

                        jj = j - self.pad if j - self.pad >= 0 else j
                        blkc = self.block_size + self.pad if j - jj == 0 else self.block_size + self.pad + 1

                        n_cols_write = raster_tools.n_rows_cols(j, self.block_size, p_info.cols)
                        n_cols = raster_tools.n_rows_cols(jj, blkc, p_info.cols)

                        proba_array = p_info.read(bands2open=self.n_jobs,
                                                  i=ii,
                                                  j=jj,
                                                  rows=n_rows,
                                                  cols=n_cols,
                                                  d_type='float32')

                        if proba_array.max() > 0:

                            # Fill 'no data' gaps.
                            for pi, proba_layer in enumerate(proba_array):

                                # Fill gaps on first round
                                proba_layer_w = moving_window(proba_layer,
                                                              statistic='mean',
                                                              window_size=3,
                                                              target_value=0,
                                                              ignore_value=0)

                                # Smooth on second round
                                out_rst.write_array(moving_window(proba_layer_w,
                                                                  statistic='mean',
                                                                  window_size=5,
                                                                  target_value=-9999,
                                                                  ignore_value=-9999,
                                                                  weights=weights)[i-ii:i-ii+n_rows_write,
                                                                                   j-jj:j-jj+n_cols_write],
                                                    i=i,
                                                    j=j,
                                                    band=pi+1)

                                out_rst.close_band()

            del p_info

            out_rst.close_file()
            del out_rst

        logger.info('  Finished\n')

    def composite(self):

        """Composites annual probabilities for all zones"""

        # Get a list of files to mosaic.
        for root, dirs, files in os.walk(self.input_dir):

            if not files:
                continue

            if 'zone_' not in root:
                continue

            # Filter GeoTiffs
            tiff_list = fnmatch.filter(files, '*.tif')

            if not tiff_list:
                continue

            current_zone = root[root.find('zone'):root.find('zone')+8]

            if current_zone[-1] == '/':
                current_zone = current_zone[:-1]

            if self.mapping_zones:

                if current_zone not in ('zone_' + ',zone_'.join(self.mapping_zones)).split(','):
                    continue

            # The class proba layers for the current zone.
            zone_proba_layers = utils.PROBA_LAYERS[current_zone]

            for year in tiff_list:

                year_str = year.replace('.tif', '')

                tiff_path = os.path.join(root, year)

                out_tiff = os.path.join(self.temp_dir,
                                        'proba_composite_{YEAR}_{ZONE}.tif'.format(YEAR=year_str,
                                                                                   ZONE=current_zone.replace('zone_', '')))

                if not os.path.isfile(out_tiff):

                    f_base = os.path.splitext(year)[0]

                    zone_year_stack = dict()

                    # Force 9-band images.
                    for bi, class_name in enumerate(sorted(zone_proba_layers, key=zone_proba_layers.get)):

                        logger.info('  Subsetting class {} for zone {} ...'.format(class_name.upper(),
                                                                                   current_zone.upper()))

                        bd = utils.BAND_ORDER[class_name]

                        # class_dict = {k: v for k, v in zone_proba_layers.items() if v == bd}

                        # 9-layer probabilities
                        tiff_path_standard = os.path.join(self.temp_dir,
                                                          '{BASE}_standard_{CLASS_NAME}.tif'.format(BASE=f_base,
                                                                                                    CLASS_NAME=class_name))

                        zone_year_stack[str(bd)] = [tiff_path_standard]

                        if os.path.isfile(tiff_path_standard):
                            os.remove(tiff_path_standard)

                        if bi == 0:

                            # Return the raster object in
                            #   order to get the zone extent.
                            out_ds = raster_tools.translate(tiff_path,
                                                            tiff_path_standard,
                                                            d_type='float32',
                                                            return_datasource=True,
                                                            bandList=[bi + 1],
                                                            scaleParams=[[0, 255, 0, 1]])

                        else:

                            # Force to the same extent
                            #   as the other years.
                            raster_tools.translate(tiff_path,
                                                   tiff_path_standard,
                                                   d_type='float32',
                                                   bandList=[bi + 1],
                                                   scaleParams=[[0, 255, 0, 1]],
                                                   projWin=[out_ds.left,
                                                            out_ds.top,
                                                            out_ds.right,
                                                            out_ds.bottom])

                    # Add empty layers for the remaining classes.
                    for class_name in list(set(utils.CLASS_LIST).difference(set(zone_proba_layers.keys()))):

                        logger.info('  Subsetting class {} for zone {} ...'.format(class_name.upper(),
                                                                                   current_zone.upper()))

                        bd = utils.BAND_ORDER[class_name]

                        # 9-layer probabilities
                        tiff_path_standard = os.path.join(self.temp_dir,
                                                          '{BASE}_standard_{CLASS_NAME}.tif'.format(BASE=f_base,
                                                                                                    CLASS_NAME=class_name))

                        zone_year_stack[str(bd)] = [tiff_path_standard]

                        if os.path.isfile(tiff_path_standard):
                            os.remove(tiff_path_standard)

                        # Scale band 1 to zero.
                        raster_tools.translate(tiff_path,
                                               tiff_path_standard,
                                               d_type='float32',
                                               bandList=[1],
                                               scaleParams=[[0, 255, 0, 0]],
                                               projWin=[out_ds.left,
                                                        out_ds.top,
                                                        out_ds.right,
                                                        out_ds.bottom])

                    out_vrt = os.path.join(self.temp_dir,
                                           'proba_composite_{YEAR}.vrt'.format(YEAR=year_str))

                    # Stack the current zone.
                    vrt_builder(zone_year_stack,
                                out_vrt,
                                force_type='float32',
                                overwrite=True,
                                be_quiet=True)

                    if os.path.isfile(out_tiff):
                        os.remove(out_tiff)

                    logger.info('  Compositing classes for zone {} ...'.format(current_zone.upper()))

                    # Store the composite as a GeoTiff.
                    raster_tools.translate(out_vrt,
                                           out_tiff,
                                           projWin=[out_ds.left,
                                                    out_ds.top,
                                                    out_ds.right,
                                                    out_ds.bottom])

                    del out_ds

                    os.remove(out_vrt)

                    for k, vl in zone_year_stack.iteritems():
                        for fn in vl:
                            os.remove(fn)

                    if os.path.isfile(out_tiff + '.aux.xml'):
                        os.remove(out_tiff + '.aux.xml')

                if year_str not in self.comp_dict:
                    self.comp_dict[year_str] = [out_tiff]
                else:
                    self.comp_dict[year_str].append(out_tiff)

        # Composite the probabilities for each year.
        year_counter = 1
        for year, image_list in self.comp_dict.iteritems():

            logger.info('  Compositing year {} ...'.format(year))

            comp_dict = {'1': image_list}

            out_vrt = os.path.join(self.temp_dir,
                                   'proba_composite_{YEAR}.vrt'.format(YEAR=year))

            out_saaeac = os.path.join(self.temp_dir,
                                      'proba_composite_{YEAR}.tif'.format(YEAR=year))

            self.proba_composites[year] = out_saaeac

            if not os.path.isfile(out_saaeac):

                vrt_builder(comp_dict,
                            out_vrt,
                            force_type='float32',
                            overwrite=True,
                            be_quiet=True)

                if year_counter == 1:

                    out_ds = raster_tools.warp(out_vrt,
                                               out_saaeac,
                                               in_epsg=4326,
                                               out_epsg=102033,
                                               return_datasource=True,
                                               resample='nearest',
                                               cell_size=250.,
                                               multithread=True,
                                               creationOptions=['GDAL_CACHEMAX=256',
                                                                'TILED=YES'])

                else:

                    raster_tools.warp(out_vrt,
                                      out_saaeac,
                                      in_epsg=4326,
                                      out_epsg=102033,
                                      resample='nearest',
                                      cell_size=250.,
                                      outputBounds=[out_ds.left,
                                                    out_ds.bottom,
                                                    out_ds.right,
                                                    out_ds.top],
                                      multithread=True,
                                      creationOptions=['GDAL_CACHEMAX=256',
                                                       'TILED=YES'])

                os.remove(out_vrt)

            # if year_counter == 1:
            #
            #     vrt_options = gdal.BuildVRTOptions(resolution='highest',
            #                                        )
            #
            #     gdal.BuildVRT(out_vrt,
            #                   image_list)
            #
            # vrt_options = gdal.BuildVRTOptions(outputBounds=[minX,
            #                                                  minY,
            #                                                  maxX,
            #                                                  maxY])

            year_counter += 1

        out_ds = None

        logger.info('  Finished\n')


def _examples():

    sys.exit("""\
    
    01_post_proba.py -i /LandMapper/ciga_lac_lulc_9class/processed/01_maps_hard -z 01 02 03
    
    """)


def main():

    parser = argparse.ArgumentParser(description='Probability post-processing',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-e', '--examples', dest='examples', action='store_true',
                        help='Show usage examples and exit')
    parser.add_argument('-i', '--input-dir', dest='input_dir', help='The input directory', default=None)
    parser.add_argument('-z', '--mapping-zones', dest='mapping_zones', help='A list of mapping zones',
                        default=['01', '02'], nargs='+')
    parser.add_argument('--prior', dest='transition_prior', help='The HMM transition prior', default=.1, type=float)
    parser.add_argument('--n-jobs', dest='n_jobs', help='The number of HMM parallel jobs', default=1, type=int)

    args = parser.parse_args()

    if args.examples:
        _examples()

    logger.info('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))

    start_time = time.time()

    ppp = PostProbaProcess(args.input_dir,
                           args.mapping_zones,
                           args.transition_prior,
                           args.n_jobs)

    ppp.run()

    logger.info('\nEnd data & time -- (%s)\nTotal processing time -- (%.2gs)\n' %
                (time.asctime(time.localtime(time.time())), (time.time() - start_time)))


if __name__ == '__main__':
    main()
