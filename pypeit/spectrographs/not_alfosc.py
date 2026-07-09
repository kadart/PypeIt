"""
Module for NOT ALFOSC spectrograph

.. include:: ../include/links.rst
"""
from IPython import embed

import re
import numpy as np

from astropy.time import Time

from pypeit import log
from pypeit import PypeItError
from pypeit import telescopes
from pypeit.core import framematch
from pypeit.spectrographs import spectrograph
from pypeit.core import parse
from pypeit.images import detector_container


class NOTALFOSCSpectrograph(spectrograph.Spectrograph):
    """
    Child to handle NOT ALFOSC spectrograph
    """
    ndet = 1
    name = 'not_alfosc'
    telescope = telescopes.NOTTelescopePar()
    camera = 'ALFOSC'
    url = 'https://www.not.iac.es/instruments/alfosc/'
    header_name = 'ALFOSC_FASU'   # [ERROR]   :: ALFOSC_FASU is not a supported spectrograph ?
    supported = True
    comment = 'For use with the standard horizontal slits only. Grisms 3, 4, 5, 7, 8, 10, 11, 17, 18, 19, 20'

    def get_detector_par(self, det, hdu=None):
        """
        Return metadata for the selected detector.

        Detector data from `here
        <http://www.not.iac.es/instruments/detectors/CCD14/>`__.

        .. warning::

            Many of the necessary detector parameters are read from the file
            header, meaning the ``hdu`` argument is effectively **required** for
            NOT/ALFOSC.  The optional use of ``hdu`` is only viable for
            automatically generated documentation.

        Args:
            det (:obj:`int`):
                1-indexed detector number.
            hdu (`astropy.io.fits.HDUList`_, optional):
                The open fits file with the raw image of interest.  If not
                provided, frame-dependent parameters are set to a default.

        Returns:
            :class:`~pypeit.images.detector_container.DetectorContainer`:
            Object with the detector metadata.
        """
        # http://www.not.iac.es/instruments/detectors/CCD14/

        if hdu is None:
            binning = '1,1'
            datasec = None
            gain = None
            ronoise = None
        else:
            binning = self.get_meta_value(self.get_headarr(hdu), 'binning')
            datasec = np.atleast_1d(self.get_meta_value(self.get_headarr(hdu), 'datasec'))
            gain = np.atleast_1d(hdu[1].header['GAIN'])  # e-/ADU
            ronoise = np.atleast_1d(hdu[1].header['RDNOISE'])  # e-

        detector_dict = dict(
            binning         = binning,
            det             = 1,
            dataext         = 1,
            specaxis        = 0,
            specflip        = True,
            spatflip        = False,
            xgap            = 0.,
            ygap            = 0.,
            ysize           = 1.,
            platescale      = 0.2138,
            mincounts       = -1e10,
            darkcurr        = 1.3,      # e-/pix/hr
            saturation      = 700000.,  # ADU
            nonlinear       = 0.86,
            datasec         = np.atleast_1d('[:,:]'),  # datasec
            oscansec        = np.atleast_1d('[10:,2068:2092]'),  # Unbinned
            numamplifiers   = 1,
            gain            = gain,     # e-/ADU
            ronoise         = ronoise   # e-
        )

        return detector_container.DetectorContainer(**detector_dict)

    def bpm(self, filename, det, shape=None, msbias=None):
        """
        Generate a default bad-pixel mask.

        Even though they are both optional, either the precise shape for
        the image (``shape``) or an example file that can be read to get
        the shape (``filename`` using :func:`get_image_shape`) *must* be
        provided.

        Args:
            filename (:obj:`str` or None):
                An example file to use to get the image shape.
            det (:obj:`int`):
                1-indexed detector number to use when getting the image
                shape from the example file.
            shape (tuple, optional):
                Processed image shape
                Required if filename is None
                Ignored if filename is not None
            msbias (`numpy.ndarray`_, optional):
                Master bias frame used to identify bad pixels

        Returns:
            `numpy.ndarray`_: An integer array with a masked value set
            to 1 and an unmasked value set to 0.  All values are set to
            0.
        """
        # Call the base-class method to generate the empty bpm
        bpm_img = super().bpm(filename, det, shape=shape, msbias=msbias)
        # Should return an empty bpm
        return bpm_img

    @classmethod
    def default_pypeit_par(cls):
        """
        Return the default parameters to use for this instrument.

        Returns:
            :class:`~pypeit.par.pypeitpar.PypeItPar`: Parameters required by
            all of PypeIt methods.
        """
        # Option when there are no bias frames provided in reduction
        #turn_on = dict(use_biasimage=False, use_overscan=True, overscan_method='median',
        #               use_darkimage=False)
        #par.reset_all_processimages_par(**turn_on)

        par = super().default_pypeit_par()

        # Ignore PCA
        par['calibrations']['slitedges']['sync_predict'] = 'nearest'
        par['calibrations']['slitedges']['bound_detector'] = True
        # Flats are sometimes quite ugly due to dust on the slit which leads to the erroneous detection of multiple slits. So set a higher edge_thresh and minimum_slit_gap.
        par['calibrations']['slitedges']['edge_thresh'] = 30
        par['calibrations']['slitedges']['minimum_slit_gap'] = 15
    
        # Set pixel flat combination method
        par['calibrations']['pixelflatframe']['process']['combine'] = 'median'
        # Wavelength calibration methods
        #par['calibrations']['wavelengths']['method'] = 'holy-grail'
        par['calibrations']['wavelengths']['method'] = 'full_template'
        par['calibrations']['wavelengths']['lamps'] = ['HeI', 'NeI', 'ArI']
        par['calibrations']['wavelengths']['sigdetect'] = 10.0
        # Set the default exposure time ranges for the frame typing
        par['calibrations']['biasframe']['exprng'] = [None, 1]
        par['calibrations']['darkframe']['exprng'] = [999999, None]     # No dark frames
        par['calibrations']['pinholeframe']['exprng'] = [999999, None]  # No pinhole frames
        par['calibrations']['arcframe']['exprng'] = [None, None]  # Long arc exposures on this telescope
        par['calibrations']['standardframe']['exprng'] = [None, 150]
        par['scienceframe']['exprng'] = [1.999999, None]

        # Multiple arcs with different lamps, so can't median combine nor clip, also need to remove continuum
        par['calibrations']['arcframe']['process']['clip'] = False
        par['calibrations']['arcframe']['process']['combine'] = 'mean'
        par['calibrations']['arcframe']['process']['subtract_continuum'] = True
        par['calibrations']['tiltframe']['process']['clip'] = False
        par['calibrations']['tiltframe']['process']['combine'] = 'mean'
        par['calibrations']['tiltframe']['process']['subtract_continuum'] = True


        # Added 08-04-2025
        # No clipping in base-process 
        par['scienceframe']['process']['clip'] = False
        par['scienceframe']['process']['combine'] = 'mean'
        par['calibrations']['standardframe']['process']['spat_flexure_correct'] = True
        par['scienceframe']['process']['spat_flexure_correct'] = True
        par['reduce']['findobj']['maxnumber_sci'] = 4
        par['reduce']['findobj']['maxnumber_std'] = 2
        #par['reduce']['skysub']['sn_gauss'] = 10.0

        #par['reduce']['extraction']['skip_optimal'] = True
        par['reduce']['extraction']['model_full_slit'] = True
        par['reduce']['extraction']['use_2dmodel_mask'] = False
        par['sensfunc']['IR']['telgridfile'] = 'TellPCA_3000_10500_R120000.fits'
        par['telluric']['telgridfile'] = 'TellPCA_3000_10500_R120000.fits'
        par['telluric']['teltype'] = 'pca'
        par['sensfunc']['polyorder'] = 2
        par['sensfunc']['extrap_blu'] = 0.1
        par['sensfunc']['extrap_red'] = 0.1
        par['sensfunc']['mask_helium_lines'] = False
        #par['sensfunc']['UVIS']['nresln'] = 15
        #par['sensfunc']['UVIS']['resolution'] = 1000.0  # global ALFOSC default; grism-specific overrides below
        # Don't do fine-correction for standard stars?
        #par['calibrations']['standardframe']['process']['use_illumflat'] = False
        #par['calibrations']['standardframe']['process']['use_pixelflat'] = False
        #par['calibrations']['flatfield']['pixelflat_min_wave'] = 4000.
        #par['reduce']['findobj']['trace_min_max'] = 20,2020


        # No overscan region!
        turn_off = dict(use_overscan=False)
        #par['calibrations']['biasframe']['process']['use_overscan'] = True

        par.reset_all_processimages_par(**turn_off)

        return par

    @staticmethod
    def ql_par():
        """
        Return quick-look specific parameters for NOT ALFOSC.
        
        This overrides the base class to add ALFOSC-specific quick-look settings
        optimized for speed over accuracy.
        
        Returns:
            dict: Dictionary with quick-look parameter overrides
        """
        # Start with base quick-look parameters
        par = super(NOTALFOSCSpectrograph, NOTALFOSCSpectrograph).ql_par()
        

        if 'scienceframe' not in par:
            par['scienceframe'] = {}
        if 'process' not in par['scienceframe']:
            par['scienceframe']['process'] = {}
        
        par['scienceframe']['process']['skip_write_2d'] = True
        par['scienceframe']['process']['use_illumflat'] = False
        
        if 'reduce' not in par:
            par['reduce'] = {}
        if 'extraction' not in par['reduce']:
            par['reduce']['extraction'] = {}
        
        par['reduce']['extraction']['skip_optimal'] = True
        par['reduce']['extraction']['model_full_slit'] = False
        par['reduce']['extraction']['use_2dmodel_mask'] = False
        

        if 'skysub' not in par['reduce']:
            par['reduce']['skysub'] = {}
        
        par['reduce']['skysub']['bspline_spacing'] = 0.8  # Coarser spacing (default 0.6)
        par['reduce']['skysub']['no_poly'] = False  # Allow polynomial subtraction
        
        if 'findobj' not in par['reduce']:
            par['reduce']['findobj'] = {}
        
        par['reduce']['findobj']['skip_second_find'] = True
        par['reduce']['findobj']['snr_thresh'] = 5.0
        

        if 'flexure' not in par:
            par['flexure'] = {}

        par['flexure']['spec_method'] = 'skip'
        
        if 'calibrations' not in par:
            par['calibrations'] = {}
        if 'wavelengths' not in par['calibrations']:
            par['calibrations']['wavelengths'] = {}
        
        par['calibrations']['wavelengths']['method'] = 'full_template'
        par['calibrations']['wavelengths']['sigdetect'] = 15.0  # default is 10
        
        return par

    def init_meta(self):
        """
        Define how metadata are derived from the spectrograph files.

        That is, this associates the PypeIt-specific metadata keywords
        with the instrument-specific header cards using :attr:`meta`.
        """
        self.meta = {}
        # Required (core)
        #self.meta['ra'] = dict(ext=0, card='OBJRA')
        self.meta['ra'] = dict(card=None, compound=True)
        self.meta['dec'] = dict(ext=0, card='OBJDEC')
        self.meta['target'] = dict(ext=0, card='OBJECT')
        self.meta['decker'] = dict(ext=0, card='ALAPRTNM')
        self.meta['binning'] = dict(card=None, compound=True)
        self.meta['datasec'] = dict(ext=0, card='DETWIN1')
        self.meta['mjd'] = dict(ext=0, card=None, compound=True)
        self.meta['exptime'] = dict(ext=0, card='EXPTIME')
        self.meta['airmass'] = dict(ext=0, card='AIRMASS')
        # Extras for config and frametyping
        self.meta['dispname'] = dict(ext=0, card='ALGRID')  # use grism #? 
        self.meta['idname'] = dict(ext=0, card='IMAGETYP')
        self.meta['instrument'] = dict(ext=0, card='INSTRUME')

        #not_alfosc_* subdirectories created for each filter (order blockers)
        # Compound: order blockers can live in FA, FB, or AL filter wheels
        self.meta['decker_secondary'] = dict(card=None, compound=True)

    def compound_meta(self, headarr, meta_key):
        if meta_key == 'binning':
            binspatial = headarr[0]['DETXBIN']
            binspec = headarr[0]['DETYBIN']
            return parse.binning2string(binspec, binspatial)
        elif meta_key == 'mjd':
            time = headarr[0]['DATE-AVG']
            ttime = Time(time, format='isot')
            return ttime.mjd
        elif meta_key == 'ra':
            objra = headarr[0]['OBJRA'] # Given in hours, not deg
            return objra*15.
        elif meta_key == 'decker_secondary':
            # Order blockers can be placed in any of three filter wheels.
            # Each wheel has its own Open-slot ID: FA=1001, FB=1002, AL=1005.
            # Return the ID of the first wheel that is not in the Open position.
            _OPEN_IDS = {'FAFLTID': 1001, 'FBFLTID': 1002, 'ALFLTID': 1005}
            for id_key, open_id in _OPEN_IDS.items():
                flt_id = headarr[0].get(id_key)
                if flt_id is not None and int(flt_id) != open_id:
                    return str(flt_id)
            # Fallback: return FAFLTID regardless
            return str(headarr[0].get('FAFLTID', ''))
        raise PypeItError("Not ready for this compound meta")

    def configuration_keys(self):
        """
        Return the metadata keys that define a unique instrument
        configuration.

        This list is used by :class:`~pypeit.metadata.PypeItMetaData` to
        identify the unique configurations among the list of frames read
        for a given reduction.

        Returns:
            :obj:`list`: List of keywords of data pulled from file headers
            and used to constuct the :class:`~pypeit.metadata.PypeItMetaData`
            object.
        """
        return ['dispname','decker',
        'decker_secondary',
        'binning','datasec']

    def raw_header_cards(self):
        """
        Return additional raw header cards to be propagated in
        downstream output files for configuration identification.

        Copying full NOT ALFOSC header to 0th extension.

        The list of raw data FITS keywords should be those used to populate
        the :meth:`~pypeit.spectrographs.spectrograph.Spectrograph.configuration_keys`
        or are used in :meth:`~pypeit.spectrographs.spectrograph.Spectrograph.config_specific_par`
        for a particular spectrograph, if different from the name of the
        PypeIt metadata keyword.

        This list is used by :meth:`~pypeit.spectrographs.spectrograph.Spectrograph.subheader_for_spec`
        to include additional FITS keywords in downstream output files.

        Returns:
            :obj:`list`: List of keywords from the raw data files that should
            be propagated in output files.
        """
        return ['BITPIX','NAXIS   ','EXTEND  ','COMMENT ','COMMENT ','BZERO   ','BSCALE  ','DATE-OBS',
'DATE_OBS','DATE    ','FILENAME','ORIGIN  ','OBSERVAT','TELESCOP','INSTRUME','DETNAME ','CHIPID  ',
'CREATOR ','XOVERSC ','YOVERSC ','DATE-AVG','ALAPRTNM','ALAPRTID','ALAPRPOS','ALAPRSTP','ALAPRORI',
'ALAPRALG','ALFLTNM ','ALFLTID ','ALFLTPOS','ALFLTSTP','ALFLTORI','ALGRNM  ','ALGRID  ','ALGRPOS ',
'ALGRSTP ','ALGRORI ','ALGRALG ','ALFOCUS ','ALCENWAV','FAFLTNM ','FAFLTID ','FAFLTPOS','FBFLTNM ','FBFLTID ',
'FBFLTPOS','CLAMP1  ','CLAMPNM1','CLAMPID1','CLAMP2  ','CLAMPNM2','CLAMPID2','CLAMP3  ','CLAMPNM3','CLAMPID3',
'CLAMP4  ','CLAMPNM4','CLAMPID4','CMIRROR ','FARETARD','FAPOLID ','FARETANG','ALAPRSLX','ALAPRSLY','BFLMP1  ',
'BFLMP1ID','BFLMP1NM','BFLMP2  ','BFLMP2ID','BFLMP2NM','BFLMP3  ','BFLMP3ID','BFLMP3NM','UT      ','ST      ',
'RA      ','DEC     ','EQUINOX ','RADECSYS','TELALT  ','AZIMUTH ','AIRMASS ','FIELD   ','ROTPOS  ','TELFOCUS',
'TRACKMOD','DTRCK_RA','DTRCK_DE','CCDPROBE','ADCARM  ','ADCMODE ','ADC1ANG ','ADC2ANG ','ADC1ENC ','ADC2ENC ',
'AUSTATUS','AUXPOS  ','AUYPOS  ','AUBXXPOS','AUBXYPOS','AUBXSIZE','AUSBXPOS','AUSBYPOS','AUFOCUS ','AUFLTID ',
'AUFLTNM ','BOXMORA ','BOXMODEC','TCSTGT  ','OBJRA   ','OBJDEC  ','OBJPMRA ','OBJPMDEC','OBJEQUIN','OBSGEO-X',
'OBSGEO-Y','OBSGEO-Z','QCRDATE ','OBJECT  ','OBSERVER','IMAGETYP','OBS_MODE','IMAGECAT','PROPID  ','PROPTITL','PINAME  ',
'GROUPID ','BLOCKID ','SEQID   ','EXPTIME ','ROTATE  ','MIRROR_X','MIRROR_Y','AMPLMODE','DETMODE0','DETWIN1 ',
'DATAMIN ','DATAMAX ','CCDTEMP ','LN2TEMP ','P_DEWAR ','SHSTAT  ','DETXBIN ','DETYBIN ','NWINDOWS',
'TSAM    ','FPIX    ','VSHI    ','VSLO    ','VPHI    ','VPLO']

    def config_independent_frames(self):
        """
        Define frame types that are independent of the fully defined
        instrument configuration.

        Bias and dark frames are considered independent of a configuration,
        but the DATE-OBS keyword is used to assign each to the most-relevant
        configuration frame group. See
        :func:`~pypeit.metadata.PypeItMetaData.set_configurations`.

        Returns:
            :obj:`dict`: Dictionary where the keys are the frame types that
            are configuration independent and the values are the metadata
            keywords that can be used to assign the frames to a configuration
            group.
        """
        return {'standard': ['dispname' , 'decker', 'decker_secondary', 'binning' , 'datasec'] , 
                'bias': ['binning','datasec']  ,
                'arc' : ['decker','dispname', 'binning' , 'datasec','decker_secondary'], 
                'tilt' : ['decker','dispname', 'binning' , 'datasec','decker_secondary'],
                'pixelflat' : ['decker','decker_secondary','dispname', 'binning' , 'datasec'] , 
                'trace' : ['decker','dispname', 'binning' , 'datasec','decker_secondary'] , 
                'illumflat' : ['decker','dispname', 'binning' , 'datasec','decker_secondary'] }

    def check_frame_type(self, ftype, fitstbl, exprng=None):
        """
        Check for frames of the provided type.

        Args:
            ftype (:obj:`str`):
                Type of frame to check. Must be a valid frame type; see
                frame-type :ref:`frame_type_defs`.
            fitstbl (`astropy.table.Table`_):
                The table with the metadata for one or more frames to check.
            exprng (:obj:`list`, optional):
                Range in the allowed exposure time for a frame of type
                ``ftype``. See
                :func:`pypeit.core.framematch.check_frame_exptime`.

        Returns:
            `numpy.ndarray`_: Boolean array with the flags selecting the
            exposures in ``fitstbl`` that are ``ftype`` type frames.
        """
        good_exp = framematch.check_frame_exptime(fitstbl['exptime'], exprng)
        if ftype == 'science':
            return good_exp & (fitstbl['idname'] == 'OBJECT')
        if ftype == 'standard':
            return good_exp & ((fitstbl['idname'] == 'STD'))
        if ftype == 'bias':
            return good_exp & (fitstbl['idname'] == 'BIAS')  
        if ftype in ['pixelflat', 'trace', 'illumflat']:
            return good_exp & (fitstbl['idname'] == 'FLAT,LAMP')
        if ftype in ['pinhole', 'dark']:
            # Don't type pinhole or dark frames
            return np.zeros(len(fitstbl), dtype=bool)
        if ftype in ['arc','tilt']:
            return good_exp & (fitstbl['idname'] == 'WAVE,LAMP')
        log.warning('Cannot determine if frames are of type {0}.'.format(ftype))
        return np.zeros(len(fitstbl), dtype=bool)

    def config_specific_par(self, inp, inp_par=None):
        """
        Modify the PypeIt parameters to hard-wired values used for
        specific instrument configurations.

        Args:
            inp (:obj:`str`, :obj:`list`, `Path`_, `astropy.io.fits.Header`_, `astropy.table.Table`_):
                Input filename, an `astropy.io.fits.Header`_ object, or a list
                of `astropy.io.fits.Header`_ objects.  Or a row from the
                metadata table.
            inp_par (:class:`~pypeit.par.parset.ParSet`, optional):
                Parameter set used for the full run of PypeIt.  If None,
                use :func:`default_pypeit_par`.

        Returns:
            :class:`~pypeit.par.parset.ParSet`: The PypeIt parameter set
            adjusted for configuration specific parameter values.
        """
        # Start with instrument wide
        par = super().config_specific_par(inp, inp_par=inp_par)

        # Set slitspatnum to half of the detector spatial size based on datasec
        # Comment out?
        
        datasec = self.get_meta_value(inp, 'datasec')
        if datasec:
            datasec_str = datasec[0] if isinstance(datasec, np.ndarray) else datasec
            # Extract x-range (spatial dimension for specaxis=0)
            match = re.match(r'\[(\d+):(\d+),', datasec_str)
            if match:
                xstart = int(match.group(1))
                xend = int(match.group(2))
                # Calculate half of the window size (not absolute position)
                window_size = xend - xstart
                spatial_half = window_size // 2
                par['rdx']['slitspatnum'] = f'DET01:{spatial_half:04d}'
                log.info(f"Setting slitspatnum to DET01:{spatial_half:04d} (half of {window_size} pixel window)")

        # Wavelength calibrations
        # Use str() to handle cases where dispname is read as int from the metadata table
        dispname = str(self.get_meta_value(inp, 'dispname'))
        # Fallback for PypeIt output files (e.g. spec1d) that store the processed
        # 'DISPNAME' header key but not the original raw instrument key 'ALGRID'.
        # Without this, config_specific_par gets dispname='None' and all grism-specific
        # parameter overrides (nresln, hydrogen_mask_wid, …) are silently skipped.
        if dispname in ('None', ''):
            try:
                from astropy.io.fits import HDUList
                if isinstance(inp, HDUList):
                    dispname = str(inp[0].header.get('DISPNAME', ''))
            except Exception:
                pass
        if dispname == '401':
            par['calibrations']['wavelengths']['reid_arxiv'] = 'not_alfosc_grism3.fits'
        elif dispname == '402':
            par['calibrations']['wavelengths']['reid_arxiv'] = 'not_alfosc_grism4.fits'
            par['calibrations']['wavelengths']['lamps'] = ['HeI','NeI']
            par['sensfunc']['trim_std_pixs'] = [80, 80]
            par['sensfunc']['algorithm'] = 'UVIS'
            par['sensfunc']['UVIS']['resolution'] = 360 
            # Mask ±80 Å around each Balmer line so the bspline interpolates through them.
            par['sensfunc']['hydrogen_mask_wid'] = 80.0
            par['sensfunc']['mask_hydrogen_lines'] = True
            par['sensfunc']['samp_fact'] = 1.0
            par['sensfunc']['UVIS']['extinct_correct'] = True
            par['sensfunc']['UVIS']['nresln'] = 20
            par['sensfunc']['polyorder'] = 3
            par['sensfunc']['UVIS']['trans_thresh'] = 0.85
            # polycorrect=False: use bspline interpolation directly over the Balmer-line masked
            # regions, rather than replacing with a polynomial that can overshoot due to broad
            # absorption wings in hot standard stars.
            par['sensfunc']['UVIS']['polycorrect'] = False
            par['sensfunc']['extrap_blu'] = 0.1
            par['sensfunc']['extrap_red'] = 0.1
            # pixelflat_max_wave zeros the flat above 8000 Å to suppress CCD fringing in
            # the extracted spectra.
            par['calibrations']['flatfield']['pixelflat_max_wave'] = 8000.
            #par['fluxcalib']['use_archived_sens'] = True # TODO: provide sensfuncs for all grisms
        elif dispname == '403':
            par['calibrations']['wavelengths']['reid_arxiv'] = 'not_alfosc_grism5.fits'
        elif dispname == '405':
            par['calibrations']['wavelengths']['reid_arxiv'] = 'not_alfosc_grism7.fits'
        elif dispname == '406':
            par['calibrations']['wavelengths']['reid_arxiv'] = 'not_alfosc_grism8.fits'
        elif dispname == '408':
            par['calibrations']['wavelengths']['reid_arxiv'] = 'not_alfosc_grism10.fits'
        elif dispname == '409':
            par['calibrations']['wavelengths']['reid_arxiv'] = 'not_alfosc_grism11.fits'
        elif dispname == '415':
            par['calibrations']['wavelengths']['reid_arxiv'] = 'not_alfosc_grism17.fits'
        elif dispname == '417':
            par['calibrations']['wavelengths']['reid_arxiv'] = 'not_alfosc_grism18.fits'
            par['calibrations']['wavelengths']['lamps'] = ['HeI','NeI','ArI','ArII']
            par['sensfunc']['trim_std_pixs'] = [50, 50]            
        elif dispname == '418':
            par['calibrations']['wavelengths']['reid_arxiv'] = 'not_alfosc_grism19.fits'
            par['calibrations']['wavelengths']['lamps'] = ['HeI','NeI','ArI','ArII']
            par['sensfunc']['trim_std_pixs'] = [80, 80]
        elif dispname == '419':
            par['calibrations']['wavelengths']['reid_arxiv'] = 'not_alfosc_grism20.fits'
            par['sensfunc']['trim_std_pixs'] = [80, 80]    
            par['sensfunc']['algorithm'] = 'IR'        
        else:
            log.warning('not_alfosc.py: YOU NEED TO ADD IN THE WAVELENGTH SOLUTION FOR THIS GRISM')

        # Return
        return par


class NOTALFOSCSpectrographVert(NOTALFOSCSpectrograph):
    """
    Child to handle Vertical slits for NOT ALFOSC spectrograph
    """
    name = 'not_alfosc_vert'
    comment = 'Grisms 3, 4, 5, 7, 8, 10, 11, 17, 18, 19, 20. For vertical slits only'

    def get_detector_par(self, det, hdu=None):
        """
        Return metadata for the selected detector.

        Detector data from `here
        <http://www.not.iac.es/instruments/detectors/CCD14/>`__.

        .. warning::

            Many of the necessary detector parameters are read from the file
            header, meaning the ``hdu`` argument is effectively **required** for
            NOT/ALFOSC.  The optional use of ``hdu`` is only viable for
            automatically generated documentation.

        Args:
            det (:obj:`int`):
                1-indexed detector number.
            hdu (`astropy.io.fits.HDUList`_, optional):
                The open fits file with the raw image of interest.  If not
                provided, frame-dependent parameters are set to a default.

        Returns:
            :class:`~pypeit.images.detector_container.DetectorContainer`:
            Object with the detector metadata.
        """
        # http://www.not.iac.es/instruments/detectors/CCD14/

        if hdu is None:
            binning = '1,1'
            gain = None
            ronoise = None
        else:
            binning = self.get_meta_value(self.get_headarr(hdu), 'binning')
            gain = np.atleast_1d(hdu[1].header['GAIN'])  # e-/ADU
            ronoise = np.atleast_1d(hdu[1].header['RDNOISE'])  # e-

        # Detector 1
        detector_dict = dict(
            binning         = binning,
            det             = 1,
            dataext         = 1,
            specaxis        = 1, #Vertical slits have horizontal spectral dispersion
            specflip        = False,
            spatflip        = False,
            xgap            = 0.,
            ygap            = 0.,
            ysize           = 1.,
            platescale      = 0.2138,
            mincounts       = -1e10,
            darkcurr        = 1.3,      # e-/pix/hr
            saturation      = 700000.,  # ADU
            nonlinear       = 0.86,
            datasec         = np.atleast_1d('[{}:{},:]'.format(1, 2102)),  # Unbinned
            oscansec        = None,
            numamplifiers   = 1,
            gain            = gain,     # e-/ADU
            ronoise         = ronoise   # e-
        )

        # Return
        return detector_container.DetectorContainer(**detector_dict)


class NOTALFOSCSpectrographPol(NOTALFOSCSpectrograph):
    """
    Child to handle NOT ALFOSC spectro-polarimetry
    """
    ndet = 1
    name = 'not_alfosc_pol'
    comment = 'For use with the standard horizontal slits only. Grisms 3, 4, 5, 7, 8, 10, 11, 17, 18, 19, 20'
    telescope = telescopes.NOTTelescopePar()
    camera = 'ALFOSC'
    url = 'https://www.not.iac.es/instruments/alfosc/'
    header_name = 'ALFOSC_POL'
    supported = False
    pypeline = 'MultiSlit'  # Pol mode is reduced with MultiSlit, not Echelle

    def get_detector_par(self, det, hdu=None):
        """
        Return metadata for the selected detector.

        Detector data from `here
        <http://www.not.iac.es/instruments/detectors/CCD14/>`__.

        .. warning::

            Many of the necessary detector parameters are read from the file
            header, meaning the ``hdu`` argument is effectively **required** for
            NOT/ALFOSC.  The optional use of ``hdu`` is only viable for
            automatically generated documentation.

        Args:
            det (:obj:`int`):
                1-indexed detector number.
            hdu (`astropy.io.fits.HDUList`_, optional):
                The open fits file with the raw image of interest.  If not
                provided, frame-dependent parameters are set to a default.

        Returns:
            :class:`~pypeit.images.detector_container.DetectorContainer`:
            Object with the detector metadata.
        """
        # http://www.not.iac.es/instruments/detectors/CCD14/

        if hdu is None:
            binning = '1,1'
            datasec = None
            gain = None
            ronoise = None
        else:
            binning = self.get_meta_value(self.get_headarr(hdu), 'binning')
            datasec = np.atleast_1d(self.get_meta_value(self.get_headarr(hdu), 'datasec'))
            gain = np.atleast_1d(hdu[1].header['GAIN'])  # e-/ADU
            ronoise = np.atleast_1d(hdu[1].header['RDNOISE'])  # e-

        detector_dict = dict(
            binning         = binning,
            det             = 1,
            dataext         = 1,
            specaxis        = 0,
            specflip        = True,
            spatflip        = False,
            xgap            = 0.,
            ygap            = 0.,
            ysize           = 1.,
            platescale      = 0.2138,
            mincounts       = -1e10,
            darkcurr        = 1.3,      # e-/pix/hr
            saturation      = 700000.,  # ADU
            nonlinear       = 0.86,
            datasec         = np.atleast_1d('[:,:]'),  # datasec
            oscansec        = np.atleast_1d('[10:,2068:2092]'),  # Unbinned
            numamplifiers   = 1,
            gain            = gain,     # e-/ADU
            ronoise         = ronoise   # e-
        )

        return detector_container.DetectorContainer(**detector_dict)

    def config_specific_par(self, inp, inp_par=None):
        """
        Override to prevent ``slitspatnum`` from being set for specpol mode.

        The base class computes ``slitspatnum`` from the datasec midpoint to
        select a single longslit, but spectropolarimetry always requires both
        the ordinary and extraordinary ray slits to be extracted.
        """
        par = super().config_specific_par(inp, inp_par=inp_par)
        # Clear slitspatnum so both O and E slits are reduced
        par['rdx']['slitspatnum'] = None
        return par

    @classmethod
    def default_pypeit_par(cls):
        """
        Return the default parameters to use for this instrument.

        Returns:
            :class:`~pypeit.par.pypeitpar.PypeItPar`: Parameters required by
            all of PypeIt methods.
        """
        par = super().default_pypeit_par()
        
        par['calibrations']['slitedges']['det_buffer'] = 60
        par['calibrations']['slitedges']['fit_min_spec_length'] = 0.4
        par['calibrations']['slitedges']['bound_detector'] = False
        par['calibrations']['slitedges']['minimum_slit_length'] = 1.
        par['calibrations']['slitedges']['sync_predict'] = 'matched'
        par['calibrations']['slitedges']['sync_to_edge'] = False
        par['calibrations']['slitedges']['edge_thresh'] = 15
        par['calibrations']['slitedges']['fit_order'] = 4
        par['calibrations']['slitedges']['minimum_slit_gap'] = 0.05
        par['calibrations']['slitedges']['sync_center'] = 'gap'
        par['calibrations']['slitedges']['gap_offset'] = 2
        par['calibrations']['slitedges']['auto_pca'] = False

        par['reduce']['findobj']['maxnumber_sci'] = 2

        par['scienceframe']['process']['spat_flexure_correct'] = False

        return par
