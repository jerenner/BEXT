#
# Curv.py
#
# Algorithm for performing curvature-based single-electron vs. double-beta track
# discrimination.
#
# Notes:
# - the extremes returned by Paolina coincide with the beginning and end of
#   the main track hits (elements [0] and [-1] of the corresponding array)
#

import os
import numpy as np
import random as rd
import scipy.integrate as itg
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.colors as mpcol
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
from math import *
from scipy.interpolate import interp1d
from Centella.AAlgo import AAlgo
from Centella.physical_constants import *

class Curv(AAlgo):

    def __init__(self,param=False,level = 1,label="",**kargs):

        """
        
        Deal here with your parameters in param and kargs.
        If param is an instace of ParamManager, parameters
        will be set as algorithm parameters by AAlgo.
            
        """
        
            
        self.name='Curv'
        
        AAlgo.__init__(self,param,level,self.name,0,label,kargs)

        ### PARAMETERS
        # Track name
        try:
            self.trk_name = self.strings['trk_name']
            self.m.log(1, "Track name = {0}".format(self.trk_name))
        except KeyError:
            self.m.log(1, "WARNING!! Parameter: 'trk_name' not defined.")
            exit(0)

        # Output base
        try:
            self.trk_outdir = self.strings['trk_outdir']
            self.m.log(1, "Track out dir = {0}".format(self.trk_outdir))
        except KeyError:
            self.m.log(1, "WARNING!! Parameter: 'trk_outdir' not defined.")
            exit(0)

        # Data directory
        try:
            self.data_dir = self.strings['data_dir']
            self.m.log(1, "Data directory = {0}".format(self.data_dir))
        except KeyError:
            self.m.log(1, "WARNING!! Parameter: 'data_dir' not defined.")
            exit(0)
 
        # Pressure (bar)
        try:
            self.Pgas = self.doubles['Pgas']
            self.m.log(1, "Gas pressure = {0}".format(self.Pgas))
        except KeyError:
            self.m.log(1, "WARNING!! Parameter: 'Pgas' not defined.")
            exit(0)

        # Temperature (K)
        try:
            self.Tgas = self.doubles['Tgas']
            self.m.log(1, "Gas temperature = {0}".format(self.Tgas))
        except KeyError:
            self.m.log(1, "WARNING!! Parameter: 'Tgas' not defined.")
            exit(0)

        # Magnetic field (T)
        try:
            self.Bfield = self.doubles['Bfield']
            self.m.log(1, "Magnetic field = {0}".format(self.Bfield))
        except KeyError:
            self.m.log(1, "WARNING!! Parameter: 'Bfield' not defined.")
            exit(0)

        # Flag for using voxels (if = 0, use MC hits)
        try:
            self.use_voxels = self.ints['use_voxels']
            self.m.log(1, "Use voxels = {0}".format(self.use_voxels))
        except KeyError:
            self.m.log(1, "WARNING!! Parameter: 'use_voxels' not defined.")
            exit(0)

        # Flag for determining start/end of track using blob energies.
        try:
            self.blob_ordering = self.ints['blob_ordering']
            self.m.log(1, "Use blob ordering = {0}".format(self.blob_ordering))
        except KeyError:
            self.m.log(1, "WARNING!! Parameter: 'blob_ordering' not defined.")
            exit(0)

        # Initial kinetic energy (MeV)
        try:
            self.KEinit = self.doubles['KEinit']
            self.m.log(1, "Initial kinetic energy = {0}".format(self.KEinit))
        except KeyError:
            self.m.log(1, "WARNING!! Parameter: 'KEinit' not defined.")
            exit(0)

        # Signal profile name
        try:
            self.prof_sname = self.strings['prof_sname']
            self.m.log(1, "Signal profile name = {0}".format(self.prof_sname))
        except KeyError:
            self.m.log(1, "WARNING!! Parameter: 'prof_sname' not defined.")
            exit(0)

        # Background profile name
        try:
            self.prof_bname = self.strings['prof_bname']
            self.m.log(1, "Background profile name = {0}".format(self.prof_bname))
        except KeyError:
            self.m.log(1, "WARNING!! Parameter: 'prof_bname' not defined.")
            exit(0)

        # Profile generation boolean
        try:
            self.par_prof_gen = self.ints['prof_gen']
            self.prof_gen = False
            if(self.par_prof_gen == 1):
                self.prof_gen = True
            self.m.log(1, "Profile generation = {0}".format(self.prof_gen))
        except KeyError:
            self.m.log(1, "WARNING!! Parameter: 'prof_gen' not defined.")
            exit(0)

        # Profile comparison boolean
        try:
            self.par_prof_cf = self.ints['prof_cf']
            self.prof_cf = False
            if(self.par_prof_cf == 1):
                self.prof_cf = True
            self.m.log(1, "Profile generation = {0}".format(self.prof_cf))
        except KeyError:
            self.m.log(1, "WARNING!! Parameter: 'prof_cf' not defined.")
            exit(0)

        # Filter draw boolean
        try:
            self.par_plt_drawfilter = self.ints['plt_drawfilter']
            self.plt_drawfilter = False
            if(self.par_plt_drawfilter == 1):
                self.plt_drawfilter = True
            self.m.log(1, "Draw plot of filter = {0}".format(self.plt_drawfilter))
        except KeyError:
            self.m.log(1, "WARNING!! Parameter: 'plt_drawfilter' not defined.")
            exit(0)

        # Track draw boolean
        try:
            self.par_plt_drawtrk = self.ints['plt_drawtrk']
            self.plt_drawtrk = False
            if(self.par_plt_drawtrk == 1):
                self.plt_drawtrk = True
            self.m.log(1, "Draw plot of track = {0}".format(self.plt_drawtrk))
        except KeyError:
            self.m.log(1, "WARNING!! Parameter: 'plt_drawtrk' not defined.")
            exit(0)

        # Output means boolean
        try:
            self.par_output_means = self.ints['output_means']
            self.output_means = False
            if(self.par_output_means == 1):
                self.output_means = True
            self.m.log(1, "Output means to a file = {0}".format(self.output_means))
        except KeyError:
            self.m.log(1, "WARNING!! Parameter: 'output_means' not defined.")
            exit(0)

        # Fixed filter boolean
        try:
            self.par_fcbar_fixed = self.ints['fcbar_fixed']
            self.fcbar_fixed = False
            if(self.par_fcbar_fixed == 1):
                self.fcbar_fixed = True
            self.m.log(1, "Use fixed filter = {0}".format(self.fcbar_fixed))
        except KeyError:
            self.m.log(1, "WARNING!! Parameter: 'fcbar_fixed' not defined.")
            exit(0)

        # Fixed filter fcbar
        try:
            self.fcbar_fix = self.doubles['fcbar_fix']
            self.m.log(1, "Fixed filter fcbar = {0}".format(self.fcbar_fix))
        except KeyError:
            self.m.log(1, "WARNING!! Parameter: 'fcbar_fixed' not defined.")
            exit(0)

        # Gas type
        try:
            self.gas_type = self.strings['gas_type']
            self.m.log(1, "Gas type = {0}".format(self.gas_type))
        except KeyError:
            self.m.log(1, "WARNING!! Parameter: 'gas_type' not defined.")
            exit(0)

        # Number of bins in sign profiles
        try:
            self.nbins_kon = self.ints['nbins_kon']
            self.m.log(1, "nbins_kon = {0}".format(self.nbins_kon))
        except KeyError:
            self.m.log(1, "WARNING!! Parameter: 'nbins_kon' not defined.")
            exit(0)

        # Minimum k/N for profile evaluation
        try:
            self.kon_min = self.doubles['kon_min']
            self.m.log(1, "k/N min = {0}".format(self.kon_min))
        except KeyError:
            self.m.log(1, "WARNING!! Parameter: 'kon_min' not defined.")
            exit(0)

        # Maximum k/N for profile evaluation
        try:
            self.kon_max = self.doubles['kon_max']
            self.m.log(1, "k/N max = {0}".format(self.kon_max))
        except KeyError:
            self.m.log(1, "WARNING!! Parameter: 'kon_max' not defined.")
            exit(0)

        # Blob radius
        try:
            self.blob_radius = self.doubles['blob_radius']
            self.m.log(1, "blob radius = {0}".format(self.blob_radius))
        except KeyError:
            self.m.log(1, "WARNING!! Parameter: 'blob_radius' not defined.")
            exit(0)

        # Amount of x-y smearing (sigma)
        try:
            self.xy_smearing = self.doubles['xy_smearing']
            self.m.log(1, "sparse width = {0}".format(self.xy_smearing))
        except KeyError:
            self.m.log(1, "WARNING!! Parameter: 'xy_smearing' not defined.")
            exit(0)

        # Sparse width
        try:
            self.sparse_width = self.ints['sparse_width']
            self.m.log(1, "sparse width = {0}".format(self.sparse_width))
        except KeyError:
            self.m.log(1, "WARNING!! Parameter: 'sparse_width' not defined.")
            exit(0)

        # Max. distance between hits in a connected track
        try:
            self.ctrack_dist = self.doubles['ctrack_dist']
            self.m.log(1, "connected track max hit distance ctrack_dist = {0}".format(self.ctrack_dist))
        except KeyError:
            self.m.log(1, "WARNING!! Parameter: 'ctrack_dist' not defined.")
            exit(0)


        # Set up the relevant paths.
        self.trk_base = "{0}/{1}".format(self.trk_outdir,self.trk_name)
        self.plt_base = "{0}/plt_curv".format(self.trk_base)
        self.prof_file = "{0}/plt_curv/prof_{1}.dat".format(self.trk_base,self.trk_name)
        self.prof_sfile = "{0}/{1}/plt_curv/prof_{2}.dat".format(self.trk_outdir,self.prof_sname,self.prof_sname)
        self.prof_bfile = "{0}/{1}/plt_curv/prof_{2}.dat".format(self.trk_outdir,self.prof_bname,self.prof_bname)

        # Create the output and plot directories.
        if(not os.path.isdir(self.trk_base)):
            os.mkdir(self.trk_base)
        if(not os.path.isdir(self.plt_base)):
            os.mkdir(self.plt_base)
            print "Creating plot directory {0}...".format(self.plt_base)

        # Define the phyiscal constants.
        self.pc_rho0 = 2.6867774e19   # density of ideal gas at T=0C, P=1 atm in cm^(-3)
        self.pc_m_Xe = 131.293        # mass of xenon in amu
        self.pc_m_sef6 = 192.96       # mass of SeF6 in amu
        self.pc_NA = NA               # Avogadro constant
        self.pc_eC = e_SI             # electron charge in C
        self.pc_me = 9.10938215e-31   # electron mass in kg
        self.pc_clight = 2.99792458e8 # speed of light in m/s
        print "Speed of light is {0} m/s".format(c_light)

        # Misc. options
        self.grdcol = 0.98

    # Remove outliers in a list.
    def remove_outliers(self,lst):

        #print "\n --------------- Removing outliers --------------------";

        nloop = 0;
        mui = np.mean(lst); sgi = np.std(lst);
        mu = mui; sg = sgi;
        while(nloop == 0 or sg < 0.99*sgi):

            nn = 0;
            while(nn < len(lst)):

                val = lst[nn];
                if(abs(val-mu) > 5*sg):
                    if(nn == 0 or nn == len(lst)-1):
                        lst[nn] = 0;
                    else:
                        lst[nn] = (lst[nn-1] + lst[nn+1])/2.;

                nn += 1;

            mui = mu; sgi = sg;
            mu = np.mean(lst); sg = np.std(lst);
            #print "Mean lst is {0}, stdev lst is {1}".format(mu,sg);

            nloop += 1;

    # Calculate the time required to create the track.
    def calc_track_time(self):

        # Prepare the stopping power table.
        if(self.gas_type == "sef6"):
            xesp_tbl = np.loadtxt("{0}/sef6_estopping_power_NIST.dat".format(self.data_dir))
            rho = 0.007887*self.Pgas #pc_rho0*(Pgas/(Tgas/273.15))*(pc_m_sef6/pc_NA);
            self.m.log("Gas type is SeF6")
        else:
            xesp_tbl = np.loadtxt("{0}/xe_estopping_power_NIST.dat".format(self.data_dir))

            #if (Pgas > 9.5 and Pgas < 10.5):
            #    rho = 0.055587;
            #elif (Pgas > 14.5 and Pgas < 15.5):
            #    rho = 0.08595;
            #elif (Pgas > 19.5 and Pgas < 20.5):
            #    rho = 0.1184;
            #else:
            #    rho = 0.0055*Pgas;
            rho = self.pc_rho0*(self.Pgas/(self.Tgas/273.15))*(self.pc_m_Xe/self.pc_NA)
            if(self.gas_type != "xe"):
                self.m.log("**** NOTE: gas type defaulting to xenon")

        self.m.log("Calculated density for gas {0} is {1}".format(self.gas_type,rho))
        e_vals = xesp_tbl[:,0]
        dEdx_vals = xesp_tbl[:,1]
        e_vals = np.insert(e_vals,0,0.0)
        dEdx_vals = np.insert(dEdx_vals,0,dEdx_vals[0])
        xesp = interp1d(e_vals,dEdx_vals*rho,kind='cubic')

        # Compute the integral.
        tval = itg.quad(lambda x: (sqrt((x+0.511)**2-0.511**2)/(x+0.511)*xesp(x))**(-1),0.00001,self.KEinit,limit=500)[0]
        tval = tval/(100*self.pc_clight)

        xval = itg.quad(lambda x: (xesp(x))**(-1),0.00001,self.KEinit,limit=500)[0]
        self.m.log("Average track length = {0} cm".format(xval))

        return tval;

    def initialize(self):

        """

        You can use here:

            self.hman     -----> histoManager instance
            self.tman     -----> treeManager instance
            self.logman   -----> logManager instance
            self.autoDoc  -----> latex_gen instance
        
        """
        
        self.m.log(1,'+++Init method of algo_example algorithm+++')

        # Calculate the average track time.
        self.Ttrack = self.calc_track_time()
        self.m.log(" *** Average time to create track is {0}".format(self.Ttrack))

        # Calculate the cyclotron frequency.
        self.wcyc = -1.0*(self.pc_eC/self.pc_me)*self.Bfield
      
        # Create the lists to be filled with information from each event.
        self.l_evtnum = []
        self.l_eblob1 = []
        self.l_eblob2 = []
        self.l_scurv_mean = []
        self.prof_sgn_kon = []
        self.prof_sgn_vals = []
        self.l_chi2S = []
        self.l_chi2B = []
        self.l_nhits = []
        self.l_bflip = []

        # Initialize the flipped tracks counter to 0.
        self.flip_trk = 0
 
        # Read in and interpolate the profiles if they exist and we are not in profile generation mode.
        if(not self.prof_gen and os.path.isfile(self.prof_sfile) and os.path.isfile(self.prof_bfile)):

            # Read in and interpolate the profiles.  In each case, place the endpoints (k/N = 0 and k/N = 1)
            #  by extrapolating linearly using the two closest points.
            sproftbl = np.loadtxt(self.prof_sfile)
            l_sprof_x = []; l_sprof_y = []; l_sprofs_y = []
            l_sprof_x.append(0.0)
            l_sprof_y.append(sproftbl[0,1] - ((sproftbl[1,1]-sproftbl[0,1])/(sproftbl[1,0]-sproftbl[0,0]))*sproftbl[0,0])
            l_sprofs_y.append(sproftbl[0,2] - ((sproftbl[1,2]-sproftbl[0,2])/(sproftbl[1,0]-sproftbl[0,0]))*sproftbl[0,0])
            for xsprof,ysprof,ysprofs in zip(sproftbl[:,0],sproftbl[:,1],sproftbl[:,2]):
                l_sprof_x.append(xsprof)
                l_sprof_y.append(ysprof)
                l_sprofs_y.append(ysprofs)
            l_sprof_x.append(1.0)
            l_sprof_y.append(sproftbl[-1,1] + ((sproftbl[-2,1]-sproftbl[-1,1])/(sproftbl[-2,0]-sproftbl[-1,0]))*(1.0 - sproftbl[-1,0]))
            l_sprofs_y.append(sproftbl[-1,2] + ((sproftbl[-2,2]-sproftbl[-1,2])/(sproftbl[-2,0]-sproftbl[-1,0]))*(1.0 - sproftbl[-1,0]))

            self.sprof = interp1d(l_sprof_x,l_sprof_y,kind='cubic')
            self.sprof_sigma = interp1d(l_sprof_x,l_sprofs_y,kind='cubic')

            bproftbl = np.loadtxt(self.prof_bfile)
            l_bprof_x = []; l_bprof_y = []; l_bprofs_y = []
            l_bprof_x.append(0.0)
            l_bprof_y.append(bproftbl[0,1] - ((bproftbl[1,1]-bproftbl[0,1])/(bproftbl[1,0]-bproftbl[0,0]))*bproftbl[0,0])
            l_bprofs_y.append(bproftbl[0,2] - ((bproftbl[1,2]-bproftbl[0,2])/(bproftbl[1,0]-bproftbl[0,0]))*bproftbl[0,0])
            for xsprof,ysprof,ysprofs in zip(bproftbl[:,0],bproftbl[:,1],bproftbl[:,2]):
                l_bprof_x.append(xsprof)
                l_bprof_y.append(ysprof)
                l_bprofs_y.append(ysprofs)
            l_bprof_x.append(1.0)
            l_bprof_y.append(bproftbl[-1,1] + ((bproftbl[-2,1]-bproftbl[-1,1])/(bproftbl[-2,0]-bproftbl[-1,0]))*(1.0 - bproftbl[-1,0]))
            l_bprofs_y.append(bproftbl[-1,2] + ((bproftbl[-2,2]-bproftbl[-1,2])/(bproftbl[-2,0]-bproftbl[-1,0]))*(1.0 - bproftbl[-1,0]))

            self.bprof = interp1d(l_bprof_x,l_bprof_y,kind='cubic')
            self.bprof_sigma = interp1d(l_bprof_x,l_bprofs_y,kind='cubic')

        else:
            self.m.log("Not performing profile comparison, as this cannot be done simultaneously with profile generation, or profile {0} or {1} does not exist.".format(self.prof_sfile,self.prof_bfile))
            self.prof_cf = False
 
        return

    def execute(self,event=""):

        """
        
        You can also use here:

            self.event ----> current event from the input data 
            
        """

        # Get the event number.
        trk_num = self.event.GetEventID()
        self.l_evtnum.append(trk_num)

        # Get all tracks from the voxelization step. 
        all_trks = self.event.GetTracks()
        if(len(all_trks) != 1):
            self.m.log("ERROR: Event has more than 1 track.")
            exit(0)

        # Get the single track for this event.
        main_trk = all_trks[0]

        # Get the order of the hits.
        hOrder = main_trk.fetch_ivstore("MainPathHits")
        #for iord in hOrder: print "{0}, ".format(iord);

        # Declare the final track.
        trk_xM_0 = []; trk_yM_0 = []; trk_zM_0 = []
        trk_eM_0 = []; trk_nM_0 = []
        ftrack = []

        # Declare the extremes.
        e1x = 0; e1y = 0; e1z = 0
        e2x = 0; e2y = 0; e2z = 0

        # Use the voxelized hits if the option is set.
        if(self.use_voxels == 1): 

            # Get the list of hits.
            hList = self.event.GetHits()

            # Get the single track for this event.
            main_trk = all_trks[0]

            # Get the order of the hits determined by Paolina.
            hOrder = main_trk.fetch_ivstore("MainPathHits")
            #for iord in hOrder: print "{0}, ".format(iord);

            # Get the computed distances.
            distExtFirst = main_trk.fetch_dvstore("DistExtFirst")
            distExtSecond = main_trk.fetch_dvstore("DistExtSecond")

            # Fill the final track on which the calculation is to be performed.
            for ihit in hOrder:
                ftrack.append(hList[ihit])

            # Get the extremes.
            e1, e2 = main_trk.GetExtremes()
            e1x = e1.GetPosition().x(); e1y = e1.GetPosition().y(); e1z = e1.GetPosition().z()
            e2x = e2.GetPosition().x(); e2y = e2.GetPosition().y(); e2z = e2.GetPosition().z()

            # Fill the coordinate arrays with the main track hits.
            nhit = 0
            trk_hitID = []
            for fhit in ftrack:
                trk_xM_0.append(fhit.GetPosition().x())
                trk_yM_0.append(fhit.GetPosition().y())
                trk_zM_0.append(fhit.GetPosition().z())
                trk_eM_0.append(fhit.GetAmplitude())
                trk_nM_0.append(nhit)
                nhit = nhit + 1

            # Determine two blob energies.
            eblob1 = 0.; eblob2 = 0.
            for hhit in hList:

                d1 = distExtFirst[hhit.GetID()]  # sqrt((xh-e1x)**2 + (yh-e1y)**2 + (zh-e1z)**2)
                d2 = distExtSecond[hhit.GetID()] # sqrt((xh-e2x)**2 + (yh-e2y)**2 + (zh-e2z)**2)
                eh = hhit.GetAmplitude()

                if(d1 < self.blob_radius):
                    eblob1 += eh
                if(d2 < self.blob_radius):
                    eblob2 += eh

            # Ensure the extremes are the ends of ftrack
            print "Extreme 1 ID = {0}, beginning of ftrack ID = {1}".format(e1.GetID(),ftrack[0].GetID())
            print "Extreme 2 ID = {0}, end of ftrack ID = {1}".format(e2.GetID(),ftrack[-1].GetID())
            

        else:

            # Get the list of all MC hits.
            hList = self.event.GetMCHits()

            # Get the MC tracks.
            mcTracks = self.event.GetMCTracks()

            # Find the energy of the most energetic and second-most energetic tracks.
            ifirst = -1; isecond = -1
            efirst = -1.; esecond = -1.
            nmctrk = 0

            for mctrk in mcTracks:
                trk_en = mctrk.GetHitsEnergy()
                print "-- MC track with energy {0}".format(trk_en)
                if(trk_en > efirst):
                    isecond = ifirst; esecond = efirst
                    ifirst = nmctrk; efirst = trk_en
                elif(trk_en > esecond):
                    isecond = nmctrk; esecond = trk_en
                nmctrk += 1
            if(nmctrk == 0):
                print "(Track {0}): ERROR - no MC tracks are present.".format(trk_num)
            if(nmctrk > 2):
                print "(Track {0}): WARNING - more than 2 MC tracks are present.".format(trk_num)

            #print "Found most energetic index {0}, E1 = {1}; second-most energetic index {2}, E2 = {3}".format(ifirst,efirst,isecond,esecond)

            ftrk = mcTracks[ifirst]
            ft_hits = ftrk.GetHits()

            # Add the least energetic electron track first (in reverse), if there is one.
            mainHits = [] 
            if(nmctrk > 1):
                strk = mcTracks[isecond]
                st_hits = strk.GetHits()
                for ihit in range(len(st_hits)):
                    mainHits.append(st_hits[len(st_hits)-1-ihit])
            #else:
            #    print "(Track {0}): WARNING - only one energetic electron track".format(trk_num)

            # Add the most energetic electron track.
            for ihit in range(len(ft_hits)):
                mainHits.append(ft_hits[ihit])
            
            # Determine the longest connected track.
            ctracks = []; ctrack_t = []
            ihit = 0; itrk = 0
            ctrack_t.append(mainHits[0])
            while(ihit < len(mainHits)-1):
                 h1 = mainHits[ihit]
                 h2 = mainHits[ihit+1]

                 # Get the distance between the consecutive hits.
                 x1 = h1.GetPosition().x(); y1 = h1.GetPosition().y(); z1 = h1.GetPosition().z()
                 x2 = h2.GetPosition().x(); y2 = h2.GetPosition().y(); z2 = h2.GetPosition().z()
                 hdist = sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
                 #print "--- Distance between consecutive hits is {0} mm".format(hdist)
      
                 # Add the next hit to the current connected track if it is connected.
                 if(hdist < self.ctrack_dist):
                     ctrack_t.append(h2);
                 # Otherwise, start a new connected track.
                 else:
                     ctracks.append(ctrack_t)
                     ctrack_t = []
                     ctrack_t.append(h2)
                 
                 ihit += 1

            # Add the final connected track to the list.
            ctracks.append(ctrack_t)

            # Get the list of hits containing the longest connected track.
            ilct = -1; llct = -1
            itrk = 0
            for ctrk in ctracks:
                ltrk = len(ctrk)
                if(ltrk > llct):
                    ilct = itrk
                    llct = ltrk
                itrk += 1
            ftrack = ctracks[ilct]

            print "(Track {0}): -- Found {1} connected tracks, of which longest has {2} hits".format(trk_num,len(ctracks),len(ctracks[ilct]))
            #for ctrk in ctracks:
            #    print "-> Track with {0} hits".format(len(ctrk))

            # Fill the coordinate arrays with the main track hits.
            trk_xM_true = []; trk_yM_true = []; trk_zM_true = []
            trk_nM_true = []; trk_eM_true = []
            nhit = 0
            for fhit in ftrack:
                trk_xM_true.append(fhit.GetPosition().x())
                trk_yM_true.append(fhit.GetPosition().y())
                trk_zM_true.append(fhit.GetPosition().z())
                trk_eM_true.append(fhit.GetAmplitude())
                trk_nM_true.append(nhit)
                nhit = nhit + 1

            # Apply sparsing if specified.
            trk_xM_sp = []; trk_yM_sp = []; trk_zM_sp = []
            trk_nM_sp = []; trk_eM_sp = []
            if(self.sparse_width > 1):

                ihit = 0; nhit = 0
                while(ihit < len(trk_xM_true)):

                    sp = ihit

                    # Create an effective (sparsed) hit.
                    xeff = 0.; yeff = 0.; zeff = 0.; eeff = 0.
                    while(sp < len(trk_xM_true) and sp < (ihit + self.sparse_width)):
                        ehit = trk_eM_true[sp]
                        xeff += trk_xM_true[sp]*ehit
                        yeff += trk_yM_true[sp]*ehit
                        zeff += trk_zM_true[sp]*ehit
                        eeff += ehit
                        sp += 1
                    ihit += self.sparse_width

                    # Finish calculating effective coordinates.
                    xeff /= eeff
                    yeff /= eeff
                    zeff /= eeff

                    # Record the values in lists for the sparsed track.
                    trk_xM_sp.append(xeff)
                    trk_yM_sp.append(yeff)
                    trk_zM_sp.append(zeff)
                    trk_eM_sp.append(eeff)
                    trk_nM_sp.append(nhit)

                    nhit += 1
            else:
                trk_xM_sp = trk_xM_true
                trk_yM_sp = trk_yM_true
                trk_zM_sp = trk_zM_true
                trk_eM_sp = trk_eM_true
                trk_nM_sp = trk_nM_true

            # Apply smearing if specified.
            if(self.xy_smearing > 0):

                for xm,ym,zm,em,nm in zip(trk_xM_sp,trk_yM_sp,trk_zM_sp,trk_eM_sp,trk_nM_sp):
                    xsm = rd.gauss(xm,self.xy_smearing)
                    ysm = rd.gauss(ym,self.xy_smearing)
                    trk_xM_0.append(xsm)
                    trk_yM_0.append(ysm)
                    trk_zM_0.append(zm)
                    trk_eM_0.append(em)
                    trk_nM_0.append(nm)
            else:
                trk_xM_0 = trk_xM_sp
                trk_yM_0 = trk_yM_sp
                trk_zM_0 = trk_zM_sp
                trk_eM_0 = trk_eM_sp
                trk_nM_0 = trk_nM_sp 

            # Set the extremes.
            e1x = trk_xM_0[0]; e1y = trk_yM_0[0]; e1z = trk_zM_0[0]
            e2x = trk_xM_0[-1]; e2y = trk_yM_0[-1]; e2z = trk_zM_0[-1]

            # Determine two blob energies.
            eblob1 = 0.; eblob2 = 0.
            #for hhit in hList:
            for xh,yh,zh,eh in zip(trk_xM_0,trk_yM_0,trk_zM_0,trk_eM_0):

                d1 = sqrt((xh-e1x)**2 + (yh-e1y)**2 + (zh-e1z)**2)
                d2 = sqrt((xh-e2x)**2 + (yh-e2y)**2 + (zh-e2z)**2)
           
                if(d1 < self.blob_radius):
                    eblob1 += eh
                if(d2 < self.blob_radius):
                    eblob2 += eh

        # Record the number of hits used in this analysis.
        self.l_nhits.append(len(trk_xM_0))

        # Add the more energetic blob as blob1.
        if(eblob1 > eblob2):
            self.l_eblob1.append(eblob1)
            self.l_eblob2.append(eblob2)
        else:
            self.l_eblob1.append(eblob2)
            self.l_eblob2.append(eblob1)

        # Determine if the blobs overlap.
        dblobs = sqrt((e1x-e2x)**2 + (e1y-e2y)**2 + (e1z-e2z)**2)
        if(dblobs < self.blob_radius):
            print "(Track {0}) WARNING: blobs overlap for track!".format(trk_num)

        # Determine which extreme corresponds to which blob.
        blob1 = 0; blob2 = 0
        if(sqrt((e1x-trk_xM_0[0])**2 + (e1y-trk_yM_0[0])**2 + (e1z-trk_zM_0[0])**2) < 0.001):
            blob1 = 1
        elif(sqrt((e1x-trk_xM_0[-1])**2 + (e1y-trk_yM_0[-1])**2 + (e1z-trk_zM_0[-1])**2) < 0.001):
            blob2 = 1
        if(sqrt((e2x-trk_xM_0[0])**2 + (e2y-trk_yM_0[0])**2 + (e2z-trk_zM_0[0])**2) < 0.001):
            blob1 = 2
        elif(sqrt((e2x-trk_xM_0[-1])**2 + (e2y-trk_yM_0[-1])**2 + (e2z-trk_zM_0[-1])**2) < 0.001):
            blob2 = 2

        # Print a warning if we have an unusual blob identification.
        if(blob1 != 1 or blob2 != 2):
            print "(Track {0}) WARNING: Blob1 = {1} and blob2 = {2}".format(trk_num,blob1,blob2);

        # Print an error if we have an invalid blob identification.
        if(blob1 == 0):
            self.m.log("***ERROR (Track {0}) Blob1 never assigned to an extreme".format(trk_num))
        if(blob2 == 0):
            self.m.log("***ERROR (Track {0}) Blob2 never assigned to an extreme".format(trk_num))
        if(blob1 == blob2):
            self.m.log("***ERROR (Track {0}) Blob1 and blob2 assigned to the same extreme".format(trk_num))

        # Decide on whether we will need to flip the track.
        if(self.blob_ordering and ((blob1 == 1 and blob2 == 2 and eblob2 < eblob1) or (blob1 == 2 and blob2 == 1 and eblob1 < eblob2))):
            trk_xM_0.reverse()
            trk_yM_0.reverse()
            trk_zM_0.reverse()
            trk_eM_0.reverse()
            print "(Track {0}) Flipping track due to blob at extreme 2 containing less energy than blob at extreme 1.".format(trk_num)
            self.flip_trk += 1
            self.l_bflip.append(1)
        else:
            self.l_bflip.append(0)

        #print "Extreme 2 at ({0},{1},{2})".format(e2x,e2y,e2z)

        # Design the LPF.
        #ux0 = trk_ux[0]; uy0 = trk_uy[0]; print "ux = {0}, uy = {1}".format(ux0,uy0);
        fsamp = len(trk_xM_0)/self.Ttrack
        if(self.fcbar_fixed):
            fcbar = self.fcbar_fix
        else:
            fcbar = abs(self.wcyc)/(2*pi*fsamp)
        #rcyc = sqrt((KEinit+0.511)**2-0.511**2)/(KEinit+0.511)*sqrt(ux0**2 + uy0**2)*pc_clight*100/abs(wcyc);
        self.m.log("Sampling frequency = {0} samp/s, cyclotron freq = {1} cyc/s, fcbar = {2}, should see {3} cycles".format(fsamp,self.wcyc/(2*pi),fcbar,abs(self.Ttrack*self.wcyc/(2*pi))))
        
        # -------------------------------------------------------------------------
        # FIR filter from: http://wiki.scipy.org/Cookbook/FIRFilter
        # The Nyquist rate of the signal.
        nyq_rate = fsamp / 2.0
    
        # The desired width of the transition from pass to stop,
        width = 0.2
    
        # The desired attenuation in the stop band, in dB.
        ripple_db = 40.0

        # Compute the order and Kaiser parameter for the FIR filter.
        N, beta = signal.kaiserord(ripple_db, width)

        # The cutoff frequency of the filter.
        cutoff_freq = 1.2 * fcbar * fsamp

        # Use firwin with a Kaiser window to create a lowpass FIR filter.
        taps = signal.firwin(N, cutoff_freq, window=('kaiser', beta), nyq=nyq_rate)
        #print taps

        # Use lfilter to filter x with the FIR filter.
        #filtered_x = lfilter(taps, 1.0, x);

        # Compute the filter delay.
        fdelay = int(N/2)
        # -------------------------------------------------------------------------
    
        # Apply the filters.
        #if(apply_lpf):
        trk_xM = signal.lfilter(taps,1.0,trk_xM_0)
        trk_yM = signal.lfilter(taps,1.0,trk_yM_0)
        trk_zM = signal.lfilter(taps,1.0,trk_zM_0)
        #print "Applying filter with coefficients:";
        #print taps;
        
        # Correct for the FIR delay.
        #print "Applying delay of {0} samples".format(fdelay);
        #print trk_xM;
        trk_xM_f = np.roll(trk_xM,-1*fdelay); trk_xM = trk_xM_f[0:-fdelay]
        trk_yM_f = np.roll(trk_yM,-1*fdelay); trk_yM = trk_yM_f[0:-fdelay]
        trk_zM_f = np.roll(trk_zM,-1*fdelay); trk_zM = trk_zM_f[0:-fdelay]
        trk_eM_f = np.roll(trk_eM_0,-1*fdelay); trk_eM = trk_eM_f[0:-fdelay]
        #print trk_xM;

        # Compute the derivatives.
        dxdn = np.diff(trk_xM)
        dydn = np.diff(trk_yM)
        dzdn = np.diff(trk_zM)
    
        d2xdn2 = np.append(np.diff(dxdn),dxdn[-2]-dxdn[-1])
        d2ydn2 = np.append(np.diff(dydn),dydn[-2]-dydn[-1])
        d2zdn2 = np.append(np.diff(dzdn),dzdn[-2]-dzdn[-1])
    
        mu_dxdn = np.mean(dxdn); sg_dxdn = np.std(dxdn)
        mu_dydn = np.mean(dydn); sg_dydn = np.std(dydn)
        mu_dzdn = np.mean(dzdn); sg_dzdn = np.std(dzdn)
        #print "Mean dxdn is {0}, stdev dxdn is {1}".format(mu_dxdn,sg_dxdn);
    
        # Remove outliers.
        self.remove_outliers(dxdn)
        self.remove_outliers(dydn)
        self.remove_outliers(dzdn)

        self.remove_outliers(d2xdn2)
        self.remove_outliers(d2ydn2)
        self.remove_outliers(d2zdn2)
    
        # Calculate first derivatives.
        zz = []; nn = []; dxdz = []; dydz = []
        n = 0
        #print "z length is {0}, derivative lengths are {1}".format(len(trk_zM[0:-1]),len(dxdn));
        for zval,ddx,ddy,ddz in zip(trk_zM[0:-1],dxdn,dydn,dzdn):
            nn.append(n)
            zz.append(zval)
            if(ddz == 0):
                dxdz.append(0.)
                dydz.append(0.)
            else:
                dxdz.append(ddx/ddz)
                dydz.append(ddy/ddz)
            n += 1
        
        self.remove_outliers(dxdz)
        self.remove_outliers(dydz)
        #print "Len = {0}, {1}".format(len(dxdn),len(d2xdn2));
    
        # Calculate the difference in derivatives.
        diff_dxdy = []
        for dx,dy in zip(dxdz,dydz):
            diff_dxdy.append(dx-dy)
    
        # Calculate second derivatives.
        d2xdz2 = []; d2ydz2 = []; rcurv = []; scurv = []; sscurv = []
        for dxz,dyz,ddz,d2dx,d2dy,d2dz in zip(dxdz,dydz,dzdn,d2xdn2,d2ydn2,d2zdn2):
    
            # 2nd derivatives
            if(ddz == 0):
                d2xz = 0.
                d2yz = 0.
            else:
                d2xz = (d2dx - d2dz*dxz)/(ddz**2)
                d2yz = (d2dy - d2dz*dyz)/(ddz**2)
            d2xdz2.append(d2xz)
            d2ydz2.append(d2yz)
    
        # Remove outliers from the second derivatives.
        self.remove_outliers(d2xdz2)
        self.remove_outliers(d2ydz2)
    
        for dzn,dxz,dyz,d2xz,d2yz in zip(dzdn,dxdz,dydz,d2xdz2,d2ydz2):
    
            # Radius of curvature and signed curvature.
            if(dxz*d2yz - dyz*d2xz == 0):
                sc = 0.
                rc = 0.
            else:
                sc = (dxz*d2yz - dyz*d2xz)/(dxz**2 + dyz**2)**1.5
                if(dzn < 0): sc *= -1  # correct for the direction of travel
                rc = abs(1.0/sc)
            rcurv.append(rc)
            scurv.append(sc)
            if(sc > 0): sscurv.append(1)
            else: sscurv.append(-1)
        
        # Remove outliers.
        self.remove_outliers(rcurv)
        self.remove_outliers(scurv)
    
        # Calculate the mean curvature.
        halflen = len(sscurv) / 2
        if(len(sscurv) % 2 == 0):
            m1 = 1.0*np.mean(sscurv[0:halflen])/halflen
            m2 = 1.0*np.mean(sscurv[halflen:])/halflen
            scurv_mean = m1 - m2
        else:
            m1 = 1.0*np.mean(sscurv[0:halflen])/halflen
            m2 = 1.0*np.mean(sscurv[halflen:])/(halflen+1)
            scurv_mean = m1 - m2

        #print "Curvature asymmetry factor is {0}".format(scurv_mean);
        self.l_scurv_mean.append(scurv_mean)
    
        # Compute FFTs
        nfreqs = np.fft.fftfreq(len(trk_xM_0))
        wfreq, hfreq = signal.freqz(taps, worN=8000)
        ffxvals = np.fft.fft(trk_xM_0)
        #rffdxdz = np.real(ffdxdz);
        #print "Number of samples = {0}".format(len(dxdz));
        #print fnfreqs;

        # Fill the profile lists.
        Nsvals = len(sscurv)
        kon = 0
        for sval in sscurv:
            self.prof_sgn_kon.append(1.0*kon/Nsvals)
            self.prof_sgn_vals.append(sval)
            kon += 1

        # Calculate the profile comparison factors.
        chi2S = 1.0; chi2B = -1.0
        if(self.prof_cf):

            # Compute the fchi2F and fchi2R.
            ntotpts = Nsvals
            chi2S = 0.; chi2B = 0.; ndof = 0
            for npt in range(ntotpts):
                kon = 1.0*npt/ntotpts
                sgn = sscurv[npt]

                if(kon > self.kon_min and kon < self.kon_max):
                    chi2S += (sgn-self.sprof(kon))**2/self.sprof_sigma(kon)**2
                    chi2B += (sgn-self.bprof(kon))**2/self.bprof_sigma(kon)**2
                    ndof += 1
                #print "Signal: k/N = {0}, sign is {1}, prof. is {2}, prof. sigma is {3}".format(kon,sgn,sprof(kon),sprof_sigma(kon));
            self.l_chi2S.append(chi2S/ndof)
            self.l_chi2B.append(chi2B/ndof)

        else:

            self.l_chi2S.append(-1.0)
            self.l_chi2B.append(-1.0)

        # Draw the track if the option is set.
        if(self.plt_drawtrk):
        
            # Create the lists of (x,y,z) for positive and negative curvature.
            pc_x = []; pc_y = []; pc_z = [];
            nc_x = []; nc_y = []; nc_z = [];
            for cv,xx,yy,zz in zip(sscurv, trk_xM, trk_yM, trk_zM):
                if(cv > 0):
                    pc_x.append(xx);
                    pc_y.append(yy);
                    pc_z.append(zz);
                else:
                    nc_x.append(xx);
                    nc_y.append(yy);
                    nc_z.append(zz);

            nn0 = []; n = 0;
            for zval in trk_zM_0:
                nn0.append(n);
                n += 1;

            fig = plt.figure(2);
            fig.set_figheight(15.0);
            fig.set_figwidth(10.0);
        
            ax1 = fig.add_subplot(321);
            ax1.plot(nn,dxdz,'.-',color='red');
            #ax1.plot(nn0,trk_xM_0,'.-',color='red');
            #ax1.plot(nn,trk_xM[0:-1],'.-',color='green');
            ax1.set_xlabel("hit number n");
            ax1.set_ylabel("dx/dz");
        
            ax2 = fig.add_subplot(322);
#           ax2.plot(nn,dydz,'.',color='green')
#           ax2.plot(nn,f_dydz,'-',color='red');
            ax2.plot(nn,d2xdz2,'.-',color='green');
            ax2.set_ylim([-15.,15.]);
            ax2.set_xlabel("hit number n");
            ax2.set_ylabel("d$^2$x/dz$^2$");
        
            #ax3 = fig.add_subplot(323); #fig = plt.figure(1);
            #scurvn, scurvbins, scurvpatches = ax3.hist(scurv, 40, normed=0, histtype='step',color='blue');
            #ax3.set_xlabel("signed curvature");
            #ax3.set_ylabel("Counts/bin");
           
            #print "Wfrequencies are:";
            #print wfreq 
            ax3 = fig.add_subplot(323);
            ax3.plot(nfreqs,abs(ffxvals),'.',color='black',label='FFT');
            ax3.plot(wfreq/pi, 80*abs(hfreq),label='LP filter x80');
            lnd = plt.legend(loc=1,frameon=False,handletextpad=0,fontsize=8);
            ax3.set_ylim([0,100]);
            ax3.set_xlabel("frequency");
            ax3.set_ylabel("FFT(x)");
            #print fnfreqs;
            #print ffxvals;
            
            ax4 = fig.add_subplot(324);
            ax4.plot(nn,sscurv,'-',color='green')
            ax4.set_ylim([-1.5,1.5]);
            ax4.set_xlabel("hit number n");
            ax4.set_ylabel("sign of curvature");
            ax4.set_title("mean sgn(curvature) = {0}".format(round(np.mean(sscurv),4)));
            
            # Create the 3D track plot.
            ax5 = fig.add_subplot(325, projection='3d');
            ax5.plot(trk_xM,trk_yM,trk_zM,'-',color='black');
            ax5.plot(pc_x,pc_y,pc_z,'+',color='red');
            ax5.plot(nc_x,nc_y,nc_z,'.',color='blue');
            ax5.set_xlabel("x (mm)");
            ax5.set_ylabel("y (mm)");
            ax5.set_zlabel("z (mm)");
            
            lb_x = ax5.get_xticklabels();
            lb_y = ax5.get_yticklabels();
            lb_z = ax5.get_zticklabels();
            for lb in (lb_x + lb_y + lb_z):
                lb.set_fontsize(8);
            
            # Create the x-y projection.
            ax6 = fig.add_subplot(326);
            ax6.plot(trk_xM,trk_yM,'-',color='black');
            ax6.plot(pc_x,pc_y,'+',color='red');
            ax6.plot(nc_x,nc_y,'.',color='blue');        
            ax6.set_xlabel("x (mm)");
            ax6.set_ylabel("y (mm)");
            
            plt.savefig("{0}/plt_signals_{1}_{2}.pdf".format(self.plt_base,self.trk_name,trk_num), bbox_inches='tight');
            plt.close();
            #plt.show();

            # Plot the 3D plot with curvature designation alone.
            fig = plt.figure(3);
            fig.set_figheight(5.0);
            fig.set_figwidth(7.5);

            ax1 = fig.add_subplot(111, projection='3d');
            ax1.plot(trk_xM,trk_yM,trk_zM,'-',color='black');
            ax1.plot(pc_x,pc_y,pc_z,'+',color='red');
            ax1.plot(nc_x,nc_y,nc_z,'.',color='blue');
            ax1.set_xlabel("x (mm)");
            ax1.set_ylabel("y (mm)");
            ax1.set_zlabel("z (mm)");

            ax1.w_xaxis.set_pane_color((1.0,1.0,1.0,1.0));
            ax1.w_yaxis.set_pane_color((1.0,1.0,1.0,1.0));
            ax1.w_zaxis.set_pane_color((1.0,1.0,1.0,1.0));
            ax1.w_xaxis._axinfo.update({'grid' : {'color': (self.grdcol, self.grdcol, self.grdcol, 1)}});
            ax1.w_yaxis._axinfo.update({'grid' : {'color': (self.grdcol, self.grdcol, self.grdcol, 1)}});
            ax1.w_zaxis._axinfo.update({'grid' : {'color': (self.grdcol, self.grdcol, self.grdcol, 1)}});
            
            lb_x = ax1.get_xticklabels();
            lb_y = ax1.get_yticklabels();
            lb_z = ax1.get_zticklabels();
            for lb in (lb_x + lb_y + lb_z):
                lb.set_fontsize(8);

            plt.savefig("{0}/plt_trkcurv_{1}_{2}.pdf".format(self.plt_base,self.trk_name,trk_num), bbox_inches='tight');
            plt.close();

            # Plot the 3D plot alone.
            fig = plt.figure(4);
            fig.set_figheight(5.0);
            fig.set_figwidth(7.5);

            ax7 = fig.add_subplot(111, projection='3d');
            #rainbw = plt.get_cmap('Blues');
            #cNorm  = mpcol.Normalize(vmin=0, vmax=1.);
            #scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=rainbw);
            #for eval in trk_eM:
                #cval = scalarMap.to_rgba(eval);
                #print "color= ", eval, cval
                #carr.append(eval/mval);
                #carr.append([0.0, eval/mval,0.0]);
                #carr.append([cval[0], cval[1], cval[2], cval[3]])
            #print "carr =", carr
            #print "Lengths ... xlen = {0}, ylen = {1}, elen = {2}".format(len(trk_xM),len(trk_yM),len(trk_eM));
            #ax1.plot(trk_xM,trk_yM,trk_zM,color='0.9');
            s7 = ax7.scatter(trk_xM,trk_yM,trk_zM,marker='o',s=30,linewidth=0.2,c=trk_eM*1000,cmap=plt.get_cmap('rainbow'),vmin=0.0,vmax=max(trk_eM*1000));
            s7.set_edgecolors = s7.set_facecolors = lambda *args:None;  # this disables automatic setting of alpha relative of distance to camera
            ax7.set_xlabel("x (mm)");
            ax7.set_ylabel("y (mm)");
            ax7.set_zlabel("z (mm)");
            ax7.grid(True);

            ax7.w_xaxis.set_pane_color((1.0,1.0,1.0,1.0));
            ax7.w_yaxis.set_pane_color((1.0,1.0,1.0,1.0));
            ax7.w_zaxis.set_pane_color((1.0,1.0,1.0,1.0));
            ax7.w_xaxis._axinfo.update({'grid' : {'color': (self.grdcol, self.grdcol, self.grdcol, 1)}});
            ax7.w_yaxis._axinfo.update({'grid' : {'color': (self.grdcol, self.grdcol, self.grdcol, 1)}});
            ax7.w_zaxis._axinfo.update({'grid' : {'color': (self.grdcol, self.grdcol, self.grdcol, 1)}});
 
            lb_x = ax7.get_xticklabels();
            lb_y = ax7.get_yticklabels();
            lb_z = ax7.get_zticklabels();
            for lb in (lb_x + lb_y + lb_z):
                lb.set_fontsize(8);

            plt.title("Filtered");
            cb7 = plt.colorbar(s7);
            cb7.set_label('Hit energy (keV)');
            plt.savefig("{0}/plt_trk_flt_{1}_{2}.pdf".format(self.plt_base,self.trk_name,trk_num), bbox_inches='tight');
            plt.close();

            # Plot the unfiltered 3D plot alone.
            fig = plt.figure(5);
            fig.set_figheight(5.0);
            fig.set_figwidth(7.5);

            ax8 = fig.add_subplot(111, projection='3d');
            #ax1.plot(trk_xM_0,trk_yM_0,trk_zM_0,'-',color='0.9');
            trk_nM_arr = np.array(trk_nM_0)
            trk_eM_arr = np.array(trk_eM_0)
            #s8 = ax8.scatter(trk_xM_0,trk_yM_0,trk_zM_0,marker='o',s=30,linewidth=0.2,c=trk_eM_arr*1000,cmap=plt.get_cmap('rainbow'),vmin=0.0,vmax=max(trk_eM_0*1000));
            s8 = ax8.scatter(trk_xM_0,trk_yM_0,trk_zM_0,marker='o',s=30,linewidth=0.2,c=trk_nM_arr,cmap=plt.get_cmap('rainbow'),vmin=0.0,vmax=max(trk_nM_0));
            s8.set_edgecolors = s8.set_facecolors = lambda *args:None;  # this disables automatic setting of alpha relative of distance to camera
            #ax1.plot(trk_xM_0,trk_yM_0,trk_zM_0,'.',color='black');
            ax8.set_xlabel("x (mm)");
            ax8.set_ylabel("y (mm)");
            ax8.set_zlabel("z (mm)");

            ax8.w_xaxis.set_pane_color((1.0,1.0,1.0,1.0));
            ax8.w_yaxis.set_pane_color((1.0,1.0,1.0,1.0));
            ax8.w_zaxis.set_pane_color((1.0,1.0,1.0,1.0));
            ax8.w_xaxis._axinfo.update({'grid' : {'color': (self.grdcol, self.grdcol, self.grdcol, 1)}});
            ax8.w_yaxis._axinfo.update({'grid' : {'color': (self.grdcol, self.grdcol, self.grdcol, 1)}});
            ax8.w_zaxis._axinfo.update({'grid' : {'color': (self.grdcol, self.grdcol, self.grdcol, 1)}});
            
            lb_x = ax8.get_xticklabels();
            lb_y = ax8.get_yticklabels();
            lb_z = ax8.get_zticklabels();
            for lb in (lb_x + lb_y + lb_z):
                lb.set_fontsize(8);

            plt.title("Unfiltered: $\chi^2_B/\chi^2_S = $ {0}".format(chi2B/chi2S));
            cb8 = plt.colorbar(s8);
            #cb8.set_label('Hit energy (keV)');
            cb8.set_label('Hit number');
            plt.savefig("{0}/plt_trk_unflt_{1}_{2}.pdf".format(self.plt_base,self.trk_name,trk_num), bbox_inches='tight');
            plt.close();
     
        if(self.plt_drawfilter):
        
            fig = plt.figure(6);
            fig.set_figheight(5.0);
            fig.set_figwidth(7.5);       

            plt.clf();
            ax1 = fig.add_subplot(111);
        
            ax1.plot(wfreq/(2*pi), np.absolute(hfreq), linewidth=2, color='black');
            ax1.vlines(fcbar, -0.05, 1.15, color='blue', linestyle='--', lw=2);
            ax1.set_xlabel('Frequency (cycles/samples)');
            ax1.set_ylabel('Lowpass filter gain');
            ax1.set_xlim(0.0, 0.5);
            ax1.set_ylim(0.0, 1.1);

            ax2 = ax1.twinx();
            ax2.plot(nfreqs[0:len(nfreqs)/2],abs(ffxvals[0:len(ffxvals)/2]),'-',color='#cc0000', lw=2);
            ax2.set_ylabel('FFT amplitude');
            #plt.title('Frequency Response');
            ax2.set_xlim(0.0, 0.5);
            ax2.set_ylim(-0.05, 120.05);
            #plt.grid(True);
            ax2.yaxis.label.set_color('#cc0000');
            for tl in ax2.get_yticklabels():
                tl.set_color('#cc0000')

            #print nfreqs
            plt.savefig("{0}/FIR_freq_resp_{1}_{2}.pdf".format(self.plt_base,self.trk_name,trk_num), bbox_inches='tight');
            plt.close();

        return True

    def finalize(self):


        # Make the plot of blob energy 1 vs. blob energy 2.
        fig = plt.figure(7);
        fig.set_figheight(5.0);
        fig.set_figwidth(7.5);

        #ax1 = fig.add_subplot(111);
        #ax1.plot(self.l_eblob1,self.l_eblob2,'.',color='black');
        hblobs, xblobs, yblobs = np.histogram2d(self.l_eblob2, self.l_eblob1, bins=(50, 50));
        #extent = [yblobs[0], yblobs[-1], yblobs[0], yblobs[-1]]
        extent = [yblobs[0], yblobs[-1], xblobs[0], xblobs[-1]]
        plt.imshow(hblobs, extent=extent, interpolation='nearest', aspect='auto', origin='lower')
        #plt.axis([yblobs[0],yblobs[-1],yblobs[0],yblobs[-1]])
        plt.xlabel("Blob 1 Energy (MeV)");
        plt.ylabel("Blob 2 Energy (MeV)");
        plt.savefig("{0}/blob_energies_{1}.pdf".format(self.plt_base,self.trk_name), bbox_inches='tight');
        plt.close();

        # Print out the number of tracks that were flipped.
        print "Flipped {0} tracks".format(self.flip_trk)
 
        # Output the list of mean curvature values and fraction of positive curvature values.
        if(self.output_means):

            print "Writing file with {0} entries...".format(len(self.l_scurv_mean))
            fm = open("{0}/scurv_means.dat".format(self.plt_base),"w")
            fm.write("# (trk) (asymm) (chi2s) (chi2b) (nhits) (Eblob1) (Eblob2) (blob flipped)\n")
            for evtnum,scurv,chi2s,chi2b,nhits,eblob1,eblob2,bflip in zip(self.l_evtnum,self.l_scurv_mean,self.l_chi2S,self.l_chi2B,self.l_nhits,self.l_eblob1,self.l_eblob2,self.l_bflip):
                fm.write("{0} {1} {2} {3} {4} {5} {6} {7}\n".format(evtnum,scurv,chi2s,chi2b,nhits,eblob1,eblob2,bflip))
            fm.close()

        # Generate the profiles.
        if(self.prof_gen):

            # ---------------------------------------------------------------------------
            # Create the sign vs. k/N profiles.
            prof_kon = []
            prof_sgnN = []; prof_sgn = []; prof_sigma = []

            for nn in range(self.nbins_kon):
                prof_kon.append(1.0*(nn+0.5)/self.nbins_kon)
                prof_sgnN.append(0); prof_sgn.append(0.); prof_sigma.append(0.)

            for kon,sgn in zip(self.prof_sgn_kon,self.prof_sgn_vals):
                bb = int(kon*self.nbins_kon)
                prof_sgnN[bb] += 1
                prof_sgn[bb] += sgn
                prof_sigma[bb] += sgn**2

            # Normalize.
            for bb in range(self.nbins_kon):
                if(prof_sgnN[bb] > 1):
                    NN = prof_sgnN[bb]
                    mu = prof_sgn[bb]/prof_sgnN[bb]
                    prof_sgn[bb] = mu
                    prof_sigma[bb] = sqrt((prof_sigma[bb])/(NN-1) - NN*mu**2/(NN-1))

            # Write the file.
            f_prof = open(self.prof_file,"w")
            f_prof.write("# (k/N) (sgn) (sigma) (N)\n")
            for kon,sgn,sigma,NN in zip(prof_kon,prof_sgn,prof_sigma,prof_sgnN):
                f_prof.write("{0} {1} {2} {3}\n".format(kon,sgn,sigma,NN))
            f_prof.close()

        # Plot the existing profiles.
        if(not self.prof_gen and os.path.isfile(self.prof_sfile) and os.path.isfile(self.prof_bfile)):

            l_profx = np.arange(0.0,1.0,0.01);

            l_sprofy = []; l_sprofy_sp = []; l_sprofy_sm = [];
            l_bprofy = []; l_bprofy_sp = []; l_bprofy_sm = [];
            for xval in l_profx:
                sval = self.sprof(xval); ssval = self.sprof_sigma(xval);
                bval = self.bprof(xval); sbval = self.bprof_sigma(xval);
                l_sprofy.append(sval);
                l_sprofy_sp.append(sval+ssval);
                l_sprofy_sm.append(sval-ssval);
                l_bprofy.append(bval);
                l_bprofy_sp.append(bval+sbval);
                l_bprofy_sm.append(bval-sbval);

            # Create plots of the sign vs. N profiles.
            fig = plt.figure(8);
            fig.set_figheight(5.0);
            fig.set_figwidth(7.5);

            ax1 = fig.add_subplot(111);
            ax1.plot(l_profx,l_sprofy,'-',lw=3,color='red',label='Signal');
            ax1.plot(l_profx,l_bprofy,'-',lw=3,color='blue',label='Background');
            lnd = plt.legend(loc=4,frameon=False,handletextpad=0,fontsize=12);
            #ax1.plot(l_profx,l_sprofy_sm,'--',color='black');
            #ax1.plot(l_profx,l_sprofy_sp,'--',color='black');

            ax1.set_xlabel("Fractional location along track (k/N)");
            ax1.set_ylabel("Average sign profile (signal)");
            plt.savefig("{0}/sbprof_{1}.pdf".format(self.plt_base,self.trk_name), bbox_inches='tight');
            plt.close();
        
        self.m.log(1,'+++End method of algo_example algorithm+++')

        return
