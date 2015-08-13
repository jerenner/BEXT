from Centella.AAlgo import AAlgo
from Centella.physical_constants import *
from math import sqrt

import matplotlib.pyplot as plt
import matplotlib.colors as mpcol
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
from ROOT import gate

import random as rd

"""
This algorithm smears (in x,y) and/or sparses (in z) the MC hits according 
to the specified xy smearing and z sparse width.

The smeared hits are placed in GATE tracks under the label gate::NOSTYPE
(or gate.NOSTYPE in Python).  They are also placed in a single hit collection
under the same label.  This is done so that:
- a "true" MC track can be constructed by examining the information in
  individual MC tracks (unsmeared): since a corresponding smeared track will
  be present for every true track, the smeared hits can be used in place of
  the true hits when constructing the track, but the true hits can be used to
  determine the ordering of the MC tracks in constructing the final continuous
  track
- algorithims such as the MST will have an indexed collection of all hits on which to
  operate

"""


class MCHitSmearing(AAlgo):

  ############################################################
  def __init__(self, param=False, level=1, label="", **kargs):

    """
    Energy Smearing Algorithm
    """
    #self.m.log(1, 'Constructor()')

    ### GENERAL STUFF
    self.name = 'MCHitSmearing'
    #self.level = level
    AAlgo.__init__(self, param, level, self.name, 0, label, kargs)

    ### PARAMETERS
    # MC hit z-sparsing
    try:
      self.sparse_width = self.ints['sparse_width']
      self.m.log(1, "z-sparsing is {0}".format(self.sparse_width))
    except KeyError:
      self.m.log(1, "WARNING!! Parameter: 'sparse_width' not defined.")
      exit(0)
    
    # MC hit x-y smearing
    try:
      self.xy_smearing = self.doubles['xy_smearing']
      self.m.log(1, "xy-smearing is {0} mm".format(self.xy_smearing))
    except KeyError:
      self.m.log(1, "WARNING!! Parameter: 'xy_smearing' not defined.")
      exit(0)
      
    # Flag to plot tracks.
    try:
        self.par_plt_tracks = self.ints['plt_tracks']
        self.plt_tracks = False
        if(self.par_plt_tracks == 1):
            self.plt_tracks = True
        self.m.log(1, "Plot tracks = {0}".format(self.plt_tracks))
    except KeyError:
        self.m.log(1, "WARNING!! Parameter: 'plt_tracks' not defined.")
        exit(0)
    
    # Plot base output directory.
    try:
        self.plt_base = self.strings['plt_base']
        self.m.log(1, "Plot base dir = {0}".format(self.plt_base))
    except KeyError:
        self.m.log(1, "WARNING!! Parameter: 'plt_base' not defined.")
        exit(0)
        
    # Plot output format.
    try:
        self.out_fmt = self.strings['out_fmt']
        self.m.log(1, "Plot output format = {0}".format(self.out_fmt))
    except KeyError:
        self.m.log(1, "WARNING!! Parameter: 'out_fmt' not defined.")
        exit(0)

    # Misc. options
    self.grdcol = 0.98

  ############################################################    
  def initialize(self):

    self.m.log(1, 'Initialize()')

    return



  ############################################################
  def execute(self, event=""):

    self.m.log(2, 'Execute()')

    # Get the event number.
    trk_num = self.event.GetEventID()   
 
    # Get the MC hits and smear and/or sparse them.
    #hList = self.event.GetMCHits()
    mctracks = self.event.GetMCTracks()
    
    for trk in mctracks:
        
        # Get all hits in the track.
        hList = trk.GetHits()
        
        # Sparse the hits in this track.
        hProc = []
        if(self.sparse_width > 1):
    
            ihit = 0; nhit = 0
            while(ihit < len(hList)):
    
                sp = ihit
    
                # Create an effective (sparsed) hit.
                xeff = 0.; yeff = 0.; zeff = 0.; eeff = 0.
                while(sp < len(hList) and sp < (ihit + self.sparse_width)):
                    ehit = hList[sp].GetAmplitude()
                    xeff += hList[sp].GetPosition().x()*ehit
                    yeff += hList[sp].GetPosition().y()*ehit
                    zeff += hList[sp].GetPosition().z()*ehit
                    eeff += ehit
                    sp += 1
    
                # Finish calculating effective coordinates.
                xeff /= eeff
                yeff /= eeff
                zeff /= eeff
    
                # Record the values in lists for the sparsed track.
                ghit = gate.Hit()
                ghit.SetID(nhit)
                #ghit.SetParticle(hList[ihit].GetParticle())
                ghit.SetAmplitude(eeff)
                ghit.SetPosition(xeff,yeff,zeff)
                #ghit.SetTime(hList[ihit].T())
                hProc.append(ghit)
    
                nhit += 1
                ihit += self.sparse_width
                
        # Otherwise, perform no sparsing.
        else:
            hProc = hList
                    
        # Smear the hits in this track.
        if(self.xy_smearing > 0):
    
            for hsp in hProc:
                xsm = rd.gauss(hsp.GetPosition().x(),self.xy_smearing)
                ysm = rd.gauss(hsp.GetPosition().y(),self.xy_smearing)
                hsp.SetPosition(xsm,ysm,hsp.GetPosition().z())

        # Make a new track to contain the smeared hits.
        newtrk = gate.Track()
        
        # Add the smeared hit to the event object and the new track.
        for hfinal in hProc:
            newtrk.AddHit(hfinal)
            self.event.AddHit(gate.NOSTYPE,hfinal)
            
        # Add the new track to the event object.
        self.event.AddTrack(gate.NOSTYPE,newtrk)
        
    if(self.plt_tracks):

        # Fill arrays for the smeared track.
        hSmeared = self.event.GetHits(gate.NOSTYPE)
        mcstrk_ID = []; mcstrk_x = []; mcstrk_y = []; mcstrk_z = []; mcstrk_E = [];
        for hh in hSmeared:
            mcstrk_ID.append(hh.GetID());
            mcstrk_x.append(hh.GetPosition().x());
            mcstrk_y.append(hh.GetPosition().y());
            mcstrk_z.append(hh.GetPosition().z());
            mcstrk_E.append(hh.GetAmplitude());
        
        # Fill the arrays for the MC (unsmeared) track.
        hMC = self.event.GetMCHits()
        mctrk_ID = []; mctrk_x = []; mctrk_y = []; mctrk_z = []; mctrk_E = [];
        for hh in hMC:
            mctrk_ID.append(hh.GetID());
            mctrk_x.append(hh.GetPosition().x());
            mctrk_y.append(hh.GetPosition().y());
            mctrk_z.append(hh.GetPosition().z());
            mctrk_E.append(hh.GetAmplitude());
        
        # ---------------------------------------------------------------------
        # Plot MC vs. smeared MC.
        fig = plt.figure(1);
        fig.set_figheight(5.0);
        fig.set_figwidth(15.0);
        
        ax1 = fig.add_subplot(121, projection='3d');
        mctrk_col = []; mid = 0;
        for mid0 in mctrk_ID:
            mctrk_col.append(mid);
            mid += 1;
        s1 = ax1.scatter(mctrk_x,mctrk_y,mctrk_z,marker='s',s=30,linewidth=0.2,c=mctrk_col,cmap=plt.get_cmap('rainbow'),vmin=min(mctrk_col),vmax=max(mctrk_col));
        s1.set_edgecolors = s1.set_facecolors = lambda *args:None;  # this disables automatic setting of alpha relative of distance to camera
        ax1.set_xlabel("x (mm)");
        ax1.set_ylabel("y (mm)");
        ax1.set_zlabel("z (mm)");
        ax1.set_title("Monte Carlo Truth");

        lb_x = ax1.get_xticklabels();
        lb_y = ax1.get_yticklabels();
        lb_z = ax1.get_zticklabels();
        for lb in (lb_x + lb_y + lb_z):
            lb.set_fontsize(8);
        
        ax1.w_xaxis.set_pane_color((1.0,1.0,1.0,1.0));
        ax1.w_yaxis.set_pane_color((1.0,1.0,1.0,1.0));
        ax1.w_zaxis.set_pane_color((1.0,1.0,1.0,1.0));
        ax1.w_xaxis._axinfo.update({'grid' : {'color': (self.grdcol, self.grdcol, self.grdcol, 1)}});
        ax1.w_yaxis._axinfo.update({'grid' : {'color': (self.grdcol, self.grdcol, self.grdcol, 1)}});
        ax1.w_zaxis._axinfo.update({'grid' : {'color': (self.grdcol, self.grdcol, self.grdcol, 1)}});
            
        cb1 = plt.colorbar(s1);
        cb1.set_label('Hit number');
        
        ax2 = fig.add_subplot(122,projection='3d');
        rcol = [];
        ind = 0;
        while(ind < len(mcstrk_x)):
            rcol.append(ind);
            ind += 1;                    
        s2 = ax2.scatter(mcstrk_x,mcstrk_y,mcstrk_z,marker='s',s=30,linewidth=0.2,c=rcol,cmap=plt.get_cmap('rainbow'),vmin=min(rcol),vmax=max(rcol));
        s2.set_edgecolors = s2.set_facecolors = lambda *args:None;  # this disables automatic setting of alpha relative of distance to camera
        ax2.set_xlabel("x (mm)");
        ax2.set_ylabel("y (mm)");
        ax2.set_zlabel("z (mm)");
        ax2.set_title("Smeared MC Truth");
        
        lb_x = ax2.get_xticklabels();
        lb_y = ax2.get_yticklabels();
        lb_z = ax2.get_zticklabels();
        for lb in (lb_x + lb_y + lb_z):
            lb.set_fontsize(8);
            
        ax2.w_xaxis.set_pane_color((1.0,1.0,1.0,1.0));
        ax2.w_yaxis.set_pane_color((1.0,1.0,1.0,1.0));
        ax2.w_zaxis.set_pane_color((1.0,1.0,1.0,1.0));
        ax2.w_xaxis._axinfo.update({'grid' : {'color': (self.grdcol, self.grdcol, self.grdcol, 1)}});
        ax2.w_yaxis._axinfo.update({'grid' : {'color': (self.grdcol, self.grdcol, self.grdcol, 1)}});
        ax2.w_zaxis._axinfo.update({'grid' : {'color': (self.grdcol, self.grdcol, self.grdcol, 1)}});
     
        cb2 = plt.colorbar(s2);
        cb2.set_label('Voxel number');

        fn_plt = "{0}/MCHitSmearing_mcvsmcs_{1}.{2}".format(self.plt_base,trk_num,self.out_fmt);
        plt.savefig(fn_plt, bbox_inches='tight');
            
        plt.close();

    return True

  ############################################################
  def finalize(self):

    self.m.log(1, 'Finalize()')

    return
