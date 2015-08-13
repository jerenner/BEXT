#
# mst.py
#
# Determines track ordering using MST-based algorithm.
#

import networkx as nx
import os
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import matplotlib.colors as mpcol
import matplotlib.cm as cmx

from math import *
from Centella.AAlgo import AAlgo
from Centella.physical_constants import *

from ROOT import gSystem
gSystem.Load("$GATE_DIR/lib/libGATE")
gSystem.Load("$GATE_DIR/lib/libGATEIO")
gSystem.Load("$GATE_DIR/lib/libGATEUtils")
from ROOT import gate
from ROOT import std 

debug = 2;

# TrackNode class
#  Node types:
#   0: normal
#   1: end node
#   2: vertex/branch point
#   3: end node that should be connected to another part of the track
#       (node specified as "link")
class TrackNode:    
    
    # Creates a new TrackNode
    def __init__(self,idnum):
        self.id = idnum;
        self.visited = False;
        self.type = 0;
        self.link = -1;  # node to which this is connected for type 3
        
    def __repr__(self):
        return "Node {0}, type {1}, visited {2}".format(self.id,self.type,self.visited);

    def __str__(self):
        return "Node {0}, type {1}, visited {2}".format(self.id,self.type,self.visited);
        
# TrackSegment class
class TrackSegment:
        
    # Creates a new TrackSegment with specified start and end nodes.
    def __init__(self,inode,fnode):
        self.inode = inode;
        self.fnode = fnode;
        self.path = [];
        self.length = 0.0;
        
    def set_path(self,pth):
        self.path = pth;
        self.inode = pth[0];
        self.fnode = pth[-1];
    
    def set_len(self,length):
        self.length = length;
   
    def __repr__(self):
        return "Segment: node {0} (type {1}) to node {2} (type {3}), len {4}, path len {5}".format(self.inode.id,self.inode.type,self.fnode.id,self.fnode.type,len(self.path),self.length);
     
    def __str__(self):
        return "Segment: node {0} (type {1}) to node {2} (type {3}), len {4}, path len {5}".format(self.inode.id,self.inode.type,self.fnode.id,self.fnode.type,len(self.path),self.length);


class mst(AAlgo):

    def __init__(self,param=False,level = 1,label="",**kargs):        
            
        self.name='mst'
        
        AAlgo.__init__(self,param,level,self.name,0,label,kargs)

        ### PARAMETERS
        # Minimum number of voxels in a segment.
        try:
            self.min_path_voxels = self.ints['min_path_voxels']
            self.m.log(1, "Minimum voxels per segment = {0}".format(self.min_path_voxels))
        except KeyError:
            self.m.log(1, "WARNING!! Parameter: 'min_path_voxels' not defined.")
            exit(0)
            
        # Number of voxels allowed in segment overlap.
        try:
            self.path_overlap_tol = self.ints['path_overlap_tol']
            self.m.log(1, "Path overlap tolerance = {0}".format(self.path_overlap_tol))
        except KeyError:
            self.m.log(1, "WARNING!! Parameter: 'path_overlap_tol' not defined.")
            exit(0)
            
        # Flag to use MC hits.
        try:
            self.par_use_voxels = self.ints['use_voxels']
            self.use_voxels = False
            if(self.par_use_voxels == 1):
                self.use_voxels = True
            self.m.log(1, "Use voxels = {0}".format(self.use_voxels))
        except KeyError:
            self.m.log(1, "WARNING!! Parameter: 'use_voxels' not defined.")
            exit(0)
            
        # Maximum distance between neighboring MC hits
        try:
            self.nbr_dist = self.doubles['nbr_dist']
            self.m.log(1, "Maximum distance between neighboring MC hits = {0}".format(self.nbr_dist))
        except KeyError:
            self.m.log(1, "WARNING!! Parameter: 'nbr_dist' not defined.")
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

    def initialize(self):

        """

        You can use here:

            self.hman     -----> histoManager instance
            self.tman     -----> treeManager instance
            self.logman   -----> logManager instance
            self.autoDoc  -----> latex_gen instance
        
        """
        
        self.m.log(1,'+++Init method of mst algorithm+++')

        return

    def execute(self,event=""):

        # Get the event number.
        trk_num = self.event.GetEventID()

        # Get all tracks from the voxelization and mc steps. 
        all_trks = self.event.GetTracks(gate.SIPM)
        if(len(all_trks) != 1):
            self.m.log("ERROR: Event has more than 1 track.")
            exit(0)

        # Get the initial track for this event.
        main_trk = all_trks[0]

        # Get the correct list of hits voxels or MC smeared (NOSTYPE).
        if(self.use_voxels):
            hList = self.event.GetHits()
            print "Performing MST with {0} voxels.".format(len(hList)); 
        else: 
            hList = self.event.GetHits(gate.NOSTYPE)
            print "Performing MST with {0} smeared/sparsed MC hits.".format(len(hList));

        # Determine the ordered track according to the MST algorithm.
        (ftrack,fsegments) = self.trk_mst(hList)

        # Place the ordering of the hits in the MST track in the vstore of
        #  the tracks object.
        mst_ord = std.vector(int)()
        for hh in ftrack:
            mst_ord.push_back(hh.GetID())
        main_trk.store("MSTHits",mst_ord)

        if(self.plt_tracks):

            # Create the paths lists for each segment.
            spaths = [];
            for nseg in fsegments: 
                spaths.append(nseg.path);
                if(debug > 1): print "{0} with {1} nodes.".format(nseg,len(nseg.path));
            
            # Fill arrays for the MST track.
            msttrk_ID = []; msttrk_x = []; msttrk_y = []; msttrk_z = []; msttrk_E = [];
            for hh in ftrack:
                msttrk_ID.append(hh.GetID());
                msttrk_x.append(hh.GetPosition().x());
                msttrk_y.append(hh.GetPosition().y());
                msttrk_z.append(hh.GetPosition().z());
                msttrk_E.append(hh.GetAmplitude());
            
            # Fill the arrays for the MC track.
            mctrk_ID = []; mctrk_x = []; mctrk_y = []; mctrk_z = []; mctrk_E = [];
            for hh in hList:
                mctrk_ID.append(hh.GetID());
                mctrk_x.append(hh.GetPosition().x());
                mctrk_y.append(hh.GetPosition().y());
                mctrk_z.append(hh.GetPosition().z());
                mctrk_E.append(hh.GetAmplitude());
            
            # ---------------------------------------------------------------------
            # Plot MC vs. MST.
            fig = plt.figure(1);
            fig.set_figheight(10.0);
            fig.set_figwidth(15.0);
            
            ax1 = fig.add_subplot(221, projection='3d');
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
            
            ax2 = fig.add_subplot(222,projection='3d');
            rcol = [];
            ind = 0;
            while(ind < len(ftrack)):
                rcol.append(ind);
                ind += 1;                    
            s2 = ax2.scatter(msttrk_x,msttrk_y,msttrk_z,marker='s',s=30,linewidth=0.2,c=rcol,cmap=plt.get_cmap('rainbow'),vmin=min(rcol),vmax=max(rcol));
            s2.set_edgecolors = s2.set_facecolors = lambda *args:None;  # this disables automatic setting of alpha relative of distance to camera
            ax2.set_xlabel("x (mm)");
            ax2.set_ylabel("y (mm)");
            ax2.set_zlabel("z (mm)");
            ax2.set_title("MST/Segmenting Algorithm Track");
            
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

            ax3 = fig.add_subplot(223,projection='3d');
            colarr = ['red','green','blue','orange','black','violet'];
            npth = 0;
            for pth in spaths:
                xpth = []; ypth = []; zpth = [];
                # Note: IDs in segments corresponding to ordering in hList. 
                for nn in pth:
                    xpth.append(hList[nn.id].GetPosition().x()); ypth.append(hList[nn.id].GetPosition().y()); zpth.append(hList[nn.id].GetPosition().z());
                ax3.plot(xpth,ypth,zpth,'-',color=colarr[npth % len(colarr)]);
                npth += 1;
            ax3.set_xlabel("x (mm)");
            ax3.set_ylabel("y (mm)");
            ax3.set_zlabel("z (mm)");
            ax3.set_title("Segments");
            
            lb_x = ax3.get_xticklabels();
            lb_y = ax3.get_yticklabels();
            lb_z = ax3.get_zticklabels();
            for lb in (lb_x + lb_y + lb_z):
                lb.set_fontsize(8);
    
            fn_plt = "{0}/mst_mcvsmst_{1}.{2}".format(self.plt_base,trk_num,self.out_fmt);
            plt.savefig(fn_plt, bbox_inches='tight');
                
            plt.close();

        print "DONE WITH MST ALGORITHM"

        return True

    def finalize(self):
        
        self.m.log(1,'+++End method of mst algorithm+++')
        return

    # Construct a track from the specified list of hits.
    #  Note: the hit/voxel IDs correspond to the ordering in hList after this method is called.
    def trk_mst(self,hList):
       
        voxelsize = 2.0;
 
        # Set up the different lists
        vtrk_ID = []; vtrk_x = []; vtrk_y = []; vtrk_z = []; vtrk_E = [];
        vid = 0;
        for hhit in hList:
            vtrk_ID.append(vid);
            vtrk_x.append(hhit.GetPosition().x());
            vtrk_y.append(hhit.GetPosition().y());
            vtrk_z.append(hhit.GetPosition().z());
            vtrk_E.append(hhit.GetAmplitude());
            vid += 1;
        
        # ---------------------------------------------------------------------------
        # Create the adjacency matrix
        # -1 --> self
        # 0 --> not a neighbor
        # (distance) --> voxels are neighbors
        # ---------------------------------------------------------------------------
    
        # Use the voxels: determine neighboring voxels by face, edge, or corner connections.
        adjMat = []; nnbrMat = []
        if(self.use_voxels):
            
            # Iterate through all voxels, and for each one find the neighboring voxels.
            for vID1,vx1,vy1,vz1,vE1 in zip(vtrk_ID,vtrk_x,vtrk_y,vtrk_z,vtrk_E):
            
                nbr_list = [];
                nnbrs = 0;
                for vID2,vx2,vy2,vz2,vE2 in zip(vtrk_ID,vtrk_x,vtrk_y,vtrk_z,vtrk_E):
                    
                    if(vx1 == vx2 and vy1 == vy2 and vz1 == vz2):
                        nbr_list.append(-1);
                    elif ((vx1 == vx2+voxelsize or vx1 == vx2-voxelsize or vx1 == vx2) and (vy1 == vy2+voxelsize or vy1 == vy2-voxelsize or vy1 == vy2) and (vz1 == vz2+voxelsize or vz1 == vz2-voxelsize or vz1 == vz2)):
                        nbr_list.append(sqrt((vx2-vx1)**2 + (vy2-vy1)**2 + (vz2-vz1)**2));
                        nnbrs += 1;
                    else:
                        nbr_list.append(0);
                        
                nnbrMat.append(nnbrs);
                adjMat.append(nbr_list);
        
        # Use the MC true hits: determine neighboring hits by radial distance.
        else:
            
            # Iterate through all voxels, and for each one find the neighboring voxels.
            for vID1,vx1,vy1,vz1,vE1 in zip(vtrk_ID,vtrk_x,vtrk_y,vtrk_z,vtrk_E):
            
                nbr_list = [];
                nnbrs = 0;
                for vID2,vx2,vy2,vz2,vE2 in zip(vtrk_ID,vtrk_x,vtrk_y,vtrk_z,vtrk_E):
                    
                    if(vx1 == vx2 and vy1 == vy2 and vz1 == vz2):
                        nbr_list.append(-1);
                    elif (sqrt((vx1-vx2)**2 + (vy1-vy2)**2 + (vz1-vz2)**2) < self.nbr_dist):
                        nbr_list.append(sqrt((vx2-vx1)**2 + (vy2-vy1)**2 + (vz2-vz1)**2));
                        nnbrs += 1;
                    else:
                        nbr_list.append(0);
                        
                nnbrMat.append(nnbrs);
                adjMat.append(nbr_list);
        
        if(debug > 2):
            print "Adjacency matrix:";
            print adjMat;
    
        # ---------------------------------------------------------------------------
        # Construct the graph for the track
        # ---------------------------------------------------------------------------
        trk = nx.Graph();
        for vid in vtrk_ID:
            tnode = TrackNode(vid);
            trk.add_node(tnode);
        #print trk
        
        # Add edges connecting each node to its neighbor nodes based on the values
        #  in the adjacency matrix.
        n1 = 0;
        for nA in trk.nodes():
            
            n2 = 0;
            for nB in trk.nodes():
                ndist = adjMat[nA.id][nB.id];
                if(ndist > 0):
        #            print "Adding edge from {0} to {1}".format(n1,n2)
    #                trk.add_edge(nA,nB,weight=(ndist+0.000001*nA.id+0.000001*nB.id));
                    trk.add_edge(nA,nB,weight=(ndist)); #+0.000001*rd.random()));
                n2 += 1; 
                    
            n1 += 1;
    
        if(debug > 2):
            #nx.draw_random(trk);
            #plt.show();
            print trk.nodes()
            print trk.edges()
    
        # -------------------------------------------------------------------------
        #  Find the segments in the track.
        # -------------------------------------------------------------------------
        #nodes = find_nodes(trk,nnbrMat,vtrk_x,vtrk_y,vtrk_z);
        mst = nx.minimum_spanning_tree(trk);
        segments = self.find_segments_mst(mst,adjMat,vtrk_x,vtrk_y,vtrk_z);
        if(debug > 1):
            print "\nSegments:";
            print segments;
            for seg in segments:
                print "  --------  segment ---------   "
                print seg.inode
                print seg.fnode
                print seg.length
                print "path ",seg.path
               
        # -------------------------------------------------------------------------
        #  Fill gaps in the determined segments.
        # -------------------------------------------------------------------------
        segments_f = [];
        for seg in segments: segments_f.append(seg);
        
        # Attempt to match endpoints of all segments.
        for seg in segments:
    
            ep1 = seg.inode;
            seg1 = self.match_to_segments(ep1,seg,segments_f,trk);
            if(seg1 != 0):
                if(debug > 1): print "Appending segment of {0} nodes: {1}".format(len(seg1.path),seg1);
                segments_f.append(seg1);
                
            ep2 = seg.fnode;
            seg2 = self.match_to_segments(ep2,seg,segments_f,trk);
            if(seg2 != 0): 
                if(debug > 1): print "Appending segment of {0} nodes: {1}".format(len(seg2.path),seg2);
                segments_f.append(seg2);
            
        # -------------------------------------------------------------------------
        #  Connect the determined segments.
        # -------------------------------------------------------------------------    
        connection = True;
        segments_c = [];
        for seg in segments_f: segments_c.append(seg);
            
        while(connection):
            
            for cseg in segments_c:
            
                # Attempt to connect cseg with one or more segments in the list.
                (connection,segments_c) = self.connect_segment_in_list(mst,adjMat,cseg,segments_c);
                
                # If a connection was made, break and restart the iteration.
                if(connection):
                    if(debug > 1): 
                        print "Connection made, restarting iteration...";
                        print segments_c;
                    break;
                    
        spaths = [];
        for nseg in segments_c: 
            spaths.append(nseg.path);
            if(debug > 1): print "{0} with {1} nodes.".format(nseg,len(nseg.path));
    
        # -------------------------------------------------------------------------
        #  Find the longest path, starting with every segment.
        # -------------------------------------------------------------------------
        lfpath = -1; final_path = [];  # length and list of segment objects for final path
        for sseg in segments_c:
            
            if(debug > 1): print "Computing longest path starting with segment {0}".format(sseg);
            
            # Set all segments to unvisited.
            for vseg in segments_c: vseg.visited = False;
                
            # Compute the longest track starting with this segment, traversing
            #  it both forward and backward.
            currpth_fwd = []; currpth_fwd = self.compute_longest_track(segments_c,sseg,True,currpth_fwd);
            currpth_rev = []; currpth_rev = self.compute_longest_track(segments_c,sseg,False,currpth_rev);
            
            if(debug > 1): print "Length in segments of forward path is {0}, reverse path is {1}".format(len(currpth_fwd),len(currpth_rev));
            
            # If this was the longest path so far, save it.
            lpath_fwd = self.path_length_seg(currpth_fwd);
            if(lpath_fwd > lfpath):
                lfpath = lpath_fwd;
                final_path = currpth_fwd;
            
            lpath_rev = self.path_length_seg(currpth_rev);
            if(lpath_rev > lfpath):
                lfpath = lpath_rev;
                final_path = currpth_rev;
                
        if(debug > 0):
            print "Longest path (length {0}) is:".format(lfpath);
            print final_path;
        
        # -------------------------------------------------------------------------
        #  Create the list of hits from the longest path.
        # -------------------------------------------------------------------------    
        ftrk = []; nseg_fpath = len(final_path);
        
        # If only one segment, just add its nodes.
        if(nseg_fpath == 1):
            for nd in final_path[0].path:
                ftrk.append(nd);
        # Otherwise, must determine the correct node ordering when moving from
        #  segment to segment.
        elif(nseg_fpath > 1):
            iseg = 0;
            while(iseg < nseg_fpath):
                
                # Get the current segment.
                seg = final_path[iseg];
                if(debug > 1): print "Got segment {0}".format(seg);
                
                # Find which endpoint matches the last segment.
                e1 = -1; e1_swp = False;
                if(iseg == 0):
                    e1 = -2;
                else:
                    lseg = final_path[iseg-1];
                    if(seg.inode == lseg.inode or seg.inode == lseg.fnode):
                        e1 = seg.inode;
                    if(seg.fnode == lseg.inode or seg.fnode == lseg.fnode):
                        
                        # If we have already assigned e1, don't reassign, but set the swap variable.
                        if(e1 == seg.inode):
                            e1_swp = True;
                        else:
                            e1 = seg.fnode;
                        
                # Find the endpoint which matches the next segment.
                e2 = -1;
                if(iseg == nseg_fpath-1):
                    e2 = -2;
                else:
                    nseg = final_path[iseg+1];
                    if(seg.inode == nseg.inode or seg.inode == nseg.fnode):
                        e2 = seg.inode;
                        
                        # Handle node clashes.
                        if(e1 == seg.inode):
                            
                            if(debug > 1): print "WARNING: node clash, both e1 and e2 assigned to seg.inode = {0}".format(seg.inode);
                            
                            # If e1 was already assigned to seg.inode, and we can possibly assign e2 to seg.fnode, assign it
                            #  to avoid assigning both to the same node.
                            if(seg.fnode == nseg.inode or seg.fnode == nseg.fnode):
                                e2 = seg.fnode;
                                if(debug > 1): print "WARNING: was able to reassign e2 to seg.fnode = {0}".format(seg.fnode);
                            # Otherwise, if e1 was already assigned to seg.inode, but could have been assigned to seg.fnode, assign it
                            #  to avoid assigning both to the same node.
                            elif(e1_swp):
                                e1 = seg.fnode;
                                if(debug > 1): print "WARNING: was able to reassign e1 to seg.fnode = {0}".format(seg.fnode);
                            # Otherwise, we have a problem because e1 cannot possibly
                            #  be reassigned to fnode, and neither can e2.
                            else:
                                print "ERROR: segment ordering error: e1 = {0}, e2 = {1}, seg = {2}".format(e1,e2,seg);
                            
                    elif(seg.fnode == nseg.inode or seg.fnode == nseg.fnode):
                        e2 = seg.fnode;
                        
                        # Handle node clashes.
                        if(e1 == seg.fnode):
                            
                            if(debug > 1): print "WARNING: node clash, both e1 and e2 assigned to seg.fnode = {0}".format(seg.fnode);
                        
                            # If e1 was already assigned to seg.fnode, and we can possibly assign e2 to seg.inode, assign it
                            #  to avoid assigning both to the same node.
                            if(seg.inode == nseg.inode or seg.inode == nseg.fnode):
                                e2 = seg.inode;
                                if(debug > 1): print "WARNING: was able to reassign e2 to seg.inode = {0}".format(seg.inode);
                            # Otherwise, we have a problem because e1 cannot possibly
                            #  be assigned to inode and e2 cannot either; both are left at fnode.
                            else:
                                print "ERROR: segment ordering error: e1 = {0}, e2 = {1}, seg = {2}".format(e1,e2,seg);
                        
                if(debug > 1): print "-- We have e1 = {0}, e2 = {1}".format(e1,e2);
                        
                # Ensure we have found a correct combination of e1 and e2.
                if((e1 == seg.inode and e2 == seg.fnode) or (e1 == -2 and e2 == seg.fnode) or (e1 == seg.inode and e2 == -2)):
                    for nd in seg.path: ftrk.append(nd);
                elif((e1 == seg.fnode and e2 == seg.inode) or (e1 == seg.fnode and e2 == -2) or (e1 == -2 and e2 == seg.inode)):
                    ind = len(seg.path)-1;
                    while(ind >= 0): 
                        ftrk.append(seg.path[ind]);
                        ind -= 1;
                else:
                    # Note that if one does not include the values of at least e1 in this print statement, there occasionally is an error
                    #  in the code.
                    print "ERROR: segment ordering error: e1 = {0}, e2 = {1}, seg = {2}".format(e1,e2,seg);
                iseg += 1;
        
        # Set up the track to be returned.
        return_trk = [];
        for hnode in ftrk:
            return_trk.append(hList[hnode.id]);
        
        return (return_trk, segments_c);

    # Finds the segments using a minimum spanning tree.
    def find_segments_mst(self,mst,adjMat,xpos,ypos,zpos,nodesMarked=False):
        
        # Get the nodes of the MST.
        mst_nodes = nx.nodes(mst); #nx.topological_sort(mst);
        
        # Find a node with only one neighbor in the tree to use as the starting point.
        # If several are found, by default use the one with the lowest ID.
        snodes = [];
        for nd in mst_nodes:
            nn = mst.neighbors(nd);
            if(len(nn) == 1):
                snodes.append(nd);
        if(len(snodes) == 0):
            snode = mst_nodes[0];
        else:
            if(debug > 0): print "Found {0} starting nodes.".format(len(snodes));
            i_snode = rd.randint(0,len(snodes)-1);
            snode = snodes[i_snode];
            #snode = snodes[0];
            #for sn in snodes:
            #    if(sn.id > snode.id):
            #        snode = sn;
        if(debug > 1): print "Set to starting node {0}".format(snode.id);
        snode.type = 1;
            
        if(debug > 0): print "Found node {0} with 1 neighbor.".format(snode);
            
        # Traverse the MST, finding all segments in the tree.
        segments = []; currpth = [];
        (fnode, dist) = self.traverse_mst(mst,adjMat,snode,-1,segments,currpth);
    
        # Create the initial track segment.
        nseg = TrackSegment(currpth[0],currpth[-1]);
        nseg.set_path(currpth);
        nseg.set_len(self.path_length(mst,adjMat,currpth));
        segments.append(nseg);
    
        return segments;
        
    # Traverses the MST, adding segments to the list as they are found.
    def traverse_mst(self,mst,adjMat,node,prev,segments,currpth):
        
        if(debug > 0): print "[traverse_mst]: node = {0}, prev = {1}".format(node,prev);
        
        # Get all neighboring nodes.
        nnodes = mst.neighbors(node);
        
        # If multiple successors, this node is a type 2 node, and the proceeding segments must be obtained.
        if(len(nnodes) > 2):
    
            if(debug > 1): print "-- Multiple successors";
            node.type = 2;
    
            num_seg = 0;
            ls_fnode = -1; ls_dist = -1.0;  # save "last segment" variables
            for nbr in nnodes:
                if(nbr != prev):    
                    
                    # Traverse in the direction of this neighbor.
                    currpth2 = [];
                    (fnode, dist) = self.traverse_mst(mst,adjMat,nbr,node,segments,currpth2);
                    currpth2.append(node);
                    
                    # For end segments longer than some given length and for inner segments, create a new segment.
                    if(dist > self.min_path_voxels or fnode.type == 2):
                        nseg = TrackSegment(currpth2[0],currpth2[-1]);
                        nseg.set_path(currpth2);
                        nseg.set_len(self.path_length(mst,adjMat,currpth2));
                        segments.append(nseg);
                        ls_fnode = fnode; ls_dist = dist;
                        num_seg += 1;
                    
            # If we have no segments, consider this an end node.
            if(num_seg == 0):
                if(debug > 1): print "-- Found end node";
                node.type = 1;
                currpth.append(node);
                return (node,0.0);
            # If we only have 1 segment, do not consider this a segment end.
            elif(num_seg == 1):
                if(debug > 1): print "-- Not enough subsegments; continuing as one continuous segment";
                
                # This segment should not go in the segment array separately, as it is part of the current segment.
                nseg = segments.pop();
                for pnode in nseg.path: currpth.append(pnode);
                #currpth.append(node);
                return (ls_fnode,ls_dist+1.0);
            
            # Otherwise, this is an inner node.
            if(debug > 1): print "-- Found inner node";
            currpth.append(node);
            return (node,0.0);
    
        # Exactly one successor.        
        elif(len(nnodes) == 2 or (len(nnodes) == 1 and prev == -1)):
            
            # Select the neighbor that is not the predecessor.
            nextnd = -1;
            for nbr in nnodes:
                if(nbr != prev):
                    nextnd = nbr;
                    break;
            if(debug > 1): print "-- Single successor, chose neighbor {0}".format(nextnd);
            
            # Continue the traversal in the direction of this neighbor.
            (fnode,dist) = self.traverse_mst(mst,adjMat,nextnd,node,segments,currpth);
            currpth.append(node);
            return (fnode,dist+1.0);
            
        # Error: node is completely isolated.
        elif(len(nnodes) < 1):
            print "ERROR: node with no neighbors";
            
        # No successors (end of path).
        else:
            
            if(debug > 1): print "-- No successors; end of segment";
            
            # No connections unaccounted for.
            node.type = 1;
            currpth.append(node);
            return (node,0.0);
          
    # Determines whether any node in path "pth" coincides with any nodes in the paths
    #  of the segments in the specified list "segments," with the exception of the
    #  two endpoints n1 and n2.
    def is_overlapped(self,pth,n1,n2,segments):
        
        # Return True if the path only consists of 2 nodes (two endpoints), as this
        #  means that the segments are already joined.
        if(len(pth) < 2):
            return True;
    
        # Iterate through the segments and find any overlapping nodes.
        for seg in segments:
            noverlap = 0;
            for pnode in pth:
                for snode in seg.path:
                    if(pnode == snode and pnode != n1 and pnode != n2):
                        noverlap += 1;
            if(noverlap > self.path_overlap_tol):
                return True;
    
        # No coincidence was found in any segment.
        return False;
          
    # Attempts to find a path in the specified graph from the given node to the 
    #  endpoints of all segments in the given segments array, ensuring that no 
    #  nodes in the path coincide with any nodes of any segment except for the 
    #  endpoint of the segments which it connects and the number of tolerance
    #  nodes specified.
    #
    # If a path is found, that path is returned as a segment.
    def match_to_segments(self,node,nseg,segments,gtrk):
        
        if(debug > 2): print "Attempting to find additional paths from node {0}".format(node);
        
        # Iterate through segments, determining if this node is already connected
        #  to a segment.
        for seg in segments:
            
            # Skip the same segment.
            if(seg == nseg): continue;
            
            # Get the endpoints.
            ep1 = seg.inode;
            ep2 = seg.fnode;
            
            if(node == ep1 or node == ep2):
                if(debug > 1): print "Already connected to either {0} or {1}".format(ep1,ep2);
                return 0;
        
        # Iterate through segments, attempting to match the node to an endpoint.
        for seg in segments:
            
            # Get the endpoints.
            ep1 = seg.inode;
            ep2 = seg.fnode;
            
            # Get the paths to the endpoints.
            pth1 = nx.shortest_path(gtrk,node,ep1);
            pth2 = nx.shortest_path(gtrk,node,ep2);
    
            # Ensure no overlap with any other nodes in any segments.
            ovr1 = self.is_overlapped(pth1,node,ep1,segments);
            ovr2 = self.is_overlapped(pth2,node,ep2,segments);
            
            # Return the path as a segment if one is found.
            if(not ovr1 and not ovr2):
                print "ERROR, found non-overlapping paths from both endpoints.";
                return 0;
            elif(not ovr1):
                if(debug > 2): print "Found additional path from {0} to {1}".format(pth1[0],pth1[-1]);
                # Do not allow a path with the same beginning and endpoints
                #  (this could happen for a 2-node path where 1 endpoint is previously
                #  unconnected to others but the other is connected).  Since the 2 nodes
                #  are endpoints, the path would pass as valid.
                if((pth1[0] == nseg.inode and pth1[-1] == nseg.fnode) or (pth1[-1] == nseg.inode and pth1[0] == nseg.fnode)):
                    if(debug > 1): print "Skipping segment with the same start and endpoints as an existing segment.";
                    return 0;
                seg1 = TrackSegment(pth1[0],pth1[-1]);
                seg1.set_path(pth1);
                return seg1;
            elif(not ovr2):
                if(debug > 2): print "Found additional path from {0} to {1}".format(pth2[0],pth2[-1]);
                # Do not allow a path with the same beginning and endpoints
                #  (this could happen for a 2-node path where 1 endpoint is previously
                #  unconnected to others but the other is connected).  Since the 2 nodes
                #  are endpoints, the path would pass as valid.
                if((pth2[0] == nseg.inode and pth2[-1] == nseg.fnode) or (pth2[-1] == nseg.inode and pth2[0] == nseg.fnode)):
                    if(debug > 1): print "Skipping segment with the same start and endpoints as an existing segment.";
                    return 0;
                seg2 = TrackSegment(pth2[0],pth2[-1]);
                seg2.set_path(pth2);
                return seg2;
            
        # Return 0 if no path is found.
        return 0;
    
    # Computes the path length of the given list of nodes on the given graph.
    def path_length(self,grph,adjMat,nodes):
    
        plen = 0.0;    
        edges = grph.edges(nodes);
        for ed in edges:
    
            plen += adjMat[ed[0].id][ed[1].id];
            
        return plen;
        
    # Returns the path length of a list of segments.
    def path_length_seg(self,slist):
    
        plen = 0.;    
        for seg in slist:
            plen += seg.length;
    
        return plen;
        
    # Connect two segments
    def connect_segments(self,mst,adjMat,seg1,seg2,segments):
        
        n1i = seg1.inode; n1f = seg1.fnode;
        n2i = seg2.inode; n2f = seg2.fnode;
        if(debug > 1): print "[connect_segments] seg1 from {0} to {1}, seg2 from {2} to {3}".format(n1i,n1f,n2i,n2f);
    
        # Find the uncommon nodes.
        if(n1i == n2i or n1i == n2f): ncommon = n1i;
        elif(n1f == n2i or n1f == n2f): ncommon = n1f;
        else:
            print "ERROR: attempting to connect two segments that do not share a common node!";
            return 0;
            
        if(ncommon == n1f): n1 = n1i;
        else: n1 = n1f;
        if(ncommon == n2f): n2 = n2i;
        else: n2 = n2f;
        
        # If the "uncommon" nodes are also equal, we have a loop.  In this case,
        #  we should attempt to keep the loop vertex as the one that connects to
        #  at least 1 other segment.
        if(n1 == n2):
            if(debug > 1): print "[connect_segments] Found a loop"
            
            # Find the endpoint at which at least 1 other segment is connected,
            #  and ensure this is treated as the uncommon node.
            swapCommon = False;
            for ss in segments:
                ep1 = ss.inode; ep2 = ss.fnode;
                if(ep1 == ncommon or ep2 == ncommon):
                    swapCommon = True;
                    break;
        
            # Swap n1 (== n2) and the common node if necessary.
            if(swapCommon):
                n1 = ncommon;
                ncommon = n2;
                n2 = n1;
        
        # Make the connecting segment n1 to n2.
        if(debug > 1): print "[connect_segments] creating segment connecting {0} to {1}".format(n1,n2);
        sconn = TrackSegment(n1,n2);
        pth1 = seg1.path;
        pth2 = seg2.path;
        if(debug > 1): print "-- Path 1 contains nodes from {0} to {1}; Path 2 contains nodes from {2} to {3}".format(pth1[0],pth1[-1],pth2[0],pth2[-1]);
    
        # Make sure we have the correct orientation for the individual segment paths.    
        if(n1 == pth1[-1]): pth1.reverse();
        elif(n1 != pth1[0]):
            print "ERROR: path for segment 1 ({0} to {1}) does not contain n1 = {2}".format(seg1.inode,seg1.fnode,n1);
            print pth1;
            return 0;
        if(n2 == pth2[0]): pth2.reverse();
        elif(n2 != pth2[-1]):
            print "ERROR: path for segment 2 ({0} to {1}) does not contain n2 = {2}".format(seg2.inode,seg2.fnode,n2);
            print pth2;
            return 0;
            
        # Make the connected path.
        cpath = [];
        for nn1 in pth1:
            cpath.append(nn1);
        for nn2 in pth2:
            cpath.append(nn2);
    
        # Assign the path to the segment.
        sconn.set_path(cpath);
        sconn.set_len(self.path_length(mst,adjMat,cpath));
    
        return sconn;
        
    # Connect the list of segments.
    def connect_segment_in_list(self,mst,adjMat,seg,segments):
        
        if(debug > 1): print "[connect_segment_in_list] Connecting segment {0}".format(seg);
    
        # Keep track of whether a connection was made.
        connection = False;
    
        # Don't make any connections if the given segment is a loop.
        #if(seg.inode == seg.fnode):  
        #    return (connection,segments);
        
        # Final list of connected segments.
        segments_f = [];
        
        # Attempt to find connections and make them as they are found.
        scurr = seg;
        for ss in segments:
            
            # Ignore the same segment.
            if(ss == seg): continue;
            
            # Connect the segment to the current segment if an endpoint matches and there is no third segment with the
            #  same endpoint.
            make_connection = False;
            if(scurr.inode == ss.inode or scurr.inode == ss.fnode):
                make_connection = True;
                # Ensure there is no third segment connected at inode of scurr.
                for s2 in segments:
                    if((s2.inode == scurr.inode or s2.fnode == scurr.inode) and s2 != scurr and s2 != ss):
                        make_connection = False;
            if(scurr.fnode == ss.inode or scurr.fnode == ss.fnode):
                make_connection = True;
                # Ensure there is no third segment connected at fnode of scurr.
                for s2 in segments:
                    if((s2.inode == scurr.fnode or s2.fnode == scurr.fnode) and s2 != scurr and s2 != ss):
                        make_connection = False;            
                        
            # If this segment is a loop, do not make the connection.
            #if(ss.inode == ss.fnode):
            #    make_connection = False;
                        
            # Make the connection if this should be the case.
            if(make_connection):
                sconn = self.connect_segments(mst,adjMat,scurr,ss,segments);
                scurr = sconn;
                if(debug > 1): print "-- Connection found with segment {0}".format(ss);
                connection = True;
            else:
                segments_f.append(ss);
                
        # Append the connected segment.
        segments_f.append(scurr);
        
        return (connection,segments_f);
        
    # Compute the longest track possible from the segments in slist which:
    # - starts with sseg
    # - visits every segment only once
    def compute_longest_track(self,slist,sseg,forward,currpth):
        
        # If the starting segment is in the current path, we have reached the
        #  end of the path - return just the current path.
        in_curr_path = False;
        for pseg in currpth:
            if(sseg == pseg):
                in_curr_path = True;
                break;
                
        if(in_curr_path):
            if(debug > 1): print "-- Segment {0} is in current path of length {1}, exiting...".format(sseg,len(currpth));
            return currpth;
                
        # If not, place the starting segment in the current path.
        currpth.append(sseg);
        
        # If we are looking for a forward traversal, the end of the segment will
        #  be endpoint 2 (fnode).  Look for segments that connect to this endpoint
        #  and traverse them in the correct direction.
        if(forward):
            ep = sseg.fnode;
        # Otherwise the end of the segment will be endpoint 1 (inode).
        else:
            ep = sseg.inode;
            
        fpath = currpth; lpath = self.path_length_seg(currpth);
        for seg in slist:
            
            # Ignore the current segment.
            if(seg == sseg):
                continue;
            
            e1 = seg.inode; e2 = seg.fnode;
            currpth2 = [];
            for pth1 in currpth: currpth2.append(pth1);
            
            # Traverse the next segment in the forward direction if its
            #  first endpoint is connected to the end of the current segment.
            if(e1 == ep):
                if(debug > 1): print "Found match of endpoint 1 of segment {0}".format(seg);
                currpth2 = self.compute_longest_track(slist,seg,True,currpth2);
                    
            # Traverse the next segment in the reverse direction if its second
            #  endpoint is connected to the end of the current segment.
            elif(e2 == ep):
                if(debug > 1): print "Found match of endpoint 2 of segment {0}".format(seg);
                currpth2 = self.compute_longest_track(slist,seg,False,currpth2);
                
            # Update the longest path.
            lpath2 = self.path_length_seg(currpth2);
            if(lpath2 > lpath):
                lpath = lpath2;
                fpath = currpth2;
                
        # Return the final longest path found.
        return fpath;
