"""
convert -verbose -density 150 -trim sigvsb_all_goodextonly_noblobordering.pdf -quality 100 -sharpen 0x1.0 sigvsb_all_goodextonly_noblobordering.png

plttrk3d.py

Plots tracks in 3d using mayavi

@author: josh
"""

from math import *
from mpl_toolkits.mplot3d import Axes3D

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mayavi.mlab as mlab
import random as rd
import numpy as np
import os

# Options
single_elec = 0;
print_trks = False;
plot_trks = True;
voxelsize = 2;
debug = 1;
npavg = 10;
grdcol = 0.98;

if(single_elec):
    evt_list = [34, 62, 194, 204, 285, 299, 358, 372, 551, 631, 825, 969, 1912, 1926, 1949];
    evt_type = "BG";
    trk_name = "nmagse2";
    dat_base = "/Users/jrenner/IFIC/nmag_runs/nmag2/centella/selected_events/single_electron";
    plt_base = "/Users/jrenner/IFIC/nmag_runs/nmag2/centella/tracks/tracks_se";
else:
    evt_list = [39, 119, 135, 146, 192, 212, 271, 288, 412, 857];
    #evt_list = [309, 402];
    evt_type = "$\\beta\\beta$";
    trk_name = "nmagbb2";
    dat_base = "/Users/jrenner/IFIC/nmag_runs/nmag2/centella/selected_events/double_beta";
    plt_base = "/Users/jrenner/IFIC/nmag_runs/nmag2/centella/tracks/tracks_bb";
        
# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for evt in evt_list:

    # -------------------------------------------------------------------------
    # Read in the voxelized and MC tracks
    # -------------------------------------------------------------------------
    vtrk_file = "{0}/voxels_trk_{1}.dat".format(dat_base,evt);
    rtrk_file = "{0}/voxels_reconstructed_trk_{1}.dat".format(dat_base,evt);
    mctrk_file = "{0}/mctruehits_trk_{1}.dat".format(dat_base,evt);

    # If no file exists for this event, continue to the next.
    if(not os.path.isfile(vtrk_file) or not os.path.isfile(mctrk_file)):
        continue;
        
    # Read the voxelized track.
    trktbl = np.loadtxt(vtrk_file);
    vtrk_ID_temp = trktbl[:,0];
    vtrk_x = trktbl[:,1];
    vtrk_y = trktbl[:,2];
    vtrk_z = trktbl[:,3];
    vtrk_E = trktbl[:,4];
    vtrk_mp = trktbl[:,5];
    if(debug > 0): print "Found {0} voxels".format(len(vtrk_ID_temp));

    # Convert to integer IDs.
    vtrk_ID = [];
    for vid in vtrk_ID_temp:
        vtrk_ID.append(int(vid));

    # Read the reconstructed track.
    rtrktbl = np.loadtxt(rtrk_file);
    rtrk_ID_temp = rtrktbl[:,0];
    rtrk_x = rtrktbl[:,1];
    rtrk_y = rtrktbl[:,2];
    rtrk_z = rtrktbl[:,3];
    rtrk_E = rtrktbl[:,4];
    if(debug > 0): print "Found {0} voxels".format(len(rtrk_ID_temp));

    # Convert to integer IDs.
    rtrk_ID = [];
    for rid in rtrk_ID_temp:
        rtrk_ID.append(int(rid));
    
    # Read the MC track.
    mctrktbl = np.loadtxt(mctrk_file);
    mctrk_ID_temp = mctrktbl[:,0];
    mctrk_x = mctrktbl[:,1];
    mctrk_y = mctrktbl[:,2];
    mctrk_z = mctrktbl[:,3];
    mctrk_E = mctrktbl[:,4];
    if(debug > 0): print "Found {0} voxels".format(len(mctrk_ID_temp));
    
    # Convert to integer IDs.
    mctrk_ID = [];
    for mid in mctrk_ID_temp:
        mctrk_ID.append(int(mid));
    

    ## ------------------------------------------------------------------------
    ## Plot the voxelized track.
    ## ------------------------------------------------------------------------
    fig = mlab.figure(bgcolor=(1,1,1),size=(250,250));
    
    vtrk_col = [];
    for vE in vtrk_E:
        vtrk_col.append(vE);
    
    # Prepare the color map.
    norm = mpl.colors.Normalize(vmin=min(vtrk_col), vmax=max(vtrk_col));
    cmap = cm.rainbow
    rcm = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    # Plot the voxels.
    col0 = rcm.to_rgba(vtrk_col[0]);
    vtrk_col = np.roll(vtrk_col,-1);
    vtrk_col[-1] = vtrk_col[-2];
    mlab.points3d(vtrk_x[0],vtrk_y[0],vtrk_z[0],scale_factor=voxelsize/1000.,mode='cube',color=col0[0:3]);
    for xval,yval,zval,cval in zip(vtrk_x,vtrk_y,vtrk_z,vtrk_col):
        col = rcm.to_rgba(cval);
        mlab.points3d([xval],[yval],[zval],scale_factor=voxelsize,mode='cube',color=col[0:3]); #colormap="gist_rainbow",scale_factor=1.5,scale_mode='none');
    #mlab.axes(color=(0,0,0),nb_labels=10,x_axis_visibility=True);
    #mlab.xlabel("x (mm)"); #,ylabel='y (mm)',zlabel='z (mm)');
    mlab.savefig("{0}/voxels/3dmesh/vtrk_3d_{1}_{2}.obj".format(plt_base,trk_name,evt),figure=fig);
    #mlab.show();
    
    fig = plt.figure(1);
    fig.set_figheight(5.0);
    fig.set_figwidth(7.5);
    
    ax1 = fig.add_subplot(111, projection='3d');
    vtrk_E_arr = np.array(vtrk_E)
    s1 = ax1.scatter(vtrk_x,vtrk_y,vtrk_z,marker='o',s=30,linewidth=0.2,c=vtrk_E_arr*1000.,cmap=plt.get_cmap('rainbow'),vmin=min(vtrk_E)*1000.,vmax=max(vtrk_E)*1000.);
    s1.set_edgecolors = s1.set_facecolors = lambda *args:None;  # this disables automatic setting of alpha relative of distance to camera
    ax1.set_xlabel("x (mm)");
    ax1.set_ylabel("y (mm)");
    ax1.set_zlabel("z (mm)");
    ax1.grid(True);

    ax1.w_xaxis.set_pane_color((1.0,1.0,1.0,1.0));
    ax1.w_yaxis.set_pane_color((1.0,1.0,1.0,1.0));
    ax1.w_zaxis.set_pane_color((1.0,1.0,1.0,1.0));
    ax1.w_xaxis._axinfo.update({'grid' : {'color': (grdcol, grdcol, grdcol, 1)}});
    ax1.w_yaxis._axinfo.update({'grid' : {'color': (grdcol, grdcol, grdcol, 1)}});
    ax1.w_zaxis._axinfo.update({'grid' : {'color': (grdcol, grdcol, grdcol, 1)}});

    lb_x = ax1.get_xticklabels();
    lb_y = ax1.get_yticklabels();
    lb_z = ax1.get_zticklabels();
    for lb in (lb_x + lb_y + lb_z):
        lb.set_fontsize(8);

    cb1 = plt.colorbar(s1);
    cb1.set_label('Hit Energy (keV)');
    plt.title("{0} Event {1}, voxelized".format(evt_type,evt),fontsize=22);
    plt.savefig("{0}/voxels/3dplt/plt_vtrk_3d_{1}_{2}.pdf".format(plt_base,trk_name,evt), bbox_inches='tight');
    plt.close();
    
    # Create the x-y projection.
    fig = plt.figure(2);
    fig.set_figheight(5.0);
    fig.set_figwidth(7.5);
    
    ax2 = fig.add_subplot(111);
    ax2.plot(vtrk_x,vtrk_y,'.',color='black');
    #ax2.plot(vtrk_x,vtrk_y,'-',color='black');
    ax2.set_xlabel("x (mm)");
    ax2.set_ylabel("y (mm)");
    plt.savefig("{0}/voxels/xyproj/plt_vtrk_xy_{1}_{2}.pdf".format(plt_base,trk_name,evt), bbox_inches='tight');
    plt.close();

    fig = plt.figure(3);
    fig.set_figheight(5.0);
    fig.set_figwidth(7.5);
    
    ax3 = fig.add_subplot(111);
    ax3.plot(vtrk_x,vtrk_z,'.',color='black');
    #ax3.plot(vtrk_x,vtrk_z,'-',color='black');
    ax3.set_xlabel("x (mm)");
    ax3.set_ylabel("z (mm)");
    plt.savefig("{0}/voxels/xzproj/plt_vtrk_xz_{1}_{2}.pdf".format(plt_base,trk_name,evt), bbox_inches='tight');
    plt.close();

    fig = plt.figure(4);
    fig.set_figheight(5.0);
    fig.set_figwidth(7.5);
    
    ax4 = fig.add_subplot(111);
    ax4.plot(vtrk_y,vtrk_z,'.',color='black');
    #ax4.plot(vtrk_y,vtrk_z,'-',color='black');
    ax4.set_xlabel("y (mm)");
    ax4.set_ylabel("z (mm)");
    plt.savefig("{0}/voxels/yzproj/plt_vtrk_yz_{1}_{2}.pdf".format(plt_base,trk_name,evt), bbox_inches='tight');
    plt.close();
    
    ## ------------------------------------------------------------------------
    ## Plot the reconstructed track.
    ## ------------------------------------------------------------------------
    rtrk_col = [];
    rid = 0;
    for rid0 in rtrk_ID:
        rtrk_col.append(rid);
        rid += 1;
    
    fig = mlab.figure(bgcolor=(1,1,1),size=(250,250));
    
    # Prepare the color map.
    norm = mpl.colors.Normalize(vmin=0, vmax=max(rtrk_col))
    cmap = cm.rainbow
    rcm = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    # Plot the voxels (note, need to plot a dummy point since after saving to .obj, the colors specified for a point will be
    #  used in the next plot command).
    col0 = rcm.to_rgba(rtrk_col[0]);
    rtrk_col = np.roll(rtrk_col,-1);
    rtrk_col[-1] = rtrk_col[-2];
    mlab.points3d(rtrk_x[0],rtrk_y[0],rtrk_z[0],scale_factor=voxelsize/1000.,mode='cube',color=col0[0:3]);
    for xval,yval,zval,cval in zip(rtrk_x,rtrk_y,rtrk_z,rtrk_col):
        col = rcm.to_rgba(cval);
        mlab.points3d([xval],[yval],[zval],scale_factor=voxelsize,mode='cube',color=col[0:3]); #colormap="gist_rainbow",scale_factor=1.5,scale_mode='none');
    mlab.points3d(rtrk_x[-1],rtrk_y[-1],rtrk_z[-1],scale_factor=voxelsize/1000.,mode='cube',color=(0,0,0));
    mc_x0 = mctrk_x[0]; mc_y0 = mctrk_y[0]; mc_z0 = mctrk_z[0];
    mc_xf = mctrk_x[-1]; mc_yf = mctrk_y[-1]; mc_zf = mctrk_z[-1];
    mlab.points3d([mc_xf-5],[mc_yf],[mc_zf],scale_factor=3*voxelsize,mode='arrow',color=(0,1,0))
    mlab.points3d([mc_x0-5],[mc_y0],[mc_z0],scale_factor=3*voxelsize,mode='arrow',color=(0,0,0))
    
    # Add the MC truth extremes.
    #mlab.axes(color=(0,0,0),nb_labels=10,x_axis_visibility=True);
    #mlab.xlabel("x (mm)"); #,ylabel='y (mm)',zlabel='z (mm)');
    mlab.savefig("{0}/reconstructed/3dmesh/rtrk_3d_{1}_{2}.obj".format(plt_base,trk_name,evt),figure=fig);
    #mlab.show();
    
    fig = plt.figure(1);
    fig.set_figheight(5.0);
    fig.set_figwidth(7.5);
    
    ax1 = fig.add_subplot(111, projection='3d');
    rtrk_col_arr = np.array(rtrk_col);
    s1 = ax1.scatter(rtrk_x,rtrk_y,rtrk_z,marker='o',s=30,linewidth=0.2,c=rtrk_col_arr,cmap=plt.get_cmap('rainbow'),vmin=0.0,vmax=max(rtrk_col));
    s1.set_edgecolors = s1.set_facecolors = lambda *args:None;  # this disables automatic setting of alpha relative of distance to camera
    ax1.set_xlabel("x (mm)");
    ax1.set_ylabel("y (mm)");
    ax1.set_zlabel("z (mm)");
    ax1.grid(True);

    ax1.w_xaxis.set_pane_color((1.0,1.0,1.0,1.0));
    ax1.w_yaxis.set_pane_color((1.0,1.0,1.0,1.0));
    ax1.w_zaxis.set_pane_color((1.0,1.0,1.0,1.0));
    ax1.w_xaxis._axinfo.update({'grid' : {'color': (grdcol, grdcol, grdcol, 1)}});
    ax1.w_yaxis._axinfo.update({'grid' : {'color': (grdcol, grdcol, grdcol, 1)}});
    ax1.w_zaxis._axinfo.update({'grid' : {'color': (grdcol, grdcol, grdcol, 1)}});

    lb_x = ax1.get_xticklabels();
    lb_y = ax1.get_yticklabels();
    lb_z = ax1.get_zticklabels();
    for lb in (lb_x + lb_y + lb_z):
        lb.set_fontsize(8);

    cb1 = plt.colorbar(s1);
    cb1.set_label('Hit number');
    plt.title("{0} Event {1}, reconstructed".format(evt_type,evt),fontsize=22);
    plt.savefig("{0}/reconstructed/3dplt/plt_rtrk_3d_{1}_{2}.pdf".format(plt_base,trk_name,evt), bbox_inches='tight');
    plt.close();
    
    # Create the x-y projection.
    fig = plt.figure(2);
    fig.set_figheight(5.0);
    fig.set_figwidth(7.5);
    
    ax2 = fig.add_subplot(111);
    ax2.plot(rtrk_x,rtrk_y,'.',color='black');
    #ax2.plot(vtrk_x,vtrk_y,'-',color='black');
    ax2.set_xlabel("x (mm)");
    ax2.set_ylabel("y (mm)");
    plt.savefig("{0}/reconstructed/xyproj/plt_rtrk_xy_{1}_{2}.pdf".format(plt_base,trk_name,evt), bbox_inches='tight');
    plt.close();

    fig = plt.figure(3);
    fig.set_figheight(5.0);
    fig.set_figwidth(7.5);
    
    ax3 = fig.add_subplot(111);
    ax3.plot(rtrk_x,rtrk_z,'.',color='black');
    #ax3.plot(vtrk_x,vtrk_z,'-',color='black');
    ax3.set_xlabel("x (mm)");
    ax3.set_ylabel("z (mm)");
    plt.savefig("{0}/reconstructed/xzproj/plt_rtrk_xz_{1}_{2}.pdf".format(plt_base,trk_name,evt), bbox_inches='tight');
    plt.close();

    fig = plt.figure(4);
    fig.set_figheight(5.0);
    fig.set_figwidth(7.5);
    
    ax4 = fig.add_subplot(111);
    ax4.plot(rtrk_y,rtrk_z,'.',color='black');
    #ax4.plot(vtrk_y,vtrk_z,'-',color='black');
    ax4.set_xlabel("y (mm)");
    ax4.set_ylabel("z (mm)");
    plt.savefig("{0}/reconstructed/yzproj/plt_rtrk_yz_{1}_{2}.pdf".format(plt_base,trk_name,evt), bbox_inches='tight');
    plt.close();
    
    ## ------------------------------------------------------------------------
    ## Plot the MC truth track.
    ## ------------------------------------------------------------------------
    mctrk_col = []; mid = 0;
    for mid0 in mctrk_ID:
        mctrk_col.append(mid);
        mid += 1;
    
    fig = mlab.figure(bgcolor=(1,1,1),size=(250,250));
    
    # Prepare the color map.
    norm = mpl.colors.Normalize(vmin=0, vmax=len(mctrk_ID))
    cmap = cm.rainbow
    rcm = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    # Plot the voxels.
    col0 = rcm.to_rgba(mctrk_col[0]);
    mctrk_col = np.roll(mctrk_col,-1);
    mctrk_col[-1] = mctrk_col[-2];
    mlab.points3d(mctrk_x[0],mctrk_y[0],mctrk_z[0],scale_factor=voxelsize/1000.,mode='cube',color=col0[0:3]);
    for xval,yval,zval,cval in zip(mctrk_x,mctrk_y,mctrk_z,mctrk_col):
        col = rcm.to_rgba(cval);
        mlab.points3d([xval],[yval],[zval],scale_factor=1.0,mode='sphere',color=col[0:3]); #colormap="gist_rainbow",scale_factor=1.5,scale_mode='none');
    #mlab.axes(color=(0,0,0),nb_labels=10,x_axis_visibility=True);
    #mlab.xlabel("x (mm)"); #,ylabel='y (mm)',zlabel='z (mm)');
    mlab.savefig("{0}/mctruth/3dmesh/mctrk_3d_{1}_{2}.obj".format(plt_base,trk_name,evt),figure=fig);
    #mlab.show();
    
    fig = plt.figure(1);
    fig.set_figheight(5.0);
    fig.set_figwidth(7.5);
    
    ax1 = fig.add_subplot(111, projection='3d');
    mctrk_ID_arr = np.array(mctrk_ID)
    s1 = ax1.scatter(mctrk_x,mctrk_y,mctrk_z,marker='o',s=30,linewidth=0.2,c=mctrk_col,cmap=plt.get_cmap('rainbow'),vmin=0.0,vmax=max(mctrk_col));
    s1.set_edgecolors = s1.set_facecolors = lambda *args:None;  # this disables automatic setting of alpha relative of distance to camera
    ax1.set_xlabel("x (mm)");
    ax1.set_ylabel("y (mm)");
    ax1.set_zlabel("z (mm)");
    ax1.grid(True);

    ax1.w_xaxis.set_pane_color((1.0,1.0,1.0,1.0));
    ax1.w_yaxis.set_pane_color((1.0,1.0,1.0,1.0));
    ax1.w_zaxis.set_pane_color((1.0,1.0,1.0,1.0));
    ax1.w_xaxis._axinfo.update({'grid' : {'color': (grdcol, grdcol, grdcol, 1)}});
    ax1.w_yaxis._axinfo.update({'grid' : {'color': (grdcol, grdcol, grdcol, 1)}});
    ax1.w_zaxis._axinfo.update({'grid' : {'color': (grdcol, grdcol, grdcol, 1)}});

    lb_x = ax1.get_xticklabels();
    lb_y = ax1.get_yticklabels();
    lb_z = ax1.get_zticklabels();
    for lb in (lb_x + lb_y + lb_z):
        lb.set_fontsize(8);

    cb1 = plt.colorbar(s1);
    cb1.set_label('Hit number');
    plt.title("{0} Event {1}, MC truth".format(evt_type,evt),fontsize=22);
    plt.savefig("{0}/mctruth/3dplt/plt_mctrk_3d_{1}_{2}.pdf".format(plt_base,trk_name,evt), bbox_inches='tight');
    plt.close();
    
    # Create the x-y projection.
    fig = plt.figure(2);
    fig.set_figheight(5.0);
    fig.set_figwidth(7.5);
    
    ax2 = fig.add_subplot(111);
    ax2.plot(mctrk_x,mctrk_y,'.',color='black');
    ax2.plot(mctrk_x,mctrk_y,'-',color='black');
    ax2.set_xlabel("x (mm)");
    ax2.set_ylabel("y (mm)");
    plt.savefig("{0}/mctruth/xyproj/plt_mctrk_xy_{1}_{2}.pdf".format(plt_base,trk_name,evt), bbox_inches='tight');
    plt.close();

    fig = plt.figure(3);
    fig.set_figheight(5.0);
    fig.set_figwidth(7.5);
    
    ax3 = fig.add_subplot(111);
    ax3.plot(mctrk_x,mctrk_z,'.',color='black');
    ax3.plot(mctrk_x,mctrk_z,'-',color='black');
    ax3.set_xlabel("x (mm)");
    ax3.set_ylabel("z (mm)");
    plt.savefig("{0}/mctruth/xzproj/plt_mctrk_xz_{1}_{2}.pdf".format(plt_base,trk_name,evt), bbox_inches='tight');
    plt.close();

    fig = plt.figure(4);
    fig.set_figheight(5.0);
    fig.set_figwidth(7.5);
    
    ax4 = fig.add_subplot(111);
    ax4.plot(mctrk_y,mctrk_z,'.',color='black');
    ax4.plot(mctrk_y,mctrk_z,'-',color='black');
    ax4.set_xlabel("y (mm)");
    ax4.set_ylabel("z (mm)");
    plt.savefig("{0}/mctruth/yzproj/plt_mctrk_yz_{1}_{2}.pdf".format(plt_base,trk_name,evt), bbox_inches='tight');
    plt.close();

    #ax1.plot(vtrk_xf,vtrk_yf,vtrk_zf,'s',color='blue');
    #ax1.plot([vtrk_x[vinit]],[vtrk_y[vinit]],[vtrk_z[vinit]],'o',color='green',markersize=15)
    #ax1.plot([vtrk_x[vfinal]],[vtrk_y[vfinal]],[vtrk_z[vfinal]],'x',color='green',markersize=15)
    #ax1.plot([mctrk_x[0]],[mctrk_y[0]],[mctrk_z[0]],'o',color='blue',markersize=15)
    #ax1.plot([mctrk_x[-1]],[mctrk_y[-1]],[mctrk_z[-1]],'x',color='blue',markersize=15)
