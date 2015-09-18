from Centella.AAlgo import AAlgo
from Centella.physical_constants import *
from math import *

from ROOT import gSystem
gSystem.Load("$GATE_DIR/lib/libGATE")
gSystem.Load("$GATE_DIR/lib/libGATEIO")
gSystem.Load("$GATE_DIR/lib/libGATEUtils")
from ROOT import gate

"""
This algorithm characterizes Background events that have passed all filters
"""


class MultiBlobFilter(AAlgo):

	############################################################
	def __init__(self, param=False, level=1, label="", **kargs):

		"""
		MultiBlobFilter Algorithm
		"""
		#self.m.log(1, 'Constructor()')

		### GENERAL STUFF
		self.name = 'MultiBlobFilter'
		#self.level = level
		AAlgo.__init__(self, param, level, self.name, 0, label, kargs)

		### PARAMETERS
		# Blob Radius
		try:
			self.blobRadius = self.doubles['blobRadius']
			self.m.log(1, "Blob Radius: %.1f mm" %(self.blobRadius))
		except KeyError:
			self.m.log(1, "WARNING!! Parameter: 'blobRadius' not defined.")
			exit(0)

		# ROI Minimum Energy
		try:
			self.blobMinE = self.doubles['blobMinE']
			self.m.log(1, "Minimum Blob Energy: %.3f MeV." %(self.blobMinE/MeV))
		except KeyError:
			self.m.log(1, "WARNING!! Parameter: 'blobMinE' not defined.")
			exit(0)

		# Use MC truth
		try:
			self.useTruth_param = self.ints['useTruth']
                        if(self.useTruth_param == 1):
                            self.useTruth = True
                        else:
                            self.useTruth = False
			self.m.log(1, "Use MC truth = {0}".format(self.useTruth))
		except KeyError:
			self.m.log(1, "WARNING!! Parameter: 'useTruth' not defined.")
			exit(0)

                # Which track ordering method to use (only appicable if not using MC truth)
                # 0: Paolina
                # 1: momentum-based
                # 2: MST
                try:
                        self.trk_omethod = self.ints['trk_omethod']
                        self.m.log(1, "Track ordering method = {0}".format(self.trk_omethod))
                except KeyError:
                        self.m.log(1, "WARNING!! Parameter: 'trk_omethod' not defined.")
                        exit(0)


	############################################################		
	def initialize(self):

		self.m.log(1, 'Initialize()')
		
		### Defining histos
		# Distance between True & Recons. extremes for events passing the filter
		histo_name = self.alabel("DistTrueRec1")
		histo_desc = "Distance True-Rec extreme 1"
		self.m.log(2, "Booking ", histo_name)
		self.m.log(3, "   Description: ", histo_desc)
		self.hman.h1(histo_name, histo_desc, 40, 0, 40) 

		# Distance between True & Recons. extremes for events passing the filter
		histo_name = self.alabel("DistTrueRec2")
		histo_desc = "Distance True-Rec extreme 2"
		self.m.log(2, "Booking ", histo_name)
		self.m.log(3, "   Description: ", histo_desc)
		self.hman.h1(histo_name, histo_desc, 40, 0, 40) 

		# Distance between True & Recons. extremes for events passing the filter
		histo_name = self.alabel("DistTrueRec")
		histo_desc = "Distance True-Rec extremes"
		self.m.log(2, "Booking ", histo_name)
		self.m.log(3, "   Description: ", histo_desc)
		self.hman.h1(histo_name, histo_desc, 40, 0, 40) 

		# Distance between True & Recons. extremes for events not passing the filter
		histo_name = self.alabel("DistTrueRec1F")
		histo_desc = "Distance True-Rec extreme 1, failed blob cut"
		self.m.log(2, "Booking ", histo_name)
		self.m.log(3, "   Description: ", histo_desc)
		self.hman.h1(histo_name, histo_desc, 40, 0, 40) 

		# Distance between True & Recons. extremes for events not passing the filter
		histo_name = self.alabel("DistTrueRec2F")
		histo_desc = "Distance True-Rec extreme 2, failed blob cut"
		self.m.log(2, "Booking ", histo_name)
		self.m.log(3, "   Description: ", histo_desc)
		self.hman.h1(histo_name, histo_desc, 40, 0, 40) 

		# Distance between True & Recons. extremes for events not passing the filter
		histo_name = self.alabel("DistTrueRecF")
		histo_desc = "Distance True-Rec extremes, failed blob cut"
		self.m.log(2, "Booking ", histo_name)
		self.m.log(3, "   Description: ", histo_desc)
		self.hman.h1(histo_name, histo_desc, 40, 0, 40) 

		# Distance between True & Recons. extremes for events not passing the filter
		histo_name = self.alabel("DistTrueRec1A")
		histo_desc = "Distance True-Rec extreme 1, before any cuts"
		self.m.log(2, "Booking ", histo_name)
		self.m.log(3, "   Description: ", histo_desc)
		self.hman.h1(histo_name, histo_desc, 40, 0, 40) 

		# Distance between True & Recons. extremes for events not passing the filter
		histo_name = self.alabel("DistTrueRec2A")
		histo_desc = "Distance True-Rec extreme 2, before any cuts"
		self.m.log(2, "Booking ", histo_name)
		self.m.log(3, "   Description: ", histo_desc)
		self.hman.h1(histo_name, histo_desc, 40, 0, 40) 

		# Distance between True & Recons. extremes for events not passing the filter
		histo_name = self.alabel("DistTrueRecA")
		histo_desc = "Distance True-Rec extremes, before any cuts"
		self.m.log(2, "Booking ", histo_name)
		self.m.log(3, "   Description: ", histo_desc)
		self.hman.h1(histo_name, histo_desc, 40, 0, 40) 

                # Energy of blob 1
                histo_name = self.alabel("EBlob1")
                histo_desc = "Blob 1 energy (MeV)"
                self.m.log(2, "Booking ", histo_name)
                self.m.log(3, "   Description: ", histo_desc)
                self.hman.h1(histo_name, histo_desc, 50, 0, 1.5)

                # Energy of blob 1
                histo_name = self.alabel("EBlob2")
                histo_desc = "Blob 2 energy (MeV)"
                self.m.log(2, "Booking ", histo_name)
                self.m.log(3, "   Description: ", histo_desc)
                self.hman.h1(histo_name, histo_desc, 50, 0, 1.5)

		### Counters:
		self.numInputEvents = 0
		self.numSwappedEvents = 0
                self.numOutputEvents = 0

		return



	############################################################
	def execute(self, event=""):

		self.m.log(2, 'Execute()')		
	
		self.numInputEvents += 1

		### Event Energy
		evtEdep = event.GetEnergy()
		self.m.log(2, "Event Edep: %.3f MeV" %(evtEdep/MeV))

                # At this point, we should only have 1 track.
                all_trks = self.event.GetTracks(gate.SIPM)
                hTrack = all_trks[0]

                # Getting the Hottest Track
                #hTrack = 0
                #maxEdep = 0
                #for track in all_trks:
                #    trackEdep = track.GetEnergy()
                #    if (trackEdep > maxEdep):
                #        maxEdep = trackEdep
                #        hTrack = track

                ############################################################################################
                ## MC truth extremes

                # Ordering extremes per euclidean distance
                ext1MCPos = ext2MCPos = 0
                MCTracks = event.GetMCTracks();

                # If Signal, True extremes are the second extremes of "primary" tracks
                if (event.GetMCEventType() == gate.BB0NU):
                    numPrimaries = 0
                    for MCTrack in MCTracks:
                        if (MCTrack.GetParticle().IsPrimary()):
                            numPrimaries += 1
                            if (numPrimaries == 1): ext1MCPos = MCTrack.GetExtremes().second.GetPosition()
                            if (numPrimaries == 2):
                                ext2MCPos = MCTrack.GetExtremes().second.GetPosition()
                                break

                # If Background, True extremes are the extremes of the Hottest TTrack
                else:
                    # Getting the hottest True Track from the Hottest Track
                    MCTrackIDs = hTrack.fetch_ivstore("mcTrackIDs")
                    maxEdep = 0
                    hMCTrack = 0
                    for MCTrack in MCTracks:
                        if MCTrack.GetID() in MCTrackIDs : 
                            eDep = MCTrack.GetEnergy()
                            if (eDep > maxEdep):
                                maxEdep = eDep
                                hMCTrack = MCTrack

                    ext1MCPos = hMCTrack.GetExtremes().first.GetPosition()
                    ext2MCPos = hMCTrack.GetExtremes().second.GetPosition()
                
                # Determine the MC truth blob energies (using Euclidean distances).
                e1x = ext1MCPos.x(); e1y = ext1MCPos.y(); e1z = ext1MCPos.z()
                e2x = ext2MCPos.x(); e2y = ext2MCPos.y(); e2z = ext2MCPos.z()
                blob1E_t = 0.; blob2E_t = 0.
                for hhit in self.event.GetMCHits():

                    xh = hhit.GetPosition().x(); yh = hhit.GetPosition().y(); zh = hhit.GetPosition().z()
                    eh = hhit.GetAmplitude()

                    d1 = sqrt((xh-e1x)**2 + (yh-e1y)**2 + (zh-e1z)**2)
                    d2 = sqrt((xh-e2x)**2 + (yh-e2y)**2 + (zh-e2z)**2)

                    if(d1 < self.blobRadius):
                        blob1E_t += eh
                    if(d2 < self.blobRadius):
                        blob2E_t += eh 

                #################################################################################
                ## Voxelized track extremes
                
                # Using Paolina-based track.
                if(self.trk_omethod == 0):
 
                    distExtFirst  = hTrack.fetch_dvstore("DistExtFirst")
                    distExtSecond = hTrack.fetch_dvstore("DistExtSecond")
    
                    # Ordering extremes per energy
                    ext1Pos = ext2Pos = 0
                    blob1E = blob2E = 0
    		
                    for hit in event.GetHits(gate.SIPM):
                        dist1 = distExtFirst[hit.GetID()]
                        if (dist1 < self.blobRadius): blob1E += hit.GetAmplitude()
    
                        dist2 = distExtSecond[hit.GetID()]
                        if (dist2 < self.blobRadius): blob2E += hit.GetAmplitude()
    
                    if (blob1E < blob2E):
                        ext1Pos = hTrack.GetExtremes().first.GetPosition()
                        ext2Pos = hTrack.GetExtremes().second.GetPosition()
                    else:
                        ext2Pos = hTrack.GetExtremes().first.GetPosition()
                        ext1Pos = hTrack.GetExtremes().second.GetPosition()

                elif(self.trk_omethod == 2):

                    # Get the extremes of the MST track.
                    hMainPath = hTrack.fetch_ivstore("MSTHits")
                    hHits = event.GetHits(gate.SIPM)
                    ext1Pos = hHits[0].GetPosition()
                    ext2Pos = hHits[-1].GetPosition()

                    # Find the distance between each hit and each extreme, and if close enough,
                    #  add its energy to the appropriate blob.
                    blob1E = blob2E = 0
                    for hit in event.GetHits(gate.SIPM):
                        dist1 = gate.distance(ext1Pos,hit.GetPosition())
                        if (dist1 < self.blobRadius): blob1E += hit.GetAmplitude()

                        dist2 = gate.distance(ext2Pos,hit.GetPosition())
                        if (dist2 < self.blobRadius): blob2E += hit.GetAmplitude() 

                    # Swap the extremes if blob2 is actually the one with less energy.
                    if (blob2E < blob1E):
                        ext2Pos = hHits[0].GetPosition()
                        ext1Pos = hHits[-1].GetPosition()
                        tempE = blob1E; blob1E = blob2E; blob2E = tempE

                # Verbosing
                self.m.log(3, 'Extreme 1 Rec. Position: (%f, %f, %f)' %(ext1Pos.x(), ext1Pos.y(), ext1Pos.z()))
                self.m.log(3, 'Extreme 2 Rec. Position: (%f, %f, %f)' %(ext2Pos.x(), ext2Pos.y(), ext2Pos.z()))

                self.m.log(3, 'Extreme 1 MC Position: (%f, %f, %f)' %(ext1MCPos.x(), ext1MCPos.y(), ext1MCPos.z()))
                self.m.log(3, 'Extreme 2 MC Position: (%f, %f, %f)' %(ext2MCPos.x(), ext2MCPos.y(), ext2MCPos.z()))

		# Matching Rec & True extremes by euclidean distance
		dist1 = dist2 = 0
		d11 = gate.distance(ext1Pos, ext1MCPos)
		d12 = gate.distance(ext1Pos, ext2MCPos)

		# Swapped extremes ??
		if (d12 < d11):
			self.m.log(2, 'Extremes Swapped')
			self.numSwappedEvents += 1
			dist1 = d12
			dist2 = gate.distance(ext2Pos, ext1MCPos)
		# Correctly matched extreme
		else:
			dist1 = d11
			dist2 = gate.distance(ext2Pos, ext2MCPos)

		self.m.log(2, 'Distance 1:', dist1)
		self.m.log(2, 'Distance 2:', dist2)


                ############################################################################
                # Apply the blob cut on either the true or voxelized blobs, as specified.

                blobCut = False
                if(self.useTruth):
                    if(blob1E_t > self.blobMinE and blob2E_t > self.blobMinE):
                        blobCut = True
                else:
                    if(blob1E > self.blobMinE and blob2E > self.blobMinE):
                        blobCut = True

                ###########################################################################
                # Filling Histograms

                self.hman.fill(self.alabel("DistTrueRecA"), dist1)
                self.hman.fill(self.alabel("DistTrueRecA"), dist2)

                self.hman.fill(self.alabel("DistTrueRec1A"), dist1)
                self.hman.fill(self.alabel("DistTrueRec2A"), dist2)

                if(self.useTruth):
                    self.hman.fill(self.alabel("EBlob1"), blob1E_t)
                    self.hman.fill(self.alabel("EBlob2"), blob2E_t)
                else:
                    self.hman.fill(self.alabel("EBlob1"), blob1E)
                    self.hman.fill(self.alabel("EBlob2"), blob2E)

                # Passed blob cut
                if(blobCut):
                    self.hman.fill(self.alabel("DistTrueRec"), dist1)
                    self.hman.fill(self.alabel("DistTrueRec"), dist2)

                    self.hman.fill(self.alabel("DistTrueRec1"), dist1)
                    self.hman.fill(self.alabel("DistTrueRec2"), dist2)

                    self.numOutputEvents += 1

                # Failed blob cut
                else:
                    self.hman.fill(self.alabel("DistTrueRecF"), dist1)
                    self.hman.fill(self.alabel("DistTrueRecF"), dist2)

                    self.hman.fill(self.alabel("DistTrueRec1F"), dist1)
                    self.hman.fill(self.alabel("DistTrueRec2F"), dist2)

		return blobCut

		

	############################################################
	def finalize(self):

		self.m.log(1, 'Finalize()')

		self.m.log(1, 'Input  Events: ', self.numInputEvents)
		self.m.log(1, 'Swapped Events: ', self.numSwappedEvents)
                self.m.log(1, 'Output Events: ', self.numOutputEvents)

		return
