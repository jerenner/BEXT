from Centella.AAlgo import AAlgo
from Centella.physical_constants import *

from ROOT import gSystem
gSystem.Load("$GATE_DIR/lib/libGATE")
gSystem.Load("$GATE_DIR/lib/libGATEIO")
gSystem.Load("$GATE_DIR/lib/libGATEUtils")
from ROOT import gate

"""
This algorithm filters events if the reconstructed extremes
are further from true extremes of a given distance
Parameters:
	blobRadius
	maxDist
"""


class goodExtremesFilter(AAlgo):

	############################################################
	def __init__(self, param=False, level=1, label="", **kargs):

		"""
		Event Characterizer Algorithm
		"""
		#self.m.log(1, 'Constructor()')

		### GENERAL STUFF
		self.name = 'goodExtremesFilter'
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

		# Maximum Distance
		try:
			self.maxDist = self.doubles['maxDist']
			self.m.log(1, "Maximum Distance: %.1f mm" %(self.maxDist))
		except KeyError:
			self.m.log(1, "WARNING!! Parameter: 'maxDist' not defined.")
			exit(0)



	############################################################		
	def initialize(self):

		self.m.log(1, 'Initialize()')
		
		### Defining histos
		# Distance between True & Recons. extreme 1
		histo_name = self.alabel("DistTrueRec1")
		histo_desc = "Distance True-Rec extreme 1"
		self.m.log(2, "Booking ", histo_name)
		self.m.log(3, "   Description: ", histo_desc)
		self.hman.h1(histo_name, histo_desc, 40, 0, 40) 

		# Distance between True & Recons. extreme 2
		histo_name = self.alabel("DistTrueRec2")
		histo_desc = "Distance True-Rec extreme 2"
		self.m.log(2, "Booking ", histo_name)
		self.m.log(3, "   Description: ", histo_desc)
		self.hman.h1(histo_name, histo_desc, 40, 0, 40) 

		### Counters
		self.numInputEvents = 0
		self.numSwappedEvents = 0
		self.numOutputEvents = 0

		return



	############################################################
	def execute(self, event=""):

		self.m.log(2, 'Execute()')		

		self.numInputEvents += 1
	
		### Getting the Hottest Track
		hTrack = 0
		maxEdep = 0
		for track in self.event.GetTracks(gate.SIPM):
			trackEdep = track.GetEnergy()
			if (trackEdep > maxEdep):
				maxEdep = trackEdep
				hTrack = track

		distExtFirst  = hTrack.fetch_dvstore("DistExtFirst")
		distExtSecond = hTrack.fetch_dvstore("DistExtSecond")


		### Getting the MC extremes
		ext1MCPos = ext2MCPos = 0
		MCTracks = event.GetMCTracks()

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

		self.m.log(2, 'Extreme 1 MC Position: (%f, %f, %f)' %(ext1MCPos.x(), ext1MCPos.y(), ext1MCPos.z()))
		self.m.log(2, 'Extreme 2 MC Position: (%f, %f, %f)' %(ext2MCPos.x(), ext2MCPos.y(), ext2MCPos.z()))


		### Getting Recons extremes
		energyStart = -1
		ext1Pos = ext2Pos = 0
		blob1E = blob2E = 0

		ext1Pos = hTrack.GetExtremes().first.GetPosition()
		ext2Pos = hTrack.GetExtremes().second.GetPosition()
		
		# Getting the energies
		for hit in hTrack.GetHits():
			dist1 = distExtFirst[hit.GetID()]
			if (dist1 < self.blobRadius): blob1E += hit.GetAmplitude()

			dist2 = distExtSecond[hit.GetID()]
			if (dist2 < self.blobRadius):	blob2E += hit.GetAmplitude()

		# Ordering & Filling Histos
		if (blob1E < blob2E):
			ext1Pos = hTrack.GetExtremes().first.GetPosition()
			ext2Pos = hTrack.GetExtremes().second.GetPosition()

		else:
			ext2Pos = hTrack.GetExtremes().first.GetPosition()
			ext1Pos = hTrack.GetExtremes().second.GetPosition()

		self.m.log(2, 'Extreme 1 Rec. Position: (%f, %f, %f)' %(ext1Pos.x(), ext1Pos.y(), ext1Pos.z()))
		self.m.log(2, 'Extreme 2 Rec. Position: (%f, %f, %f)' %(ext2Pos.x(), ext2Pos.y(), ext2Pos.z()))


		### Matching Rec & True extremes by euclidean distance
		swapped = False
		dist1 = dist2 = 0
		d11 = gate.distance(ext1Pos, ext1MCPos)
		d12 = gate.distance(ext1Pos, ext2MCPos)
		d21 = gate.distance(ext2Pos, ext1MCPos)
		d22 = gate.distance(ext2Pos, ext2MCPos)

		if ((d11+d12) < (d12+d21)):
			dist1 = d11
			dist2 = d22
		else:
			dist1 = d12
			dist2 = d21
			swapped = True 
			self.numSwappedEvents += 1
			self.m.log(2, 'Swapped extremes')


		self.m.log(2, 'Distance 1:', dist1)
		self.m.log(2, 'Distance 2:', dist2)


		# Filling Histograms
		self.hman.fill(self.alabel("DistTrueRec1"), dist1)
		self.hman.fill(self.alabel("DistTrueRec2"), dist2)

		if ((dist1 < self.maxDist) and (dist2 < self.maxDist)):
                #if ((dist1 >= self.maxDist) or (dist2 >= self.maxDist)):
			self.numOutputEvents += 1
			return True

		return False

		

	############################################################
	def finalize(self):

		self.m.log(1, 'Finalize()')

		self.m.log(1, 'Input  Events: ', self.numInputEvents)
		self.m.log(1, 'Swapped Events: ', self.numSwappedEvents)
		self.m.log(1, 'Output Events: ', self.numOutputEvents)

		self.logman["USER"].ints[self.alabel("InputEvents")] = self.numInputEvents
		self.logman["USER"].ints[self.alabel("SwappedEvents")] = self.numSwappedEvents
		self.logman["USER"].ints[self.alabel("OutputEvents")] = self.numOutputEvents

		return
