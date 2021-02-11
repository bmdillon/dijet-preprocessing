import sys
import os
import numpy as np
import pyjet
import h5py     
from pyjet import cluster,DTYPE_PTEPM
import pandas as pd


##########################
# function for clustering the LHCO test dataset in chunks
##########################

def cluster_lhco_testdata_part(h5file,R,p,evrange):

    f = pd.read_hdf(h5file,start=evrange[0],stop=evrange[1])
    events_combined = f.T
    leadpT = {}
    subleadpT={}
    leadM = {} # take the two leading pT jets, look at mass of heaviest and lightest jets
    subleadM={}
    alljets = {}

    for mytype in ['background','signal']:

        leadpT[mytype]=[]
        subleadpT[mytype]=[]
        leadM[mytype]=[]
        subleadM[mytype]=[]
        leadMpT[mytype]=[]
        subleadMpT[mytype]=[]
        alljets[mytype]=[]

        for i in range(evrange[0],evrange[1]): #len(events_combined)):
            if (i%10000==0):
                print(mytype,i)
                pass
            issignal = events_combined[i][2100]
            if (mytype=='background' and issignal):
                continue
            elif (mytype=='signal' and issignal==0):
                continue
            pseudojets_input = np.zeros(len([x for x in events_combined[i][::3] if x > 0]), dtype=DTYPE_PTEPM)

            for j in range(700):
                if (events_combined[i][j*3]>0):
                    pseudojets_input[j]['pT'] = events_combined[i][j*3]
                    pseudojets_input[j]['eta'] = events_combined[i][j*3+1]
                    pseudojets_input[j]['phi'] = events_combined[i][j*3+2]
                    pass
                pass

            sequence = cluster(pseudojets_input, R=1.0, p=p)
            jets = sequence.inclusive_jets(ptmin=20)
            leadpT[mytype] += [jets[0].pt]
            if len(jets)>1:
                subleadpT[mytype] += [jets[1].pt]
            jets=sorted(jets,key=lambda PseudoJet: PseudoJet.pt,reverse=True)[0:2]
            jets=sorted(jets,key=lambda PseudoJet: PseudoJet.mass,reverse=True)
            leadM[mytype] += [jets[0].mass]
            if len(jets)>1:
                subleadM[mytype] += [jets[1].mass]
            alljets[mytype] += [jets]

    return alljets, leadpT, subleadpT, leadM, subleadM

##########################
# function for clustering the LHCO blackbox datasets in chunks
##########################

def cluster_lhco_dataset_parallel(h5file,R,p,evrange):

    f = pd.read_hdf(h5file,start=evrange[0],stop=evrange[1])
    events_combined = f.T
    leadpT=[]
    subleadpT=[]
    leadM=[]
    subleadM=[]
    alljets=[]

    for i in range(evrange[0],evrange[1]):

        if (i%10000==0):
            print(i)
            pass
        pseudojets_input = np.zeros(len([x for x in events_combined[i][::3] if x > 0]), dtype=DTYPE_PTEPM)

        for j in range(700):
            if (events_combined[i][j*3]>0):
                pseudojets_input[j]['pT'] = events_combined[i][j*3]
                pseudojets_input[j]['eta'] = events_combined[i][j*3+1]
                pseudojets_input[j]['phi'] = events_combined[i][j*3+2]
                pass
            pass

        sequence = cluster(pseudojets_input, R=1.0, p=p)
        jets = sequence.inclusive_jets(ptmin=20)
        leadpT += [jets[0].pt]

        if len(jets)>1:
            subleadpT += [jets[1].pt]
        jets=sorted(jets,key=lambda PseudoJet: PseudoJet.pt,reverse=True)[0:2]
        jets=sorted(jets,key=lambda PseudoJet: PseudoJet.mass,reverse=True)
        leadM += [jets[0].mass]
        if len(jets)>1:
            subleadM += [jets[1].mass]
        alljets += [jets]

    return alljets, leadpT, subleadpT, leadM, subleadM


##########################
# function for getting the Lund history for a single jet, i.e. the clustering history but with each 4-momentum having a 'plane-id'
##########################

def get_lund_history(jet,R,p):

    # re-clustering the jet
    clustered_jet = cluster(jet.constituents_array(), R=R, p=p)
    splittings=[]
    
    # each level in the clustering history is represented by a list of subjets
    # each subjet has [4-momentum,plane_id]
    lund_history = []

    # looping over the levels
    for t in range(0,clustered_jet.n_exclusive_jets(0)):

        # list momenta of all subjets at current splitting
        j_obs = [ [j.mass,j.pt,j.eta,j.phi] for j in clustered_jet.exclusive_jets(t) ]
        j_obs.sort( key = lambda i: i[1], reverse=True )
        j_obs_id = [ [j_obs[i],i] for i in range(len(j_obs)) ]

        if len(j_obs) <= 2:
            lund_history.append( j_obs_id )

        if len(j_obs)>2:
            # list momenta,id of all subjets at previous splitting
            pj_obs_id = [ j for j in lund_history[-1] ]
            pj_obs = [ j[0] for j in lund_history[-1] ]
            
            # work out which subjet split, and label it p_obs
            p_obs = [ pj_obs[i] for i in range(len(pj_obs)) if pj_obs[i] not in j_obs ]
            p_obs_id = [ pj_obs_id[i] for i in range(len(pj_obs)) if pj_obs[i] not in j_obs ]
            
            # work out which subjets didn't split, label them np_obs
            np_obs = [ pj_obs[i] for i in range(len(pj_obs)) if pj_obs[i] in j_obs ]
            np_obs_id = [ pj_obs_id[i] for i in range(len(pj_obs)) if pj_obs[i] in j_obs ]

            # work out what it split into, and put them in d_obs
            d_obs = [ j_obs[i] for i in range(len(j_obs)) if j_obs[i] not in pj_obs ]
            d_obs.sort( key = lambda i: i[1], reverse=True )
            pid = p_obs_id[0][1]
            d_obs_id = [ [d_obs[i],i+pid] for i in range(len(d_obs)) ]
            
            if len(d_obs) == 2 and len(p_obs) == 1:
                lund_history.append( np_obs_id + d_obs_id )

    return lund_history

##########################
# function which takes the lund history and computes the observables at all splittings in the jet, with a label indicating which plane it comes from
##########################

def get_lund_splittings(jet_history):
    
    lund_splittings = []

    for i in range( len(jet_history)-1 ):
        
        # get info from current (j) and previous (pj) subjets in history
        pj_obs_id = jet_history[i]
        j_obs_id = jet_history[i+1]
        pj_obs = [ j[0] for j in pj_obs_id ]
        j_obs = [ j[0] for j in j_obs_id ]

        # get the subjet that split
        p_obs_id = [ pj_obs_id[i] for i in range(len(pj_obs)) if pj_obs[i] not in j_obs ]
        p_obs = [ i[0] for i in p_obs_id ]

        # get what it split into
        d_obs_id = [ j_obs_id[i] for i in range(len(j_obs)) if j_obs[i] not in pj_obs ]
        d_obs = [ i[0] for i in d_obs_id ]

        # check that len(d_obs) is 2, and len(p_obs) is 1, in case there is a very soft splitting where it appears that d_obs has just one component due the numerical accuracy
        # calculate observables
        if len(d_obs) == 2 and len(p_obs) == 1:
            pt = p_obs[0][1]
            pmass = p_obs[0][0]
            max_d_mass = max( d_obs[0][0], d_obs[1][0] )
            min_d_mass = min( d_obs[0][0], d_obs[1][0] )
            d_pts = [ d_obs[0][1], d_obs[1][1] ]
            max_d_pt = max( d_pts )
            min_d_pt = min( d_pts )
            plane_id = p_obs_id[0][1]
            try:
                mass_drop = max_d_mass/pmass
            except ZeroDivisionError:
                mass_drop = 0
            try:
                d_mass_ratio = min_d_mass/max_d_mass
            except ZeroDivisionError:
                d_mass_ratio = 0
            dR = np.sqrt( ( d_obs[0][2] - d_obs[1][2])**2 + (d_obs[0][3] - d_obs[1][3] )**2 )
            logidR = np.log( 1/dR )
            logkt = np.log( min_d_pt*dR )
            z = min_d_pt/(min_d_pt+max_d_pt)
            kap = z*dR
            
            # assign to a splitting
            splitting = [ plane_id, pt, pmass, mass_drop, d_mass_ratio, dR, logidR, logkt, z, kap ]

            # append to lund splittings
            lund_splittings.append( splitting )

    return lund_splittings

