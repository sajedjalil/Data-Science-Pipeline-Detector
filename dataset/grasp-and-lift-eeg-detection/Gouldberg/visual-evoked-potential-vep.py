# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 11:37:26 2015

@author: alexandrebarachant

This script illustrates the presence of a VEP associated with the led lighting 
ON when the subject is cued to start the movement.

the html part has been inspired from the  Normalized Kaggle Distance for 
visualization from Triskelion.
"""


import numpy as np
import pandas as pd
from mne.io import RawArray
from mne.channels import read_montage
from mne import create_info, find_events, Epochs, concatenate_raws, pick_types
from mne import compute_raw_data_covariance
from mne.viz import plot_image_epochs, plot_topomap
from mne.viz import plot_topomap, plot_topo
from scipy.linalg import eigh, inv
import matplotlib.pyplot as plt

from glob import glob

def fit_xdawn(evoked, signal_cov):
    """Minimal implementation of xdawn."""
    cov_evoked = np.cov(evoked)
    evals, evecs = eigh(cov_evoked, signal_cov)
    evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors
    evecs /= np.sqrt(np.sum(evecs ** 2, axis=0))
    A = inv(evecs.T)
    return evecs, A

def apply_xdawn(epochs, V, A, n_components=3):
    """Xdawn denoising."""
    data = epochs.get_data()
    sources = np.dot(V.T, data).transpose((1,0,2))
    sources[:,n_components:,:] = 0
    data = np.dot(A, sources).transpose((1,0,2))
    epochs._data = data
    return epochs

def creat_mne_raw_object(fname, read_events=True):
    """Create a mne raw instance from csv file."""
    # Read EEG file
    data = pd.read_csv(fname)

    # get chanel names
    ch_names = list(data.columns[1:])

    # read EEG standard montage from mne
    montage = read_montage('standard_1005', ch_names)

    ch_type = ['eeg']*len(ch_names)
    data = 1e-6*np.array(data[ch_names]).T

    if read_events:
        # events file
        ev_fname = fname.replace('_data', '_events')
        # read event file
        events = pd.read_csv(ev_fname)
        events_names = events.columns[1:]
        events_data = np.array(events[events_names]).T

        # define channel type, the first is EEG, the last 6 are stimulations
        ch_type.extend(['stim']*6)
        ch_names.extend(events_names)
        # concatenate event file and data
        data = np.concatenate((data, events_data))

    # create and populate MNE info structure
    info = create_info(ch_names, sfreq=500.0, ch_types=ch_type,
                       montage=montage)
    info['filename'] = fname

    # create raw object
    raw = RawArray(data, info, verbose=False)

    return raw

subject = 3

fnames =  glob('../input/train/subj%d_series[3-8]_data.csv' % (subject))
fnames.sort()
# read and concatenate all the files
raw = [creat_mne_raw_object(fname) for fname in fnames]
raw = concatenate_raws(raw)
# pick eeg signal
picks = pick_types(raw.info, eeg=True)    

raw.filter(1, 20, picks=picks)

events = find_events(raw,stim_channel='HandStart')
epochs = Epochs(raw, events, {'HandStart' : 1}, -0.2, 0.6, proj=False,
                picks=picks, baseline=None, preload=True, 
                add_eeg_ref=True, verbose =False)
  
evoked = epochs.average()

evoked.plot(show=False)
plt.savefig('evoked_time.png' ,bbox_inches='tight', dpi=300)

plot_topo(evoked, show=False)
plt.savefig('evoked_topo.png' ,bbox_inches='tight', facecolor='k', dpi=300)

evoked.plot_topomap(times=[-0.1, 0, 0.1, 0.15, 0.25, 0.3], show=False)
plt.savefig('evoked_topomap.png' ,bbox_inches='tight', dpi=300)

plot_image_epochs(epochs, picks=[-3], sigma=5, vmin=-75, vmax=75, show=False)
plt.savefig('epochs_image.png' ,bbox_inches='tight', dpi=300)



#denoising with Xdawn
signal_cov = compute_raw_data_covariance(raw, picks=picks).data

V, A = fit_xdawn(evoked.data, signal_cov)
epochs_dn = apply_xdawn(epochs, V, A, n_components=4)
plot_image_epochs(epochs_dn, picks=[-3], sigma=5, vmin=-70, vmax=70, show=False)
plt.savefig('epochs_xdawn_image.png' ,bbox_inches='tight', dpi=300)


# Generate html file
with open("output.html","wb") as outfile:
    html = """<!DOCTYPE html>
<html>
	<head>
		<meta charset="utf-8">
		<title>VEP - Alexandre Barachant</title>
		<meta name="robots" content="noindex,nofollow,noodp,nonothing,goaway">
		<link href='https://fonts.googleapis.com/css?family=Open+Sans:400,800' rel='stylesheet' type='text/css'>
		<style>
			* { margin:0; padding: 0;}
			body { font-family: "Open Sans",Verdana,sans-serif; margin: 30px auto; width: 700px; }
			p { font-size: 18px; line-height: 27px; padding-bottom: 27px; color: #111; text-align: justify; }
			h1 { font-weight: 800; font-size: 33px; color: #2C3E50; padding-bottom: 10px;}
			h2 { font-weight: 800; font-size: 22px; color: #2C3E50; padding-bottom: 10px; }
			h3 { font-weight: 800; font-size: 18px; color: #34495E; padding-bottom: 10px; }
			small { color: #7F8C8D; }
			ul.panes { display: block; overflow: hidden; list-style-type: none; padding: 10px 0px;}
			li.pane { float: left; width: 300px; margin-right: 40px; padding-bottom: 40px; }
			ul { list-style-type: none; }
			span { display: block; width: 50px; padding-right: 15px; text-align: right; height: 20px; float: left; }
			li ul li { color: #7F8C8D; }
			li ul li:first-child { color: #111; }
			.node { font: 400 14px "Open Sans", Verdana, sans-serif; fill: #333; cursor:pointer;}
			.node:hover {fill: #000;}
			.link {stroke: steelblue; stroke-opacity:.4;fill: none; pointer-events: none;}
			.node:hover,.node--source,.node--target { font-weight: 700;}
			.node--source { fill: #2ca02c;}
			.node--target { fill: #d62728;}
			.link--source,.link--target { stroke-opacity: 1; stroke-width: 2px;}
			.link--source { stroke: #d62728;}
			.link--target { stroke: #2ca02c;}
                   img { width: 600px; display: block; margin: 0 auto; }
		</style>
	</head>
	<body>
		<h1>Visual Evoked Potential (VEP)</h1>
            <p>Evoked potential are time locked brain potential elicited by 
            an external stimultation. They are generally related to the sensory
            or the cognitive system. In our case, the subject is instructed to
            start the movement when a LED light ON. Therefore, we expect to see
            a potential over the visual cortex in response to this event.</p>
                       
		<h2>VEP Analysis</h2>
            <p>VEP analysis are usually done by averaging time-domain signal
            across several trials in order to reduce noise (which is asumed to be zeros
            mean and not in phase with the event). 
            VEPs must be in sync with the event you are 
            using to epoch signal. In this case, we don't have access to the
            LedON events, but depending on with how much reproducibility the 
            subject start moving the hand, we can use the Event Handstart to epochs signal.</p>
            
            <p>In this example, I used the subject 3. I skept the 2 first series
            because they showed a bad reproducibility on the timing. We obtain 198 'HandStart' events.
            Signal is bandpass filtered between 1 and 20 Hz (VEP are low frequency) and then epoched
            from -200ms to 600ms with respect to the onset of the event.</p>
            
		<img src="evoked_time.png">
  
            <p>The average show a first negative peak around 0ms, and a second positive at 150ms.
            typical VEP appears arrond 300ms after the events. Therefore we can guess that the 
            subject start the movement in 150ms after the cue.</p>
             
            <img src="evoked_topo.png">
            <img src="evoked_topomap.png">

            <p>The first figure represent the individual potential for each electrode, with respect to their position on the scalp.
            The second figure represent the topomap of the amplitude of the VEP for different timing.
            We see the strongest response over the visual cortex (back of the head).
            Interestingly, we can also see a response on the frontal electrodes. This may be an occular artifact,
            or an effect of the referencing of the signal.</p>            
            
            <h2>Epochs denoising with xDawn</h2>
            
            <p>On the single trial basis, epochs are really noisy and VEP are hard to detect.
            We offen use spatial filtering (linear combination of electrodes) like ICA to denoise the signal.
            Here is a plot of each epochs before denoising.</p>
            
            <img src="epochs_image.png">
            
            <p> In this example, I used the algorithm xDAWN [1,2] to denoise the signal. 
            xDAWN build spatial filters in order to maximize the signal to noise ratio of the 
            evoked response. After spatial filtering, noisy components are zeroed and back projected 
            in the sensor space. This dramatically increase the quality of each response.</p>
            
            <img src="epochs_xdawn_image.png">
            
            <p>Interestingly, we can see a time shift in the latency on the VEP peak, probably due to
            mental fatigue of the subject.</p>
            
		    <p> A full implementation of xDawn algorithm is provided in the last code of MNE.
		    Since it wasn't available here, I made a minimalist implementation.</p>
		    
		    <h2>References</h2>
		    <p>[1] Rivet, B., Souloumiac, A., Attina, V., & Gibert, G. (2009). xDAWN
            algorithm to enhance evoked potentials: application to brain-computer
            interface. Biomedical Engineering, IEEE Transactions on, 56(8), 2035-2043.</p>
   
            <p>[2] Rivet, B., Cecotti, H., Souloumiac, A., Maby, E., & Mattout, J. (2011,
            August). Theoretical analysis of xDAWN algorithm: application to an
            efficient sensor selection in a P300 BCI. In Signal Processing Conference,
            2011 19th European (pp. 1382-1386). IEEE.</p>
      </body>
</html>"""

    outfile.write(html.encode('utf-8'))

