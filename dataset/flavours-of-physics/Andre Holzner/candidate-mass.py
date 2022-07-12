# Note: Kaggle only runs Python 3, not Python 2
import pandas
import numpy as np
import pylab

#----------------------------------------------------------------------

def addMass(dataFrame):

    # units are in MeV (see https://www.kaggle.com/c/flavours-of-physics/forums/t/16214/units/91355#post91355 )
    # without loss of generality we assume that phi of the tau candidate is zero

    # muon mass in MeV/c^2
    mmu = 105.6583715

    # calculate tau energy
    dataFrame['tau_e'] = np.sqrt(dataFrame.p0_p ** 2 + mmu**2) + \
                     np.sqrt(dataFrame.p1_p ** 2 + mmu**2) + \
                     np.sqrt(dataFrame.p2_p ** 2 + mmu**2)

    # calculate pz of tau candidate
    dataFrame['tau_pz'] = dataFrame.p0_pt * np.sinh(dataFrame.p0_eta) + \
                      dataFrame.p1_pt * np.sinh(dataFrame.p1_eta) + \
                      dataFrame.p2_pt * np.sinh(dataFrame.p2_eta)

    # calculate momentum of tau candidate
    dataFrame['tau_p'] = np.sqrt(dataFrame.pt ** 2 + dataFrame.tau_pz ** 2)

    # calculate eta of tau candidate
    dataFrame['tau_eta'] = np.arcsinh(dataFrame.tau_pz / dataFrame.pt)

    # calculate mass of tau candidate

    dataFrame['tau_m'] = np.sqrt(np.maximum(dataFrame.tau_e ** 2 - dataFrame.tau_p ** 2, 0))


#----------------------------------------------------------------------

def plotMass(inputName, selectSignal = None):

    signalLabel = {
        True: "signal",
        False: "background",
        }

    inputFname = "../input/" + inputName + ".csv"

    data = pandas.read_csv(inputFname)

    #----------
    # calculate the mass
    #----------
    addMass(data)

    #----------
    # plot the mass distribution
    #----------
    pylab.figure()

    nbins, xmin, xmax = 100, 1500, 2100

    binwidth = (xmax - xmin) / float(nbins)

    if selectSignal != None:
        thisData = data[(data.signal == 0) ^ selectSignal ]
    else:
        thisData = data
    
    pylab.hist(thisData.tau_m.as_matrix(), bins = np.linspace(xmin, xmax, nbins + 1))

    pylab.xlabel('$\\tau$ candidate mass [MeV/$c^2$]')
    pylab.ylabel('events per %.1f MeV/$c^2$' % binwidth)

    if selectSignal != None:
        pylab.title(inputName + ".csv (%s)" % signalLabel[selectSignal])
    else:
        pylab.title(inputName + ".csv")

    #----------
    # add lines for some particle masses
    #----------
    for label, mass, color in (
        ["$\\tau$",    mtau,  "green"],
        ["$D_s^\pm$", mdsubs, "purple" ],
        ):
        pylab.plot([mass, mass], pylab.gca().get_ylim(), label = label, linewidth = 2)
        
    pylab.legend(loc = 'upper left')

    #----------
    pylab.grid()
    pylab.show()


    if selectSignal != None:
        outputSuffix = "-" + signalLabel[selectSignal]
    else:
        outputSuffix = ""

    pylab.savefig(inputName + outputSuffix + ".png")


#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------

# tau mass in MeV/c^2
mtau = 1776.82

# D_s^+/- mass in MeV/c^2
mdsubs = 1968.30

for inputName in [
    "training",
    "test",
    "check_agreement",
    "check_correlation",
    ]:

    if inputName == 'training':
        plotMass(inputName, selectSignal = True)
        plotMass(inputName, selectSignal = False)
    else:
        plotMass(inputName)
        
    
