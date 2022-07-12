# Note: Kaggle only runs Python 3, not Python 2
import pandas
import numpy
from matplotlib import pyplot as plt

print("Compare a couple of variables")
#        2         3        4                   5             6     7         8     9      10       11                                     15                                                           20                                            26
#   id, LifeTime, dira, FlightDistance, FlightDistanceError, IP, IPSig, VertexChi2, pt, DOCAone, DOCAtwo, DOCAthree, IP_p0p2, IP_p1p2, isolationa, isolationb, isolationc, isolationd, isolatione, isolationf, iso, CDF1, CDF2, CDF3, ISO_SumBDT, p0_IsoBDT: string;
#      27                                     30                                                        35                                         40                               45                                  49        50
#   p1_IsoBDT, p2_IsoBDT, p0_track_Chi2Dof, p1_track_Chi2Dof, p2_track_Chi2Dof, p0_IP, p1_IP, p2_IP, p0_IPSig, p1_IPSig, p2_IPSig, p0_pt, p1_pt, p2_pt, p0_p, p1_p, p2_p, p0_eta, p1_eta, p2_eta, SPDhits, production, signal, mass, min_ANNmuon: string;

data=numpy.genfromtxt('../input/training.csv',skiprows=1,delimiter=',')

SPDhits=data[:,46]
signal=data[:,48]
mass=data[:,49]
LifeTime=data[:,1]
FlightDistance=data[:,3]
IP=data[:,5]
IPSig=data[:,6]
pt=data[:,8]
# p0_p = p0_pt * cosh(p0_eta)
p0_p=data[:,40]
p0_pt=data[:,37]
p0_eta=data[:,43]
# from sklearn import preprocessing
# p0_p_scaled = preprocessing.scale(p0_p)
p0_eta_cosh = numpy.cosh(p0_eta)

# plt.rcParams["figure.figsize"] = 8,5
# plt.plot(IP,IPSig,'*',linewidth=0)
# plt.scatter(p0_p, p0_eta_cosh, label="Sample Distribution")
# red dashes, blue squares and green triangles
# plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
# plt.plot(p0_p, p0_pt, 'g^', label="Sample Distribution")

from scipy import stats
def drawme( x, y, xname, yname ):
   slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x, y)
   print ("==== new chart =========== "+xname+" by "+yname)
   print("slope="+str(slope)+", intercept="+str(intercept)+", p_value="+str(p_value)+", r_value="+str(r_value) )
#   stp_value = "%.2f" % p_value
#   plt.title("p_value="+str(p_value))
# Calculate some additional outputs
   predict_y = intercept + slope * x
   print(str(len(x))+":"+str(len(predict_y)))
   pred_error = y - predict_y
   degrees_of_freedom = len(x) - 2
   residual_std_error = numpy.sqrt(numpy.sum(pred_error**2) / degrees_of_freedom)
# Plotting
   plt.plot(x, y, 'o')
   plt.plot(x, predict_y, 'k-')
   plt.xlabel(xname)
   plt.ylabel(yname)

   return;

# Now you can call drawme function

def setupme( v1, v2, v1name, v2name ):
   print("==========================================================")
   print(v1name+"="+str(len(v1))+":"+str(v1[0])+":"+str(v1[1])+":"+str(v1[2]))
   highx = -9999999
   highy = -9999999
   
   breakfield = v1
   i0 = 0                    
   triple1 = numpy.array([])
   notriple1 = numpy.array([])
   
   notriple1 = numpy.array([])
   while i0 < len(signal):              
      if breakfield[i0] > highx: highx = breakfield[i0]
      if signal[i0] < 0.5:
          notriple1 = numpy.append(notriple1, breakfield[i0])
      if signal[i0] >= 0.5:
          triple1 = numpy.append(triple1, breakfield[i0])
      i0 = i0 + 1
   triple1 = numpy.append(triple1, highx)
   notriple1 = numpy.append(notriple1, highx)
#   print('triple1='+str(len(triple1))+', notriple1='+str(len(notriple1)))
   breakfield = v2
   i0 = 0                    
   triple2 = numpy.array([])
   notriple2 = numpy.array([])
   while i0 < len(signal):   
      if breakfield[i0] > highy: highy = breakfield[i0]
      if signal[i0] < 0.5:
          notriple2 = numpy.append(notriple2, breakfield[i0])
      if signal[i0] >= 0.5:
          triple2 = numpy.append(triple2, breakfield[i0])
      i0 = i0 + 1
   triple2 = numpy.append(triple2, highy)
   notriple2 = numpy.append(notriple2, highy)
#   print('triple2='+str(len(triple2))+', notriple2='+str(len(notriple2)))

   plt.subplot(1,2,1)
   plt.title("No Triple Muon")
# drawme(LifeTime, FlightDistance, 'Life Time', 'Flight Distance')
   drawme(notriple1, notriple2, v1name, v2name)
   plt.subplot(1,2,2)
   plt.title("Triple Muon")
   drawme(triple1, triple2, v1name, '')
   return;


setupme(LifeTime, FlightDistance, 'Life Time', 'Flight Distance')
print("Saved LifeTime/FlightDistance regression")
plt.savefig("r1.png")
plt.clf()

setupme(IP, FlightDistance, 'Impact Parameter', 'Flight Distance')
print("Saved IP/FlightDistance regression")
plt.savefig("r2.png")
plt.clf()

setupme(SPDhits, FlightDistance, 'SPD hits', 'Flight Distance')
print("Saved IP/SPDhits regression")
plt.savefig("r3.png")
plt.clf()


