import numpy as np
import scipy
from landau import landau as landau
import psutil
import os

def usage():
    process = psutil.Process(os.getpid())
    return process.memory_info()[0] / float(2**20)

def print_mem_usage():
    print(f"\033[93mCurrent Mem Usage:{usage():.2f}Mb\033[0m")


def readFileName(path):
    angle_dict = {
        "angle1": 45,
        "angle2": 40.5,
        "angle3": 28,
        "angle4": 20.5,
        "angle5": 11,
        "angle6": 86.5,
    }
    voltage_dict = {
        "V48": 48,
        "V30": 30,
        "V20": 20,
        "V15": 15,
        "V10": 10,
        "V8": 8,
        "V6": 6,
        "V4": 4,
        "V2": 2,
        "V0": 0,
    }
    file_name = "_".join(path.split("/")[-1].split(".")[0].split("_")[3:]).removesuffix("_decode")
    angle = 0
    for k in angle_dict.keys():
        if k in file_name:
            angle = angle_dict[k]
            break
    voltage = -1
    for k in voltage_dict.keys():
        if k in file_name:
            voltage = voltage_dict[k]
            break
    
    if "kit" in path:
        telescope = "kit"
    elif "lancs" in path:
        telescope = "lancs"
    if voltage == -1:
        voltage = 48
    return file_name,voltage,angle,telescope

def isFiltered(dataAnalysis, filter_dict={}):
    # Takes in an array of data_class and filters and sorts them
    # filter_dict has keys that are attributes of data_class with values that you want to filter for
    # Returns filtered list sorted by angle and then by voltage
    for f in filter_dict.keys():
        attr = getattr(dataAnalysis, "get_"+f)
        if isinstance(f, list):
            if not np.isin(attr, filter_dict[f]):
                return True
        else:
            if not attr == filter_dict[f]:
                return True
    return False

def calcToT(TS,TS2):
    return (TS2 * 2 - TS) % 256

def calcHit_Voltage(rows,columns,ToTs,Layers):
    hit_voltage = np.empty(len(rows), dtype=float)
    for k in range(1, 5):
        calibration_array = np.load(f"/home/atlas/rballard/Code/tot_calibration/data_{k}.npy")
        calibration_array_indexes = calibration_array[
            columns[Layers == k], rows[Layers == k]
        ]
        u_0 = calibration_array_indexes[:, 0]
        a = calibration_array_indexes[:, 1]
        b = calibration_array_indexes[:, 2]
        c = calibration_array_indexes[:, 3]
        hit_voltage[Layers == k] = np.real(lambert_W_ToT_to_u(ToTs[Layers == k]*2, u_0, a, b, c))
    return hit_voltage

def calcHit_VoltageError(rows,columns,ToTs,Layers):
    hit_voltage = np.empty(len(rows), dtype=float)
    hit_voltageError = np.empty(len(rows), dtype=float)
    for k in range(1, 5):
        calibration_array = np.load(f"/home/atlas/rballard/Code/tot_calibration/data_{k}.npy")
        calibration_array_indexes = calibration_array[
            columns[Layers == k], rows[Layers == k]
        ]
        u_0 = calibration_array_indexes[:, 0]
        a = calibration_array_indexes[:, 1]
        b = calibration_array_indexes[:, 2]
        c = calibration_array_indexes[:, 3]
        hit_voltage[Layers == k] = np.real(lambert_W_ToT_to_u(ToTs[Layers == k]*2, u_0, a, b, c))
        upper = np.real(lambert_W_ToT_to_u(ToTs[Layers == k]+2, u_0, a, b, c))
        lower = np.real(lambert_W_ToT_to_u(ToTs[Layers == k]-2, u_0, a, b, c))
        upperError = upper-hit_voltage[Layers == k]
        lowerError = hit_voltage[Layers == k]-lower
        hit_voltageError[Layers == k] = np.max([upperError,lowerError],axis=0)
    return hit_voltageError

def lambert_W_ToT_to_u(ToT, u_0, a, b, c):
    u = u_0 + (a / b) * scipy.special.lambertw((b / a) * u_0 * np.exp((ToT - c - (b * u_0)) / a))
    return u

def lambert_W_u_to_ToT(u, u_0, a, b, c):
    u = np.reshape(u, np.size(u))
    ToT = np.empty(np.shape(u))
    ToT[u > u_0] = (a * np.log((u[u > u_0] - u_0) / u_0)) + (b * u[u > u_0]) + c
    ToT[u <= u_0] = 0
    ToT[ToT < 0] = 0
    ToT = list(ToT)
    return ToT


def calcClusters(Layers,TriggerID,TS, time_variance=np.uint16(80), trigger_variance=np.int32(1)):
    clusters = np.empty(len(Layers), dtype=object)
    count = 0
    count_2 = 0
    max_cluster_size = 300
    # Does one layer at a time. Ensures each cluster is on one layer
    for j in range(1, 5):
        diff_TS = np.min([np.diff(TS[Layers==j]),np.diff((TS[Layers==j]+512)%1024)],axis=0)
        diff_TriggerID = np.diff(TriggerID[Layers==j])
        pixels = np.zeros(max_cluster_size, dtype=int)
        onLayer = np.where(Layers==j)[0]
        for i,index in enumerate(onLayer):
            if index != onLayer[-1]:
                if (
                    # Doesn't account for looping around 1024
                    (abs(diff_TS[i]) <= time_variance)
                    and (abs(diff_TriggerID[i]) <= trigger_variance)
                    and (count_2 < max_cluster_size - 1)
                ):
                    pixels[count_2] = index
                    count_2 += 1
                else:
                    pixels[count_2] = index
                    count_2 += 1
                    clusters[count] = np.array(pixels[:count_2], dtype=int)
                    count += 1
                    pixels[:] = 0
                    count_2 = 0
            else:
                pixels[count_2] = index
                count_2 += 1
                clusters[count] = np.array(pixels[:count_2], dtype=int)
                count += 1
                pixels[:] = 0
                count_2 = 0
    clusters = clusters[:count]
    print(f"{len(clusters)} clusters found")
    return clusters

def calcHit_VoltageByPixel(clusters,clusterWidths,maxClusterWidth=40,excludeCrossTalk=True,returnIndexes=False,measuredAttribute = "Hit_Voltage"):
    uniqueClusterWidths,unique_counts = np.unique(clusterWidths,return_counts=True)
    hitPositionArray = np.empty((maxClusterWidth, maxClusterWidth, np.max(unique_counts)), dtype=float)
    hitPositionErrorArray = np.empty((maxClusterWidth, maxClusterWidth, np.max(unique_counts)), dtype=float)
    counts = np.zeros(maxClusterWidth, dtype=int)
    if returnIndexes:
        indexes = np.empty((maxClusterWidth, np.max(unique_counts)), dtype=int)
    for i,cluster in enumerate(clusters):
        width = cluster.getRowWidth(excludeCrossTalk)
        if width > 0 and width <= maxClusterWidth and len(np.unique(cluster.getColumns(excludeCrossTalk))) == 1:
            Hit_Voltages = np.zeros(width, dtype=float)
            Hit_VoltageErrors = np.zeros(width, dtype=float)
            sort_array = np.argsort(cluster.getRows(excludeCrossTalk))
            # Hit_Voltage = Hit_Voltages[cluster]
            x = cluster.getRows(excludeCrossTalk)[sort_array]
            if measuredAttribute == "Hit_Voltage":
                Hit_Voltages[(x - np.min(x)).astype(int)] = cluster.getHit_Voltages(excludeCrossTalk)[sort_array]
                Hit_VoltageErrors[(x - np.min(x)).astype(int)] = cluster.getHit_VoltageErrors(excludeCrossTalk)[sort_array]
            elif measuredAttribute == "ToT":
                Hit_Voltages[(x - np.min(x)).astype(int)] = cluster.getToTs(excludeCrossTalk)[sort_array]
                Hit_VoltageErrors[(x - np.min(x)).astype(int)] = cluster.getToTErrors(excludeCrossTalk)[sort_array]
            # Hit_Voltage = Hit_Voltage[np.argsort(x)]
            if counts[width - 2] < np.max(unique_counts):
                hitPositionArray[width - 2, :width, counts[width - 2]] = Hit_Voltages
                hitPositionErrorArray[width - 2, :width, counts[width - 2]] = Hit_VoltageErrors
                if returnIndexes:
                    indexes[width - 2, counts[width - 2]] = cluster.index
                counts[width - 2] += 1
    if returnIndexes:
        return hitPositionArray[:,:,:np.max(counts)+2],hitPositionErrorArray[:,:,:np.max(counts)+2],counts,indexes[:,:np.max(counts)+2]
    return hitPositionArray[:,:,:np.max(counts)+2],hitPositionErrorArray[:,:,:np.max(counts)+2],counts

class _ShapeInfo:
    def __init__(self, name, integrality=False, domain=(-np.inf, np.inf),
                 inclusive=(True, True)):
        self.name = name
        self.integrality = integrality
        self.endpoints = domain
        self.inclusive = inclusive

        domain = list(domain)
        if np.isfinite(domain[0]) and not inclusive[0]:
            domain[0] = np.nextafter(domain[0], np.inf)
        if np.isfinite(domain[1]) and not inclusive[1]:
            domain[1] = np.nextafter(domain[1], -np.inf)
        self.domain = domain

def landauFunc(x,x_mpv, xi,scaler):
    y = landau.pdf(x, x_mpv, xi) * scaler
    y = np.reshape(y, np.size(y))
    y[x<0.16] = 0
    return y

def gaussianFunc(x, mu, sig,scaler):
    return (1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)) * scaler

def gaussianCDFFunc(x,mu, sig):
    return scipy.stats.norm.cdf(x, mu, sig)


def TStoMS(TS):
    return TS*25/1000000

def MStoTS(Time):
    return int(Time*1000000/25)


def neg_log_likelihood_truncated(params, data,threshold = 0.162):
    x_mpv, xi = params
    if xi <= 0:
        return np.inf

    # Normalization constant for Landau above threshold
    Z = 1 - landau.cdf(threshold, x_mpv, xi)
    if Z <= 0 or np.isnan(Z) or np.isinf(Z):
        return np.inf

    # Evaluate truncated PDF for data > threshold
    pdf_vals = landau.pdf(data, x_mpv, xi) / Z
    if np.any(pdf_vals <= 0) or np.any(np.isnan(pdf_vals)) or np.any(np.isinf(pdf_vals)):
        return np.inf

    return -np.sum(np.log(pdf_vals))

def lowess(x, y, f=1./3.):
    """
    Basic LOWESS smoother with uncertainty. 
    Note:
        - Not robust (so no iteration) and
             only normally distributed errors. 
        - No higher order polynomials d=1 
            so linear smoother.
    """
    # get some paras
    xwidth = f*(x.max()-x.min()) # effective width after reduction factor
    N = len(x) # number of obs
    # Don't assume the data is sorted
    order = np.argsort(x)
    # storage
    y_sm = np.zeros_like(y)
    y_stderr = np.zeros_like(y)
    # define the weighting function -- clipping too!
    tricube = lambda d : np.clip((1- np.abs(d)**3)**3, 0, 1)
    # run the regression for each observation i
    for i in range(N):
        dist = np.abs((x[order][i]-x[order]))/xwidth
        w = tricube(dist)
        # form linear system with the weights
        A = np.stack([w, x[order]*w]).T
        b = w * y[order]
        ATA = A.T.dot(A)
        ATb = A.T.dot(b)
        # solve the system
        sol = np.linalg.solve(ATA, ATb)
        # predict for the observation only
        yest = A[i].dot(sol)# equiv of A.dot(yest) just for k
        place = order[i]
        y_sm[place]=yest 
        sigma2 = (np.sum((A.dot(sol) -y [order])**2)/N )
        # Calculate the standard error
        y_stderr[place] = np.sqrt(sigma2 * 
                                A[i].dot(np.linalg.inv(ATA)
                                                    ).dot(A[i]))
    return y_sm, y_stderr

def errorFunc_Voltage(x,n):
    error = (np.exp((x-0.16)/(-0.3))/10)**np.sqrt(n)+0.1*np.sqrt(n)
    error[0] = error[0]*2
    return error

def calcDepth(d:float,clusterWidth:int,angle:float,depthCorrection:bool=True,upTwo:bool=False):
    # 50 for 5 microns row width
    shift = 0
    if upTwo:
        shift = 2
    x = np.linspace(1-1/((clusterWidth+1)), 0, clusterWidth)*d*50
    if depthCorrection and clusterWidth+shift > (d*np.tan(np.deg2rad(angle))):
        x = x*((clusterWidth+shift)/(d*np.tan(np.deg2rad(angle))))
    return x
def adjustPeakVoltage(y:list,y_err,d:float,clusterWidth:int):
    y[1:-1] = y[1:-1]*(1/np.sqrt((((d*0.7)**2)/((clusterWidth-1)**2))+1))
    y_err[1:-1] = y_err[1:-1]*(1/np.sqrt((((d*0.7)**2)/((clusterWidth-1)**2))+1))
    return y,y_err

def histogramErrors(values,errors,binEdges):
    histErrors = np.zeros(len(binEdges[1:]))
    errorOut = (gaussianCDFFunc(binEdges[0],values,errors)-gaussianCDFFunc(0,values,errors))**2
    histErrors[0] +=np.sqrt(np.sum(errorOut))
    for i in range(len(binEdges[1:])):
        #print(len(values),len(errors))
        inRange = (values>binEdges[i]) & (values<binEdges[i+1])
        errorOut = (gaussianCDFFunc(binEdges[i+1],values[np.invert(inRange)],errors[np.invert(inRange)])-gaussianCDFFunc(binEdges[i],values[np.invert(inRange)],errors[np.invert(inRange)]))**2
        errorIn = (gaussianCDFFunc(binEdges[i],values[inRange],errors[inRange])+(1-gaussianCDFFunc(binEdges[i+1],values[inRange],errors[inRange])))**2
        histErrors[i] += np.sqrt(np.sum(np.append(errorOut,errorIn)))
        #print(gaussianCDFFunc(binEdges[i+1],values,errors),gaussianCDFFunc(binEdges[i+1],values,errors))
        #histErrors[i] = np.sum([scipy.integrate.quad(gaussianFunc,binEdges[i],binEdges[i+1],args=(value,error)) for value,error in zip(values,errors)])
    histErrors[histErrors<0.01] = 0.01
    return histErrors

def chargeCollectionEfficiencyFunc(depth,V_0,t_epi,edl,base=0.13):
    voltage = np.zeros(depth.shape)
    voltage[depth<t_epi] = V_0
    voltage[depth>=t_epi] = np.exp(-(depth[depth>=t_epi]-t_epi)/edl)*V_0
    voltage[voltage < 0.160] = 0.160
    return voltage

def fitVoltageDepth(x,y,yerr):
    bounds = [(0,np.inf), (0,300), (1,np.inf)]
    bounds = tuple(zip(*bounds))
    initial_guess = [0.25, 10, 80]
    cut = x>0#(x<70) & (y > 0.17)
    popt, pcov = scipy.optimize.curve_fit(chargeCollectionEfficiencyFunc, x[cut], y[cut], p0=initial_guess, maxfev=10000000, bounds=bounds,sigma=yerr[cut]/y[cut],absolute_sigma=False)
    return popt,pcov
def fitAndPlotCCE(ax,plot,x,y,yerr):
    popt, pcov = fitVoltageDepth(x,y,yerr)
    #pcov = pcov / scipy.stats.chisquare(popt).statistic/(np.sum(cut)-3)
    (V_0,t_epi,edl) = popt
    (V_0_e,t_epi_e,edl_e) = np.sqrt(np.diag(pcov))
    x = np.linspace(0,120,1000)
    y = chargeCollectionEfficiencyFunc(x,*popt)
    ax.plot(x, y, color=plot.colorPalette[0], linestyle="dashed",label=f"V_0   :{V_0:.5f} ± {V_0_e:.5f}\nt_epi :{t_epi:.3f} ± {t_epi_e:.3f}\n"+f"edl    :{edl:.3f} ± {edl_e:.3f}")
    possibleLines = np.random.default_rng().multivariate_normal(popt, pcov, 1000)
    ylist = [chargeCollectionEfficiencyFunc(x,param[0],param[1],param[2]) for param in possibleLines]
    ylist = [chargeCollectionEfficiencyFunc(x,_V_0,_t_epi,_edl) for _V_0 in [V_0-V_0_e,V_0+V_0_e] for _t_epi in [t_epi-t_epi_e,t_epi+t_epi_e] for _edl in [edl-edl_e,edl+edl_e]]
    ax.fill_between(x, np.min(ylist,axis=0), np.max(ylist,axis=0),color=plot.colorPalette[6],zorder=-1)

def depletionWidthFunc(V,a,b):
    return (a*(V+b))**0.5
