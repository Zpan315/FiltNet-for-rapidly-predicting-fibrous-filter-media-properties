import numpy as np

def PoreSizeDistribution(Pore_PSD):
    full_list = np.zeros(200)
    Pore_PSD = [float(x) for x in Pore_PSD]
    Original_list = Pore_PSD

    length = min(len(Original_list), len(full_list))
    full_list[:length] = Original_list[:length]

    return(full_list)

def PoreSizeDistributionCumulative(Pore_PSD):
    full_list = np.ones(200)
    Pore_PSD = [float(x) for x in Pore_PSD]
    Original_list = Pore_PSD

    length = min(len(Original_list), len(full_list))
    full_list[:length] = Original_list[:length]

    return(full_list) 