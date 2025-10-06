Header =  {
  'Release'      : '2022',
  'Revision'     : '59056',
  'BuildDate'    : '29 Sep 2022',
  'CreationDate' : '12 Apr 2024',
  'CreationTime' : '12:58:45',
  'Creator'      : 'Dell',
  'Platform'     : '64 bit Windows',
  }

Description = '''
Macro file for GeoDict 2022
recorded at 12:58:45 on 12 Apr 2024
by T-MAC of SCUT
'''


def Porosimetry(MF_1DStatistics_ResultFileName):
    PorosimetryPSD_args_1 = {
    'FluidSmallPores' : {
        'Type'        : 'Fluid',
        'Name'        : 'Air',
        'Information' : 'Small Pores',
        },
    'ResultFileName'           : MF_1DStatistics_ResultFileName,
    'FluidLargePores' : {
        'Type'        : 'Fluid',
        'Name'        : 'Water',
        'Information' : 'Large Pores',
        },
    'BinSize'                  : (2, 'Voxels'),
    'Fractions'                : ([10, 50, 90], '%'),
    'WriteAllGdtFiles'         : False,
    'IntrusionDirectionXMinus' : False,
    'IntrusionDirectionXPlus'  : False,
    'IntrusionDirectionYMinus' : False,
    'IntrusionDirectionYPlus'  : False,
    'IntrusionDirectionZMinus' : True,
    'IntrusionDirectionZPlus'  : False,
    'WriteGSD'                 : False,
    'ConstituentMaterials' : {
        'Temperature' : (293.15, 'K'),
        'Material00' : {
        'Type'        : 'Fluid',
        'Name'        : 'Air',
        'Information' : '',
        },
        'Material01' : {
        'Type'            : 'Solid',
        'Name'            : 'Manual',
        'Information'     : '',
        'SolidProperties' : {
            'Density' : (1500, 'kg/m^3'),
            },
        },
        'Material02' : {
        'Type'            : 'Solid',
        'Name'            : 'Manual',
        'Information'     : 'Overlap',
        'SolidProperties' : {
            'Density' : (0, 'kg/m^3'),
            },
        },
        'Material03' : {
        'Type' : 'Undefined',
        },
        'Material04' : {
        'Type' : 'Undefined',
        },
        'Material05' : {
        'Type' : 'Undefined',
        },
        'Material06' : {
        'Type' : 'Undefined',
        },
        'Material07' : {
        'Type' : 'Undefined',
        },
        'Material08' : {
        'Type' : 'Undefined',
        },
        'Material09' : {
        'Type' : 'Undefined',
        },
        'Material10' : {
        'Type' : 'Undefined',
        },
        'Material11' : {
        'Type' : 'Undefined',
        },
        'Material12' : {
        'Type' : 'Undefined',
        },
        'Material13' : {
        'Type' : 'Undefined',
        },
        'Material14' : {
        'Type' : 'Undefined',
        },
        'Material15' : {
        'Type' : 'Undefined',
        },
        },
    }
    gd.runCmd("PoroDict:PorosimetryPSD", PorosimetryPSD_args_1, Header['Release'])