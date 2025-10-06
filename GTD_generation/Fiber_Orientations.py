Header =  {
  'Release'      : '2022',
  'Revision'     : '59056',
  'BuildDate'    : '29 Sep 2022',
  'CreationDate' : '12 Apr 2024',
  'CreationTime' : '13:48:19',
  'Creator'      : 'Dell',
  'Platform'     : '64 bit Windows',
  }

Description = '''
Macro file for GeoDict 2022
recorded at 13:48:19 on 12 Apr 2024
by T-MAC of SCUT
'''

def Fiber_Orientations(MF_FiberOrientations_ResultFileName):
    FiberParameterEstimator_args_1 = {
    'ResultFileName'              : MF_FiberOrientations_ResultFileName,
    'Method'                      : 'SLD',   # Possible values: PCA, SLD
    'WriteGOF'                    : False,
    'WinSizeAutomatic'            : False,
    'WinSize'                     : 32,
    'NumberOfOrientationTensorsX' : 1,
    'NumberOfOrientationTensorsY' : 1,
    'NumberOfOrientationTensorsZ' : 1,
    'AnalyzeMode'                 : 'Solid', # Possible values: Pore, Solid, ChosenMaterial, ChosenMaterialIDs, All
    'Material' : {
        'Type' : 'Undefined',
        },
    'MaterialIDs'                 : 'NONE',
    }
    gd.runCmd("FiberFind:FiberParameterEstimator", FiberParameterEstimator_args_1, Header['Release'])