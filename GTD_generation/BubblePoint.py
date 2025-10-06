Header =  {
  'Release'      : '2022',
  'Revision'     : '59056',
  'BuildDate'    : '29 Sep 2022',
  'CreationDate' : '12 Apr 2024',
  'CreationTime' : '12:59:47',
  'Creator'      : 'Dell',
  'Platform'     : '64 bit Windows',
  }

Description = '''
Macro file for GeoDict 2022
recorded at 12:59:47 on 12 Apr 2024
by T-MAC of SCUT
'''

def BubblePoint(MF_BubblePoint_ResultFileName):
    BubblePoint_args_1 = {
    'HydraulicDiameterEstimation' : True,
    'RemoveSpikes'                : False,
    'HighRes'                     : False,
    'ResultFileName'              : MF_BubblePoint_ResultFileName,
    'ContactAngle'                : (0, 'Deg'),
    'SurfaceTension'              : (0.07275, 'N/m'),
    'Direction'                   : 2,
    'BoundaryConditionX'          : 'Encase',          # Possible values: Symmetric, Periodic, Encase
    'BoundaryConditionY'          : 'Encase',          # Possible values: Symmetric, Periodic, Encase
    'BoundaryConditionZ'          : 'Symmetric',       # Possible values: Symmetric, Periodic, Encase
    }
    gd.runCmd("PoroDict:BubblePoint", BubblePoint_args_1, Header['Release'])