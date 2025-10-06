Header =  {
  'Release'      : '2022',
  'Revision'     : '59056',
  'BuildDate'    : '29 Sep 2022',
  'CreationDate' : '12 Apr 2024',
  'CreationTime' : '12:57:48',
  'Creator'      : 'Dell',
  'Platform'     : '64 bit Windows',
  }

Description = '''
Macro file for GeoDict 2022
recorded at 12:57:48 on 12 Apr 2024
by T-MAC of SCUT
'''

def Tortuosity(MF_GeodesicTortuosity_ResultFileName):
    GeodesicTortuosity_args_1 = {
    'AnalyzeMode'    : 'Pore', # Possible values: Pore, Solid, ChosenMaterial, ChosenMaterialIDs, All
    'Material' : {
        'Type' : 'Undefined',
        },
    'MaterialIDs'    : 'NONE',
    'ResultFileName' : MF_GeodesicTortuosity_ResultFileName,
    'PeriodicX'      : False,
    'PeriodicY'      : False,
    'PeriodicZ'      : False,
    'Direction'      : 2,
    'OptimizePaths'  : False,
    }
    gd.runCmd("MatDict:GeodesicTortuosity", GeodesicTortuosity_args_1, Header['Release'])