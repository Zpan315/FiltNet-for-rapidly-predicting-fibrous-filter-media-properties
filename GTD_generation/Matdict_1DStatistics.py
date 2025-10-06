Header =  {
  'Release'      : '2022',
  'Revision'     : '59056',
  'BuildDate'    : '29 Sep 2022',
  'CreationDate' : '12 Apr 2024',
  'CreationTime' : '12:56:59',
  'Creator'      : 'Dell',
  'Platform'     : '64 bit Windows',
  }

Description = '''
Macro file for GeoDict 2022
recorded at 12:56:59 on 12 Apr 2024
by T-MAC of SCUT
'''

def MatDict_1DStatistics(MF_1DStatistics_ResultFileName):
    SaveStatistics_args_1 = {
    'BackgroundMode'        : 'Pore',
    'DirectionX'            : True,
    'DirectionY'            : False,
    'DirectionZ'            : True,
    'AnalyzeLayerThickness' : False,
    'ResultFileName'        : MF_1DStatistics_ResultFileName,
    'MethodsEnabled'        : 0,
    'WriteFile'             : False,
    }
    gd.runCmd("MatDict:SaveStatistics", SaveStatistics_args_1, Header['Release'])