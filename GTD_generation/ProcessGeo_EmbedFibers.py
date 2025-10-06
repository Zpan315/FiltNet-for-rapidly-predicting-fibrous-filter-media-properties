Header =  {
  'Release'      : '2022',
  'Revision'     : '59056',
  'BuildDate'    : '29 Sep 2022',
  'CreationDate' : '12 Apr 2024',
  'CreationTime' : '13:05:11',
  'Creator'      : 'Dell',
  'Platform'     : '64 bit Windows',
  }

Description = '''
Macro file for GeoDict 2022
recorded at 13:05:11 on 12 Apr 2024
by T-MAC of SCUT
'''




def EmbedFibers():
	Embed_args_1 = {
		'XMinus'     : 0,
		'XPlus'      : 0,
		'YMinus'     : 0,
		'YPlus'      : 0,
		'ZMinus'     : 20,
		'ZPlus'      : 20,
		'UseGAD'     : True,
		'MaterialID' : 0,
		}
	gd.runCmd("ProcessGeo:Embed", Embed_args_1, Header['Release'])
