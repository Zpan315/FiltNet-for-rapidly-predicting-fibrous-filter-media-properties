Header =  {
  'Release'      : '2022',
  'Revision'     : '59056',
  'BuildDate'    : '29 Sep 2022',
  'CreationDate' : '12 Apr 2024',
  'CreationTime' : '12:54:43',
  'Creator'      : 'Dell',
  'Platform'     : '64 bit Windows',
  }

Description = '''
Macro file for GeoDict 2022
recorded at 12:54:43 on 12 Apr 2024
by T-MAC of SCUT 
'''

def PaperGeo_Circle(New_VoxelLength,New_NZ,New_LayerHeight,New_SolidVolumePercentage,New_RandomSeed,New_ResultFileName,New_DiameterDistribution,New_DiameterSD,New_DiameterWD):
    Create_args_1 = {
    'Domain' : {
        'PeriodicX'         : False,
        'PeriodicY'         : False,
        'PeriodicZ'         : False,
        'OriginX'           : (0, 'm'),
        'OriginY'           : (0, 'm'),
        'OriginZ'           : (0, 'm'),
        'VoxelLength'       : (New_VoxelLength, 'm'),
        'DomainMode'        : 'VoxelNumber',   # Possible values: VoxelNumber, Length, VoxelNumberAndLength
        'NX'                : 400,
        'NY'                : 400,
        'NZ'                : New_NZ,
        'Material' : {
        'Name'        : 'Air',
        'Type'        : 'Fluid', # Possible values: Fluid, Solid, Porous, Undefined
        'Information' : '',
        },
        'OverlapMode'       : 'GivenMaterial', # Possible values: OverlapMaterial, NewMaterial, OldMaterial, GivenMaterial
        'OverlapMaterialID' : 2,
        'NumOverlapRules'   : 1,
        'OverlapRule1' : {
        'MaterialID1'       : 1,
        'MaterialID2'       : 1,
        'OverlapMaterialID' : 1,
        },
        'HollowMaterialID'  : 0,
        'PostProcessing' : {
        'ResolveOverlap'    : False,
        'MarkContactVoxels' : False,
        'ContactMaterialID' : 2,
        },
        },
    'LayerHeight'              : New_LayerHeight,
    'StoppingCriterion'        : 'SolidVolumePercentage',# Possible values: SolidVolumePercentage, NumberOfObjects, Grammage, Density, WeightPercentage, FillToRim, SVP, Number
    'SolidVolumePercentage'    : (New_SolidVolumePercentage, '%'),
    'Grammage'                 : (40, 'g/m^2'),
    'SaveGadStep'              : 10,
    'RecordIntermediateResult' : False,
    'InExisting'               : False,
    'KeepStructure'            : False,
    'MaximalTime'              : (6, 'h'),
    'OverlapMode'              : 'AllowOverlap',  # Possible values: AllowOverlap, RemoveOverlap, ForceConnection, IsolationDistance, ProhibitWithExisting, ProhibitOverlap, MatchSVFDistribution
    'NumberOfObjects'          : 1000,
    'RemoveOverlap' : {
        'Iterations'        : 1000,
        'OverlapSVP'        : (0, '%'),
        'SVPUnchanged'      : 20,
        'AllowShift'        : True,
        'AllowRotation'     : False,
        'AllowDeformation'  : True,
        'NumberOfShifts'    : 10,
        'ShiftDistance'     : (2, 'Voxel'),
        'NumberOfRotations' : 20,
        'MaximalAngle'      : 60,
        'DistanceMode'      : 'Touch',      # Possible values: Touch, Overlap, Isolation, AvoidContact
        'IsolationDistance' : (0, 'm'),
        'MaximalOverlap'    : (0, 'm'),
        },
    'IsolationDistance'        : (-1.2e-05, 'm'),
    'PercentageType'           : 1,
    'RandomSeed'               : New_RandomSeed,
    'ResultFileName'           : New_ResultFileName,
    'MatrixDensity'            : (0, 'g/cm^3'),
    'MaterialMode'             : 'Material',      # Possible values: Material, MaterialID
    'MaterialIDMode'           : 'MaterialIDPerObjectType',# Possible values: MaterialIDPerObjectType, MaterialIDPerMaterial
    'OverlapMaterial' : {
        'Name'        : 'Manual',
        'Type'        : 'Solid',   # Possible values: Fluid, Solid, Porous, Undefined
        'Information' : 'Overlap',
        },
    'ContactMaterial' : {
        'Name'        : 'Manual',
        'Type'        : 'Solid',    # Possible values: Fluid, Solid, Porous, Undefined
        'Information' : 'Contacts',
        },
    'NumberOfFiberTypes'       : 1,
    'FiberType1' : {
        'Material' : {
        'Name'        : 'Manual',
        'Type'        : 'Solid',  # Possible values: Fluid, Solid, Porous, Undefined
        'Information' : '',
        },
        'Probability'                   : 1,
        'SpecificWeight'                : (1.5, 'g/cm^3'),
        'Type'                          : 'ShortCircularPaperFiberGenerator',
        'DiameterDistribution' : {
        'Type'               : 'Gaussian', # Possible values: Constant, UniformlyInInterval, Gaussian, Table, LogNormal
        'MeanValue'          : New_DiameterDistribution,
        'StandardDeviation'  : New_DiameterSD,
        'Bound'              : New_DiameterWD,
        'CutOffDistribution' : True,
        },
        'CenterDistribution' : {
        'Type' : 'Uniformly', # Possible values: Uniformly, OnStructure, UniformlyInBox, DensityDistribution
        },
        'LengthDistribution' : {
        'Type'  : 'Constant', # Possible values: Constant, UniformlyInInterval, Gaussian, Table, LogNormal
        'Value' : 0.001,
        },
        'RoundedEnd'                    : True,
        'TorsionStartAngleDistribution' : {
        'Type'  : 'Constant', # Possible values: Constant, UniformlyInInterval, Gaussian, Table, LogNormal
        'Value' : 0,
        },
        'TorsionMaxChange'              : 0,
        'RotationAngleDistribution' : {
        'Type'  : 'Constant', # Possible values: Constant, UniformlyInInterval, Gaussian, Table, LogNormal
        'Value' : 0,
        },
        'OrientationMode'               : 0,
        'Anisotropy'                    : 1,
        'OscillationMode'               : 2,
        'Stiffness'                     : 1e-05,
        },
    'Temperature'              : (293.15, 'K'),
    }
    gd.runCmd("PaperGeo:Create", Create_args_1, Header['Release'])