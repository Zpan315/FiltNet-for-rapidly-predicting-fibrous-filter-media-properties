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


def FilterEfficiency(MF_FilterEfficiency_ResultFileName,offsetValue):
    Embed_args_1 = {
    'XMinus'     : 0,
    'XPlus'      : 0,
    'YMinus'     : 0,
    'YPlus'      : 0,
    'ZMinus'     : 125,
    'ZPlus'      : 125,
    'UseGAD'     : False,
    'MaterialID' : 0,
    }
    gd.runCmd("ProcessGeo:Embed", Embed_args_1, Header['Release'])

    Efficiency_args_1 = {
    'FlowSolver' : {
        'SlipLength'             : 6.6e-08,
        'Solver'                 : 'LIR',      # Possible values: LOAD, EJ, SIMPLE_FFT, LIR
        'MeanVelocity'           : 0.105,
        'FlowPDE'                : 'STOKES',   # Possible values: STOKES, NAVIER_STOKES
        'BCFlowDirection'        : 'Periodic', # Possible values: Periodic, Symmetric, VinPout, Mirror, Undefined
        'BCTangentialDirections' : 'Periodic', # Possible values: Periodic, Symmetric, NoSlip, NoSlipFixedSize, Encase, EncaseFixedSize, Undefined
        'AnalyzeGeometry'        : True,
        'LIR' : {
        'UseTolerance'          : False,
        'Tolerance'             : 0.001,
        'UseErrorBound'         : True,
        'ErrorBound'            : 0.01,
        'MaxNumberOfIterations' : 100000,
        'MaximalSolverRunTime'  : (240, 'h'),
        'UseMaxIterations'      : False,
        'UseMaxTime'            : False,
        'UseLateral'            : False,
        'UseMultigrid'          : False,
        'Optimization'          : 'Speed',
        'GridType'              : 'LIR-Tree',
        'Relaxation'            : 1,
        'UseKrylov'             : 'Automatic', # Possible values: Automatic, Enabled, Disabled
        'Refinement'            : 'ENABLED',   # Possible values: ENABLED, DISABLED, MANUAL
        'WriteCompressedFields' : True,
        },
        },
    'ParticleInitialPosition' : {
        'UseInitialParticleFile'       : False,
        'PositionMode'                 : 'InflowPlane', # Possible values: InflowPlane, Box, Sphere, ChosenMaterialIDs, Everywhere
        'InitialPositionInPlaneOffset' : offsetValue,
        'ParticlePositionWeights'      : 'Uniform',     # Possible values: Uniform, Velocity, Distance, GivenField
        },
    'ParticleInjection'          : 'BatchStart',     # Possible values: BatchStart, BatchContinuous, Continuous
    'ParticleSliding'            : 'None',           # Possible values: None, Sieving, SievingAndHamaker
    'ParticleEndPosition' : {
        'UseEndMaterial' : False,
        'MaterialIDs'    : 'NONE',
        },
    'ConstituentMaterials' : {
        'Temperature' : (293.15, 'K'),
        'Material00' : {
        'Name'            : 'Air',
        'Type'            : 'Fluid', # Possible values: Fluid, Solid, Porous, Undefined
        'Information'     : '',
        'FluidProperties' : {
            'Density'   : (1.204, 'kg/m^3'),
            'Viscosity' : (1.834e-05, 'kg/(ms)'),
            },
        },
        'Material01' : {
        'Name'        : 'Manual',
        'Type'        : 'Solid',  # Possible values: Fluid, Solid, Porous, Undefined
        'Information' : '',
        },
        'Material02' : {
        'Name'        : 'Manual',
        'Type'        : 'Solid',   # Possible values: Fluid, Solid, Porous, Undefined
        'Information' : 'Overlap',
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
        'ChosenFluid' : {
        'Fluid' : 'Air',
        },
        },
    'UseEStatic'                 : False,
    'ElectrostaticSurfaceCharge' : (1e-06, 'C/m^2'),
    'ParticlesPerType'           : 1000,
    'ParticleDistribution' : {
        'NumberOfParticleTypes' : 19,
        'ProbabilityType'       : 'COUNT',
        'MaterialIDs'           : [0, 1],
        'Charge'                : 'NONE',        # Possible values: NONE, CALCULATED, PER_TYPE, RANDOM
        'Density'               : 'CONSTANT',    # Possible values: NONE, CALCULATED, PER_TYPE, CONSTANT, UDF
        'Diameter'              : 'PER_TYPE',    # Possible values: NONE, CALCULATED, PER_TYPE, CONSTANT, UDF
        'CollisionDiameter'     : 'NONE',        # Possible values: NONE, CALCULATED, PER_TYPE, CONSTANT, UDF
        'DepositionDiameter'    : 'NONE',        # Possible values: NONE, CALCULATED, PER_TYPE, CONSTANT, UDF
        'Diffusivity'           : 'CALCULATED',  # Possible values: NONE, CALCULATED, PER_TYPE, CONSTANT, UDF
        'DensityValue'          : (2165, 'g/l'),
        'MaterialID0' : {
        'CollisionModel'      : 'COLLISION_CAUGHT', # Possible values: COLLISION_CAUGHT, COLLISION_HAMAKER, COLLISION_SIEVING, COLLISION_NMR, COLLISION_ADSORPTION, COLLISION_UDF
        'CollisionParameters' : 'PER_TYPE',         # Possible values: NONE, CALCULATED, PER_TYPE, CONSTANT, UDF
        'MaximalCapacity'     : 0.5,
        'PassThroughModel'    : 'ALL_PASS',         # Possible values: ALL_PASS, IMPASSABLE, CONST_ABSORPTION, CLOGGING, CONST_EFFICIENCY, VELOCITY_DEP_EFFICIENCY, REFLECTION_PROB, UDF
        },
        'MaterialID1' : {
        'CollisionModel'      : 'COLLISION_CAUGHT', # Possible values: COLLISION_CAUGHT, COLLISION_HAMAKER, COLLISION_SIEVING, COLLISION_NMR, COLLISION_ADSORPTION, COLLISION_UDF
        'CollisionParameters' : 'PER_TYPE',         # Possible values: NONE, CALCULATED, PER_TYPE, CONSTANT, UDF
        'MaximalCapacity'     : 0,
        'PassThroughModel'    : 'IMPASSABLE',       # Possible values: ALL_PASS, IMPASSABLE, CONST_ABSORPTION, CLOGGING, CONST_EFFICIENCY, VELOCITY_DEP_EFFICIENCY, REFLECTION_PROB, UDF
        },
        'ParticleType1' : {
            'Probability' : 0.05263157895,
            'Diameter'    : (1e-08, 'm'),
            'MaterialID0' : {
                'CollisionParameters' : 'NONE',
                },
            'MaterialID1' : {
                'CollisionParameters' : 'NONE',
                },
            },
        'ParticleType2' : {
            'Probability' : 0.05263157895,
            'Diameter'    : (2e-08, 'm'),
            'MaterialID0' : {
                'CollisionParameters' : 'NONE',
                },
            'MaterialID1' : {
                'CollisionParameters' : 'NONE',
                },
            },
        'ParticleType3' : {
            'Probability' : 0.05263157895,
            'Diameter'    : (3e-08, 'm'),
            'MaterialID0' : {
                'CollisionParameters' : 'NONE',
                },
            'MaterialID1' : {
                'CollisionParameters' : 'NONE',
                },
            },
        'ParticleType4' : {
            'Probability' : 0.05263157895,
            'Diameter'    : (4e-08, 'm'),
            'MaterialID0' : {
                'CollisionParameters' : 'NONE',
                },
            'MaterialID1' : {
                'CollisionParameters' : 'NONE',
                },
            },
        'ParticleType5' : {
            'Probability' : 0.05263157895,
            'Diameter'    : (5e-08, 'm'),
            'MaterialID0' : {
                'CollisionParameters' : 'NONE',
                },
            'MaterialID1' : {
                'CollisionParameters' : 'NONE',
                },
            },
        'ParticleType6' : {
            'Probability' : 0.05263157895,
            'Diameter'    : (6e-08, 'm'),
            'MaterialID0' : {
                'CollisionParameters' : 'NONE',
                },
            'MaterialID1' : {
                'CollisionParameters' : 'NONE',
                },
            },
        'ParticleType7' : {
            'Probability' : 0.05263157895,
            'Diameter'    : (7e-08, 'm'),
            'MaterialID0' : {
                'CollisionParameters' : 'NONE',
                },
            'MaterialID1' : {
                'CollisionParameters' : 'NONE',
                },
            },
        'ParticleType8' : {
            'Probability' : 0.05263157895,
            'Diameter'    : (8e-08, 'm'),
            'MaterialID0' : {
                'CollisionParameters' : 'NONE',
                },
            'MaterialID1' : {
                'CollisionParameters' : 'NONE',
                },
            },
        'ParticleType9' : {
            'Probability' : 0.05263157895,
            'Diameter'    : (9e-08, 'm'),
            'MaterialID0' : {
                'CollisionParameters' : 'NONE',
                },
            'MaterialID1' : {
                'CollisionParameters' : 'NONE',
                },
            },
        'ParticleType10' : {
            'Probability' : 0.05263157895,
            'Diameter'    : (1e-07, 'm'),
            'MaterialID0' : {
                'CollisionParameters' : 'NONE',
                },
            'MaterialID1' : {
                'CollisionParameters' : 'NONE',
                },
            },
        'ParticleType11' : {
            'Probability' : 0.05263157895,
            'Diameter'    : (2e-07, 'm'),
            'MaterialID0' : {
                'CollisionParameters' : 'NONE',
                },
            'MaterialID1' : {
                'CollisionParameters' : 'NONE',
                },
            },
        'ParticleType12' : {
            'Probability' : 0.05263157895,
            'Diameter'    : (3e-07, 'm'),
            'MaterialID0' : {
                'CollisionParameters' : 'NONE',
                },
            'MaterialID1' : {
                'CollisionParameters' : 'NONE',
                },
            },
        'ParticleType13' : {
            'Probability' : 0.05263157895,
            'Diameter'    : (4e-07, 'm'),
            'MaterialID0' : {
                'CollisionParameters' : 'NONE',
                },
            'MaterialID1' : {
                'CollisionParameters' : 'NONE',
                },
            },
        'ParticleType14' : {
            'Probability' : 0.05263157895,
            'Diameter'    : (5e-07, 'm'),
            'MaterialID0' : {
                'CollisionParameters' : 'NONE',
                },
            'MaterialID1' : {
                'CollisionParameters' : 'NONE',
                },
            },
        'ParticleType15' : {
            'Probability' : 0.05263157895,
            'Diameter'    : (6e-07, 'm'),
            'MaterialID0' : {
                'CollisionParameters' : 'NONE',
                },
            'MaterialID1' : {
                'CollisionParameters' : 'NONE',
                },
            },
        'ParticleType16' : {
            'Probability' : 0.05263157895,
            'Diameter'    : (7e-07, 'm'),
            'MaterialID0' : {
                'CollisionParameters' : 'NONE',
                },
            'MaterialID1' : {
                'CollisionParameters' : 'NONE',
                },
            },
        'ParticleType17' : {
            'Probability' : 0.05263157895,
            'Diameter'    : (8e-07, 'm'),
            'MaterialID0' : {
                'CollisionParameters' : 'NONE',
                },
            'MaterialID1' : {
                'CollisionParameters' : 'NONE',
                },
            },
        'ParticleType18' : {
            'Probability' : 0.05263157895,
            'Diameter'    : (9e-07, 'm'),
            'MaterialID0' : {
                'CollisionParameters' : 'NONE',
                },
            'MaterialID1' : {
                'CollisionParameters' : 'NONE',
                },
            },
        'ParticleType19' : {
            'Probability' : 0.05263157895,
            'Diameter'    : (1e-06, 'm'),
            'MaterialID0' : {
                'CollisionParameters' : 'NONE',
                },
            'MaterialID1' : {
                'CollisionParameters' : 'NONE',
                },
            },
        },
    'Parallelization' : {
        'Mode' : 'LOCAL_MAX', # Possible values: Sequential, LOCAL_THREADS, LOCAL_MPI, LOCAL_MAX, CLUSTER, Undefined
        },
    'FilterSolver' : {
        'MaximalTime'          : 1000,
        'CunninghamCorrection' : True,
        'CunninghamLambda'     : 6.6e-08,
        },
    'EStaticSolver' : {
        'DirichletBoundaryOffset' : 50,
        },
    'BrownianMotion'             : True,
    'UseParticleMotionUDF'       : False,
    'ParticleMotionUDFFileName'  : '',
    'RandomSeed'                 : 42,
    'ReflectParticleAtInflow'    : True,
    'ResultFileName'             : MF_FilterEfficiency_ResultFileName,
    'Output' : {
        'KeepAllFiles'         : False,
        'KeepGPPMode'          : 'Remove',       # Possible values: Keep, Remove
        'KeepInitGPPMode'      : 'Remove',       # Possible values: Keep, Remove
        'KeepESTMode'          : 'Remove',       # Possible values: Keep, Remove
        'KeepVAPMode'          : 'Remove',       # Possible values: Keep, Remove
        'KeepGPTMode'          : 'Remove',       # Possible values: Keep, Remove
        'WriteCollisionPoints' : False,
        'TrajectoryPrecision'  : (0.5, 'Voxel'),
        },
    }
    gd.runCmd("FilterDict:Efficiency", Efficiency_args_1, Header['Release'])