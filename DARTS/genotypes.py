from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

DARTS = DARTS_V2

FTSO = Genotype(
    normal=[
        ('sep_conv_3x3', 0), 
        ('sep_conv_3x3', 1), 
        ('sep_conv_3x3', 0), 
        ('sep_conv_3x3', 1), 
        ('sep_conv_3x3', 0), 
        ('sep_conv_3x3', 1), 
        ('sep_conv_3x3', 0), 
        ('sep_conv_3x3', 1)
    ], 
    normal_concat=[2, 3, 4, 5], 
    reduce=[
        ('sep_conv_3x3', 1), 
        ('sep_conv_3x3', 0), 
        ('sep_conv_3x3', 1), 
        ('sep_conv_3x3', 2), 
        ('sep_conv_3x3', 1), 
        ('sep_conv_3x3', 3), 
        ('sep_conv_3x3', 0), 
        ('sep_conv_3x3', 1)], 
    reduce_concat=[2, 3, 4, 5]
)

GAEA = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 2), ('skip_connect', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 2), ('dil_conv_5x5', 1), ('sep_conv_5x5', 4), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))


PERMS = {'GAEA_PERM1': ([19, 8, 23, 9, 22, 11, 30, 24, 20, 14, 28, 13, 16, 5, 7, 26, 31, 4, 27, 25, 18, 10, 17, 0, 2, 3, 6, 21, 12, 1, 29, 15],
                        [5, 27, 9, 22, 14, 30, 29, 11, 4, 12, 3, 10, 28, 7, 21, 23, 31, 17, 0, 16, 19, 18, 25, 1, 24, 6, 15, 8, 13, 2, 26, 20]),
         'GAEA_PERM2': ([27, 17, 24, 30, 31, 6, 16, 10, 22, 0, 9, 11, 7, 4, 5, 2, 23, 19, 25, 18, 20, 28, 26, 12, 15, 13, 29, 21, 14, 1, 3, 8],
                        [13, 28, 20, 23, 14, 22, 26, 21, 9, 19, 25, 6, 1, 5, 12, 11, 24, 29, 7, 10, 3, 30, 16, 4, 2, 8, 27, 17, 15, 0, 18, 31]),
         'GAEA_PERM3': ([0, 3, 10, 4, 13, 8, 7, 25, 30, 12, 5, 9, 28, 16, 15, 20, 2, 14, 22, 23, 19, 6, 24, 18, 26, 17, 27, 11, 29, 1, 31, 21],
                        [21, 0, 25, 10, 11, 1, 29, 26, 7, 2, 31, 24, 22, 30, 28, 6, 9, 15, 23, 14, 5, 27, 13, 4, 19, 16, 3, 12, 18, 17, 8, 20]),
         'GAEA_PERM4': ([14, 30, 7, 13, 20, 0, 12, 25, 11, 27, 15, 31, 1, 6, 8, 24, 23, 3, 5, 19, 10, 16, 22, 4, 18, 21, 2, 9, 29, 26, 28, 17],
                        [21, 14, 3, 18, 22, 8, 29, 30, 16, 20, 15, 19, 27, 4, 11, 0, 12, 6, 13, 23, 28, 5, 7, 31, 2, 17, 24, 1, 9, 26, 25, 10])}
GAEA_PERM1 =  Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 2), ('skip_connect', 0), ('dil_conv_3x3', 2), ('skip_connect', 0), ('dil_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('dil_conv_5x5', 0), ('skip_connect', 1), ('dil_conv_5x5', 0), ('dil_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 1), ('skip_connect', 0)], reduce_concat=range(2, 6))
GAEA_PERM2 = Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 0), ('skip_connect', 2), ('dil_conv_5x5', 0), ('dil_conv_3x3', 2), ('skip_connect', 2), ('dil_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 0), ('dil_conv_3x3', 2), ('dil_conv_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 1)], reduce_concat=range(2, 6))
GAEA_PERM3 = Genotype(normal=[('dil_conv_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('dil_conv_5x5', 0), ('skip_connect', 3), ('dil_conv_5x5', 0), ('dil_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('avg_pool_3x3', 1), ('dil_conv_3x3', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 0), ('skip_connect', 1), ('dil_conv_3x3', 0), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))
GAEA_PERM4 = Genotype(normal=[('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 0), ('skip_connect', 2), ('dil_conv_5x5', 0), ('skip_connect', 3), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('dil_conv_5x5', 0), ('dil_conv_3x3', 3), ('dil_conv_3x3', 0)], reduce_concat=range(2, 6))
