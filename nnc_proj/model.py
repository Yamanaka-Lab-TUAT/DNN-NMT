import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

def network(x, r, test=False):
    # Input:x -> 1,128,128
    # Input_2:r -> 1
    # Convolution -> 16,42,42
    h = PF.convolution(x, 16, (7,7), (1,1), (3,3), name='Convolution')
    # BatchNormalization
    h = PF.batch_normalization(h, (1,), 0.8999999761581421, 9.999999747378752e-05, not test, name='BatchNormalization')
    # ELU
    h = F.elu(h, 1.0)
    # MaxPooling -> 16,14,14
    h = F.max_pooling(h, (3,3), (3,3))
    # Convolution_2 -> 32,6,6
    h = PF.convolution(h, 32, (5,5), (1,1), (2,2), name='Convolution_2')
    # BatchNormalization_2
    h = PF.batch_normalization(h, (1,), 0.8999999761581421, 9.999999747378752e-05, not test, name='BatchNormalization_2')
    # ELU_2
    h = F.elu(h, 1.0)
    # Convolution_3 -> 64,2,2
    h = PF.convolution(h, 64, (5,5), (0,0), name='Convolution_3')
    # BatchNormalization_7
    h = PF.batch_normalization(h, (1,), 0.8999999761581421, 9.999999747378752e-05, not test, name='BatchNormalization_7')
    # ELU_7
    h = F.elu(h, 1.0)
    # MaxPooling_2 -> 64,1,1
    h = F.max_pooling(h, (2,2), (2,2))
    # Reshape -> 64
    h = F.reshape(h, (h.shape[0],64,))

    # Concatenate -> 65
    h1 = F.concatenate(r, h, axis=1)

    # Affine_3 -> 512
    h2 = PF.affine(h1, (512,), name='Affine_3')

    # Affine_2 -> 512
    h3 = PF.affine(h1, (512,), name='Affine_2')
    # BatchNormalization_3
    h2 = PF.batch_normalization(h2, (1,), 0.8999999761581421, 9.999999747378752e-05, not test, name='BatchNormalization_3')
    # BatchNormalization_4
    h3 = PF.batch_normalization(h3, (1,), 0.8999999761581421, 9.999999747378752e-05, not test, name='BatchNormalization_4')
    # ELU_3
    h2 = F.elu(h2, 1.0)
    # ELU_4
    h3 = F.elu(h3, 1.0)
    # Affine_7
    h2 = PF.affine(h2, (512,), name='Affine_7')
    # Affine_6
    h3 = PF.affine(h3, (512,), name='Affine_6')
    # BatchNormalization_5
    h2 = PF.batch_normalization(h2, (1,), 0.8999999761581421, 9.999999747378752e-05, not test, name='BatchNormalization_5')
    # BatchNormalization_6
    h3 = PF.batch_normalization(h3, (1,), 0.8999999761581421, 9.999999747378752e-05, not test, name='BatchNormalization_6')
    # ELU_5
    h2 = F.elu(h2, 1.0)
    # ELU_6
    h3 = F.elu(h3, 1.0)
    # Affine_5 -> 50,2
    h2 = PF.affine(h2, (50,2), name='Affine_5')
    # Affine_4 -> 2,2
    h3 = PF.affine(h3, (2,2), name='Affine_4')
    return h2, h3
