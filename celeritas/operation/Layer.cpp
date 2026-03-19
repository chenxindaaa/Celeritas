#include "Layer.h"

namespace eCEL {

template class Layer<int>;
template class Layer<float>;
template class Layer<double>;

template class AddLayer<int>;
template class AddLayer<float>;
template class AddLayer<double>;

template class EmbeddingLayer<int>;
template class EmbeddingLayer<float>;
template class EmbeddingLayer<double>;

template class RmsNormLayer<int>;
template class RmsNormLayer<float>;
template class RmsNormLayer<double>;

template class MatmultLayer<int>;
template class MatmultLayer<float>;
template class MatmultLayer<double>;

}  // namespace eCEL
