#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
#include "ap_int.h"

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_batchnorm.h"
#include "nnet_utils/nnet_batchnorm_stream.h"
#include "nnet_utils/nnet_conv1d.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
#include "nnet_utils/nnet_pooling.h"
#include "nnet_utils/nnet_pooling_stream.h"
#include "nnet_utils/nnet_sepconv1d_stream.h"

// hls-fpga-machine-learning insert weights
#include "weights/s2.h"
#include "weights/b2.h"
#include "weights/w26.h"
#include "weights/b26.h"
#include "weights/w27.h"
#include "weights/b27.h"
#include "weights/w28.h"
#include "weights/b28.h"
#include "weights/w14.h"
#include "weights/b14.h"
#include "weights/w17.h"
#include "weights/b17.h"
#include "weights/w20.h"
#include "weights/b20.h"
#include "weights/w23.h"
#include "weights/b23.h"

// hls-fpga-machine-learning insert layer-config
// batchnorm
struct config2 : nnet::batchnorm_config {
    static const unsigned n_in = N_INPUT_1_1*N_INPUT_2_1;
    static const unsigned n_filt = 12;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in, reuse_factor);
    static const bool store_weights_in_bram = false;
    typedef batchnorm_default_t bias_t;
    typedef batchnorm_default_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// Dense_phi1
struct config26_mult : nnet::dense_config {
    static const unsigned n_in = 12;
    static const unsigned n_out = 16;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef model_default_t accum_t;
    typedef dense_phi1_bias_t bias_t;
    typedef dense_phi1_weight_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config26 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 32;
    static const unsigned n_chan = 12;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 16;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 32;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 32;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 32;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::fill_buffer_26<data_T, CONFIG_T>;
    typedef model_default_t accum_t;
    typedef dense_phi1_bias_t bias_t;
    typedef dense_phi1_weight_t weight_t;
    typedef config26_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_unscaled<K, S, W>;
    template<class data_T, class res_T, class CONFIG_T>
    using pointwise_conv = nnet::pointwise_conv_26<data_T, res_T, CONFIG_T>;
};
const ap_uint<config26::filt_width> config26::pixels[] = {0};

// Activation_phi1
struct relu_config5 : nnet::activ_config {
    static const unsigned n_in = 512;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef Activation_phi1_table_t table_t;
};

// Dense_phi2
struct config27_mult : nnet::dense_config {
    static const unsigned n_in = 16;
    static const unsigned n_out = 16;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef model_default_t accum_t;
    typedef dense_phi2_bias_t bias_t;
    typedef dense_phi2_weight_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config27 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 32;
    static const unsigned n_chan = 16;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 16;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 32;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 32;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 32;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::fill_buffer_27<data_T, CONFIG_T>;
    typedef model_default_t accum_t;
    typedef dense_phi2_bias_t bias_t;
    typedef dense_phi2_weight_t weight_t;
    typedef config27_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_unscaled<K, S, W>;
    template<class data_T, class res_T, class CONFIG_T>
    using pointwise_conv = nnet::pointwise_conv_27<data_T, res_T, CONFIG_T>;
};
const ap_uint<config27::filt_width> config27::pixels[] = {0};

// Activation_phi2
struct relu_config8 : nnet::activ_config {
    static const unsigned n_in = 512;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef Activation_phi2_table_t table_t;
};

// Dense_phi3
struct config28_mult : nnet::dense_config {
    static const unsigned n_in = 16;
    static const unsigned n_out = 16;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef model_default_t accum_t;
    typedef dense_phi3_bias_t bias_t;
    typedef dense_phi3_weight_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config28 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 32;
    static const unsigned n_chan = 16;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 16;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 32;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 32;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 32;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::fill_buffer_28<data_T, CONFIG_T>;
    typedef model_default_t accum_t;
    typedef dense_phi3_bias_t bias_t;
    typedef dense_phi3_weight_t weight_t;
    typedef config28_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_unscaled<K, S, W>;
    template<class data_T, class res_T, class CONFIG_T>
    using pointwise_conv = nnet::pointwise_conv_28<data_T, res_T, CONFIG_T>;
};
const ap_uint<config28::filt_width> config28::pixels[] = {0};

// Activation_phi3
struct relu_config11 : nnet::activ_config {
    static const unsigned n_in = 512;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef Activation_phi3_table_t table_t;
};

// qActivationForPool
struct linear_config12 : nnet::activ_config {
    static const unsigned n_in = 512;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef qActivationForPool_table_t table_t;
};

// avgpool
struct config13 : nnet::pooling1d_config {
    static const unsigned n_in = 32;
    static const unsigned n_filt = 16;
    static const nnet::Pool_Op pool_op = nnet::Average;
    static const unsigned reuse_factor = 1;
    typedef avgpool_default_t accum_t;
};

// qDense_rho1
struct config14 : nnet::dense_config {
    static const unsigned n_in = 16;
    static const unsigned n_out = 16;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 2;
    static const unsigned n_nonzeros = 254;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias14_t bias_t;
    typedef weight14_t weight_t;
    typedef layer14_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// qActivation_rho1
struct relu_config16 : nnet::activ_config {
    static const unsigned n_in = 16;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef qActivation_rho1_table_t table_t;
};

// Dense_rho2
struct config17 : nnet::dense_config {
    static const unsigned n_in = 16;
    static const unsigned n_out = 32;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 1;
    static const unsigned n_nonzeros = 511;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias17_t bias_t;
    typedef weight17_t weight_t;
    typedef layer17_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// Activation_rho2
struct relu_config19 : nnet::activ_config {
    static const unsigned n_in = 32;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef Activation_rho2_table_t table_t;
};

// qDense_rho3
struct config20 : nnet::dense_config {
    static const unsigned n_in = 32;
    static const unsigned n_out = 16;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 5;
    static const unsigned n_nonzeros = 507;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias20_t bias_t;
    typedef weight20_t weight_t;
    typedef layer20_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// qActivation_rho3
struct relu_config22 : nnet::activ_config {
    static const unsigned n_in = 16;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef qActivation_rho3_table_t table_t;
};

// qDense_rho4
struct config23 : nnet::dense_config {
    static const unsigned n_in = 16;
    static const unsigned n_out = 1;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 16;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias23_t bias_t;
    typedef weight23_t weight_t;
    typedef layer23_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// output_class
struct sigmoid_config25 : nnet::activ_config {
    static const unsigned n_in = 1;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef output_class_table_t table_t;
};


#endif
