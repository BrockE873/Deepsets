#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 32
#define N_INPUT_2_1 12
#define N_INPUT_1_1 32
#define N_INPUT_2_1 12
#define N_OUTPUTS_26 32
#define N_FILT_26 16
#define N_LAYER_1_3 32
#define N_LAYER_2_3 16
#define N_OUTPUTS_27 32
#define N_FILT_27 16
#define N_LAYER_1_6 32
#define N_LAYER_2_6 16
#define N_OUTPUTS_28 32
#define N_FILT_28 16
#define N_LAYER_1_9 32
#define N_LAYER_2_9 16
#define N_LAYER_1_9 32
#define N_LAYER_2_9 16
#define N_FILT_13 16
#define N_LAYER_14 16
#define N_LAYER_14 16
#define N_LAYER_17 32
#define N_LAYER_17 32
#define N_LAYER_20 16
#define N_LAYER_20 16
#define N_LAYER_23 1
#define N_LAYER_23 1

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<20,9,AP_RND,AP_SAT,0> input_t;
typedef ap_fixed<20,9,AP_RND,AP_SAT,0> layer2_t;
typedef ap_fixed<20,9,AP_RND,AP_SAT,0> batchnorm_default_t;
typedef ap_fixed<20,9> model_default_t;
typedef ap_fixed<20,9> layer26_t;
typedef ap_fixed<8,1> dense_phi1_weight_t;
typedef ap_fixed<8,1> dense_phi1_bias_t;
typedef ap_ufixed<8,0,AP_RND_CONV,AP_SAT,0> layer5_t;
typedef ap_fixed<18,8> Activation_phi1_table_t;
typedef ap_fixed<20,9> layer27_t;
typedef ap_fixed<8,1> dense_phi2_weight_t;
typedef ap_fixed<8,1> dense_phi2_bias_t;
typedef ap_ufixed<8,0,AP_RND_CONV,AP_SAT,0> layer8_t;
typedef ap_fixed<18,8> Activation_phi2_table_t;
typedef ap_fixed<20,9> layer28_t;
typedef ap_fixed<8,1> dense_phi3_weight_t;
typedef ap_fixed<8,1> dense_phi3_bias_t;
typedef ap_ufixed<8,0,AP_RND_CONV,AP_SAT,0> layer11_t;
typedef ap_fixed<18,8> Activation_phi3_table_t;
typedef ap_fixed<20,9,AP_RND,AP_SAT,0> layer12_t;
typedef ap_fixed<18,8> qActivationForPool_table_t;
typedef ap_fixed<30,14,AP_RND,AP_SAT,0> avgpool_default_t;
typedef ap_fixed<20,9,AP_RND,AP_SAT,0> layer13_t;
typedef ap_fixed<20,9> layer14_t;
typedef ap_fixed<8,1> weight14_t;
typedef ap_fixed<8,1> bias14_t;
typedef ap_uint<1> layer14_index;
typedef ap_ufixed<8,0,AP_RND_CONV,AP_SAT,0> layer16_t;
typedef ap_fixed<18,8> qActivation_rho1_table_t;
typedef ap_fixed<20,9> layer17_t;
typedef ap_fixed<8,1> weight17_t;
typedef ap_fixed<8,1> bias17_t;
typedef ap_uint<1> layer17_index;
typedef ap_ufixed<8,0,AP_RND_CONV,AP_SAT,0> layer19_t;
typedef ap_fixed<18,8> Activation_rho2_table_t;
typedef ap_fixed<20,9> layer20_t;
typedef ap_fixed<8,1> weight20_t;
typedef ap_fixed<8,1> bias20_t;
typedef ap_uint<1> layer20_index;
typedef ap_ufixed<8,0,AP_RND_CONV,AP_SAT,0> layer22_t;
typedef ap_fixed<18,8> qActivation_rho3_table_t;
typedef ap_fixed<20,9> layer23_t;
typedef ap_fixed<8,1> weight23_t;
typedef ap_fixed<8,1> bias23_t;
typedef ap_uint<1> layer23_index;
typedef ap_fixed<20,9,AP_RND,AP_SAT,0> result_t;
typedef ap_fixed<18,8> output_class_table_t;

#endif
