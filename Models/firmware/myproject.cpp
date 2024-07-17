#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    input_t inputs[N_INPUT_1_1*N_INPUT_2_1],
    result_t layer25_out[N_LAYER_23]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=inputs complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer25_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=inputs,layer25_out 
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<batchnorm_default_t, 12>(s2, "s2.txt");
        nnet::load_weights_from_txt<batchnorm_default_t, 12>(b2, "b2.txt");
        nnet::load_weights_from_txt<dense_phi1_weight_t, 192>(w26, "w26.txt");
        nnet::load_weights_from_txt<dense_phi1_bias_t, 16>(b26, "b26.txt");
        nnet::load_weights_from_txt<dense_phi2_weight_t, 256>(w27, "w27.txt");
        nnet::load_weights_from_txt<dense_phi2_bias_t, 16>(b27, "b27.txt");
        nnet::load_weights_from_txt<dense_phi3_weight_t, 256>(w28, "w28.txt");
        nnet::load_weights_from_txt<dense_phi3_bias_t, 16>(b28, "b28.txt");
        nnet::load_weights_from_txt<weight14_t, 256>(w14, "w14.txt");
        nnet::load_weights_from_txt<bias14_t, 16>(b14, "b14.txt");
        nnet::load_weights_from_txt<weight17_t, 512>(w17, "w17.txt");
        nnet::load_weights_from_txt<bias17_t, 32>(b17, "b17.txt");
        nnet::load_weights_from_txt<weight20_t, 512>(w20, "w20.txt");
        nnet::load_weights_from_txt<bias20_t, 16>(b20, "b20.txt");
        nnet::load_weights_from_txt<weight23_t, 16>(w23, "w23.txt");
        nnet::load_weights_from_txt<bias23_t, 1>(b23, "b23.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer2_t layer2_out[N_INPUT_1_1*N_INPUT_2_1];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::normalize<input_t, layer2_t, config2>(inputs, layer2_out, s2, b2); // batchnorm

    layer26_t layer26_out[N_OUTPUTS_26*N_FILT_26];
    #pragma HLS ARRAY_PARTITION variable=layer26_out complete dim=0
    nnet::pointwise_conv_1d_cl<layer2_t, layer26_t, config26>(layer2_out, layer26_out, w26, b26); // Dense_phi1

    layer5_t layer5_out[N_LAYER_1_3*N_LAYER_2_3];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::relu<layer26_t, layer5_t, relu_config5>(layer26_out, layer5_out); // Activation_phi1

    layer27_t layer27_out[N_OUTPUTS_27*N_FILT_27];
    #pragma HLS ARRAY_PARTITION variable=layer27_out complete dim=0
    nnet::pointwise_conv_1d_cl<layer5_t, layer27_t, config27>(layer5_out, layer27_out, w27, b27); // Dense_phi2

    layer8_t layer8_out[N_LAYER_1_6*N_LAYER_2_6];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    nnet::relu<layer27_t, layer8_t, relu_config8>(layer27_out, layer8_out); // Activation_phi2

    layer28_t layer28_out[N_OUTPUTS_28*N_FILT_28];
    #pragma HLS ARRAY_PARTITION variable=layer28_out complete dim=0
    nnet::pointwise_conv_1d_cl<layer8_t, layer28_t, config28>(layer8_out, layer28_out, w28, b28); // Dense_phi3

    layer11_t layer11_out[N_LAYER_1_9*N_LAYER_2_9];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    nnet::relu<layer28_t, layer11_t, relu_config11>(layer28_out, layer11_out); // Activation_phi3

    layer12_t layer12_out[N_LAYER_1_9*N_LAYER_2_9];
    #pragma HLS ARRAY_PARTITION variable=layer12_out complete dim=0
    nnet::linear<layer11_t, layer12_t, linear_config12>(layer11_out, layer12_out); // qActivationForPool

    layer13_t layer13_out[N_FILT_13];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0
    nnet::global_pooling1d_cl<layer12_t, layer13_t, config13>(layer12_out, layer13_out); // avgpool

    layer14_t layer14_out[N_LAYER_14];
    #pragma HLS ARRAY_PARTITION variable=layer14_out complete dim=0
    nnet::dense<layer13_t, layer14_t, config14>(layer13_out, layer14_out, w14, b14); // qDense_rho1

    layer16_t layer16_out[N_LAYER_14];
    #pragma HLS ARRAY_PARTITION variable=layer16_out complete dim=0
    nnet::relu<layer14_t, layer16_t, relu_config16>(layer14_out, layer16_out); // qActivation_rho1

    layer17_t layer17_out[N_LAYER_17];
    #pragma HLS ARRAY_PARTITION variable=layer17_out complete dim=0
    nnet::dense<layer16_t, layer17_t, config17>(layer16_out, layer17_out, w17, b17); // Dense_rho2

    layer19_t layer19_out[N_LAYER_17];
    #pragma HLS ARRAY_PARTITION variable=layer19_out complete dim=0
    nnet::relu<layer17_t, layer19_t, relu_config19>(layer17_out, layer19_out); // Activation_rho2

    layer20_t layer20_out[N_LAYER_20];
    #pragma HLS ARRAY_PARTITION variable=layer20_out complete dim=0
    nnet::dense<layer19_t, layer20_t, config20>(layer19_out, layer20_out, w20, b20); // qDense_rho3

    layer22_t layer22_out[N_LAYER_20];
    #pragma HLS ARRAY_PARTITION variable=layer22_out complete dim=0
    nnet::relu<layer20_t, layer22_t, relu_config22>(layer20_out, layer22_out); // qActivation_rho3

    layer23_t layer23_out[N_LAYER_23];
    #pragma HLS ARRAY_PARTITION variable=layer23_out complete dim=0
    nnet::dense<layer22_t, layer23_t, config23>(layer22_out, layer23_out, w23, b23); // qDense_rho4

    nnet::sigmoid<layer23_t, result_t, sigmoid_config25>(layer23_out, layer25_out); // output_class

}
