#include "ocr.h"

/*
This is an initialisation of the weights of the hidden layer, the He initialisation, ideal for ReLU
(float)rand() / RAND_MAX gives a random nuber between 0 and 1
- 0.5 centers this number around 0, the range at this moment is [-0.5, 0.5]
* sqrt(2 / hidden->nb_inputs) gives us a random number in the range [-0.0505, 0.0505]
*/
float   get_random_weight_He()
{
    unsigned int random_int = rand();
    float weight = (((float) random_int / RAND_MAX) - 0.5) * sqrt(2.0 / IMAGE_SIZE);
    return (weight);
}

/*
This is an initialisation of the weights of the output layer, the Xavier/Glorot initialisation, ideal for softmax
(float)rand() / RAND_MAX gives a random nuber between 0 and 1
- 0.5 centers this number around 0, the range at this moment is [-0.5, 0.5]
* sqrt(2 / output->nb_inputs + output->nb_outputs) gives us a random number in the range [-0.088, 0.088]
*/
float   get_random_weight_Xavier_Glorot()
{
    unsigned int random_int = rand();
    float weight = (((float) random_int / RAND_MAX) - 0.5) * sqrt(2.0 / (NB_HIDDEN_NEURONS + NB_OUTPUT_NEURONS));
    return (weight);
}

void init_hidden_layer(t_hidden *hidden)
{
    int pixel_index = 0;
    int neuron_index = 0;

    while (pixel_index < IMAGE_SIZE)
    {
        neuron_index = 0;
        while(neuron_index < NB_HIDDEN_NEURONS)
        {
            hidden->weights[pixel_index][neuron_index] = get_random_weight_He(hidden);
            hidden->weights_grad[pixel_index][neuron_index] = 0;
            neuron_index++;
        }
        pixel_index++;
    }
    neuron_index = 0;
    while (neuron_index < NB_HIDDEN_NEURONS)
    {
        hidden->biases[neuron_index] = 0;
        hidden->biases_grad[neuron_index] = 0;
        neuron_index++;
    }
}

void init_output_layer(t_output *output)
{
    int hidden_neuron_index = 0;
    int output_neuron_index = 0;

    while (hidden_neuron_index < NB_HIDDEN_NEURONS)
    {
        output_neuron_index = 0;
        while(output_neuron_index < NB_OUTPUT_NEURONS)
        {
            output->weights[hidden_neuron_index][output_neuron_index] = get_random_weight_Xavier_Glorot(output);
            output->weights_grad[hidden_neuron_index][output_neuron_index] = 0;
            output_neuron_index++;
        }
        hidden_neuron_index++;
    }
    output_neuron_index = 0;
    while (output_neuron_index < NB_OUTPUT_NEURONS)
    {
        output->biases[output_neuron_index] = 0;
        output->biases_grad[output_neuron_index] = 0;
        output_neuron_index++;
    }
}

void    init_sums(t_sums *sums)
{
    int neuron_index = 0;

    while(neuron_index < NB_HIDDEN_NEURONS)
    {
        sums->z1[neuron_index] = 0;
        neuron_index++;
    }

    neuron_index = 0;
    while(neuron_index < NB_OUTPUT_NEURONS)
    {
        sums->z2[neuron_index] = 0;
        neuron_index++;
    }
}