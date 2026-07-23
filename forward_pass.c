#include "ocr.h"

void    forward_pass_hidden(t_sums *sum, const t_image *image, const t_network *network)
{
    int hidden_neuron_index = 0;
    int pixel_index;

    while (hidden_neuron_index < NB_HIDDEN_NEURONS)
    {
        pixel_index = 0;
        while (pixel_index < 784)
        {
            sum->z1[hidden_neuron_index] += image->data[pixel_index]* network->hidden.weights[pixel_index][hidden_neuron_index];
            pixel_index++;
        }
        sum->z1[hidden_neuron_index] += network->hidden.biases[hidden_neuron_index];
        hidden_neuron_index++;
    }
    hidden_neuron_index = 0;
    while (hidden_neuron_index < NB_HIDDEN_NEURONS)
    {
        sum->a1[hidden_neuron_index] = 0;
        if (sum->z1[hidden_neuron_index] > 0)
            sum->a1[hidden_neuron_index] = sum->z1[hidden_neuron_index];
        hidden_neuron_index++;
    }
}

void    forward_pass_output(t_sums *sum, const t_network *network)
{
    int output_neuron_index = 0;
    int hidden_neuron_index;


    while (output_neuron_index < NB_OUTPUT_NEURONS)
    {
        hidden_neuron_index = 0;
        while (hidden_neuron_index < NB_HIDDEN_NEURONS)
        {
            sum->z2[output_neuron_index] += sum->a1[hidden_neuron_index] * network->output.weights[hidden_neuron_index][output_neuron_index];
            hidden_neuron_index++;
        }
        sum->z2[output_neuron_index] += network->output.biases[output_neuron_index];
        output_neuron_index++;
    }
}

//Softmax
//Standard practice is to subtract max(z2) to exp(z2[j])
void    softmax(t_sums *sum)
{
    float exp_sum = 0;
    float z_max = 0;
    int output_neuron_index = 0;

    while (output_neuron_index < NB_OUTPUT_NEURONS)
    {
        if (z_max < sum->z2[output_neuron_index])
            z_max = sum->z2[output_neuron_index];
        output_neuron_index++;
    }
    output_neuron_index = 0;
    while (output_neuron_index < NB_OUTPUT_NEURONS)
    {
        exp_sum += exp(sum->z2[output_neuron_index] - z_max);
        output_neuron_index++;
    }
    output_neuron_index = 0;
    while (output_neuron_index < NB_OUTPUT_NEURONS)
    {
        sum->a2[output_neuron_index]= exp(sum->z2[output_neuron_index] - z_max) / exp_sum;
        output_neuron_index++;
    }
}

//Make a prediction of the input number based on the highest calculated probability 
void make_prediction(t_sums *sums)
{
    int     result = 0;
    int     output_neuron_index = 0;

    sums->prob_max = 0;
    while (output_neuron_index < NB_OUTPUT_NEURONS)
    {
        if (sums->prob_max < sums->a2[output_neuron_index])
        {
            result = output_neuron_index;
            sums->prob_max = sums->a2[output_neuron_index];
        }
        output_neuron_index++;
    }
    sums->prediction_result = result;
}

//In our case, we only calculate the loss on the right label
void  cross_entropy_loss(t_sums *sums, const t_image *image)
{
    sums->loss = - log(sums->a2[image->label]);
}

void    forward_pass(t_sums *sums, const t_image *image, const t_network *network)
{
    init_sums(sums);
    forward_pass_hidden(sums, image, network);
    forward_pass_output(sums, network);
    softmax(sums);
    make_prediction(sums);
    cross_entropy_loss(sums, image);
}