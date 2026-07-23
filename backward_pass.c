#include "ocr.h"

void    init_grads(t_grads *grads, t_network *network)
{
    int output_index = 0;
    int input_index = 0;

    while (output_index < NB_HIDDEN_NEURONS)
    {
        grads->z1[output_index] = 0;
        grads->a1[output_index] = 0;
        input_index = 0;
        while(input_index < 784)
        {
            network->hidden.weights_grad[input_index][output_index] = 0;
            input_index++;
        }
        network->hidden.biases_grad[output_index] = 0;
        output_index++;
    }
    output_index = 0;
    while (output_index < 10)
    {
        input_index = 0;
        while(input_index < 128)
        {
            network->output.weights_grad[input_index][output_index] = 0;
            input_index++;
        }
        network->output.biases_grad[output_index] = 0;
        grads->z2[output_index] = 0;
        output_index++;
    }
}

//grads->z2[k] = output->biases_grad donc possibilité d'utiliser moins de mémoire
void    backward_pass_output(const t_sums *sums, t_grads *grads, const t_image *image, t_output *output)
{
    int output_neuron_index = 0;
    int hidden_neuron_index;

    while (output_neuron_index < NB_OUTPUT_NEURONS)
    {
        grads->z2[output_neuron_index] = sums->a2[output_neuron_index];
        if (output_neuron_index == image->label)
            grads->z2[output_neuron_index]--;
        output->biases_grad[output_neuron_index] = grads->z2[output_neuron_index];
        output_neuron_index++;
    }

    hidden_neuron_index = 0;
    while (hidden_neuron_index < NB_HIDDEN_NEURONS)
    {
        output_neuron_index = 0;
        while (output_neuron_index < 10)
        {
            output->weights_grad[hidden_neuron_index][output_neuron_index] = sums->a1[hidden_neuron_index] * grads->z2[output_neuron_index];
            output_neuron_index++;
        }
        hidden_neuron_index++;
    }
}

void    backward_pass_hidden(const t_sums *sums, t_grads *grads, const t_image *image, t_network *network)
{
    int hidden_neuron_index;
    int output_neuron_index = 0;
    int pixel_index = 0;


    hidden_neuron_index = 0;
    while (hidden_neuron_index < NB_HIDDEN_NEURONS)
    {
        output_neuron_index = 0;
        while (output_neuron_index < NB_OUTPUT_NEURONS)
        {
            grads->a1[hidden_neuron_index] += grads->z2[output_neuron_index] * network->output.weights[hidden_neuron_index][output_neuron_index];
            output_neuron_index++;
        }
        if (sums->z1[hidden_neuron_index] > 0)
            grads->z1[hidden_neuron_index] = grads->a1[hidden_neuron_index];
        hidden_neuron_index++;
    }
    pixel_index = 0;
    while (pixel_index < IMAGE_SIZE)
    {
        hidden_neuron_index = 0;
        while (hidden_neuron_index < NB_HIDDEN_NEURONS)
        {
            network->hidden.weights_grad[pixel_index][hidden_neuron_index] = image->data[pixel_index] * grads->z1[hidden_neuron_index];
            hidden_neuron_index++;
        }
        pixel_index++;
    }
    hidden_neuron_index = 0;
    while (hidden_neuron_index < NB_HIDDEN_NEURONS)
    {
        network->hidden.biases_grad[hidden_neuron_index] = grads->z1[hidden_neuron_index];
        hidden_neuron_index++;
    }
}

void    update_weights_and_biases(t_network *network)
{
    int pixel_index = 0;
    int hidden_neuron_index;
    int output_neuron_index = 0;

    while (pixel_index < IMAGE_SIZE)
    {
        hidden_neuron_index = 0;
        while (hidden_neuron_index < NB_HIDDEN_NEURONS)
        {
            network->hidden.weights[pixel_index][hidden_neuron_index] -= LEARNING_RATE * network->hidden.weights_grad[pixel_index][hidden_neuron_index];
            hidden_neuron_index++;
        }
        pixel_index++;
    }
    hidden_neuron_index = 0;
    while (hidden_neuron_index < NB_HIDDEN_NEURONS)
    {
        network->hidden.biases[hidden_neuron_index] -= LEARNING_RATE * network->hidden.biases_grad[hidden_neuron_index];
        hidden_neuron_index++;
    }
    hidden_neuron_index = 0;
    while (hidden_neuron_index < NB_HIDDEN_NEURONS)
    {
        output_neuron_index = 0;
        while (output_neuron_index < NB_OUTPUT_NEURONS)
        {
            network->output.weights[hidden_neuron_index][output_neuron_index] -= LEARNING_RATE * network->output.weights_grad[hidden_neuron_index][output_neuron_index];
            output_neuron_index++;
        }
        hidden_neuron_index++;
    }
    output_neuron_index = 0;
    while (output_neuron_index < NB_OUTPUT_NEURONS)
    {
        network->output.biases[output_neuron_index] -= LEARNING_RATE * network->output.biases_grad[output_neuron_index];
        output_neuron_index++;
    }
}

void    backward_pass(const t_sums *sums, t_grads *grads, const t_image *image, t_network *network)
{
    init_grads(grads, network);
    backward_pass_output(sums, grads, image, &(network->output));
    backward_pass_hidden(sums, grads, image, network);
    update_weights_and_biases(network);
}