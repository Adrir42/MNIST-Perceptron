#include "ocr.h"

/*
This is an initialisation of the weights of the hidden layer, the He initialisation, ideal for ReLU
(float)rand() / RAND_MAX gives a random nuber between 0 and 1
- 0.5 centers this number around 0, the range at this moment is [-0.5, 0.5]
* sqrt(2 / hidden->nb_inputs) gives us a random number in the range [-0.0505, 0.0505]
*/
float   get_random_weight_He(t_hidden *hidden)
{
    unsigned int j = rand();
    float i = (((float)j / RAND_MAX) - 0.5) * sqrt(2.0 / hidden->nb_inputs);
    return (i);
}

/*
This is an initialisation of the weights of the output layer, the Xavier/Glorot initialisation, ideal for softmax
(float)rand() / RAND_MAX gives a random nuber between 0 and 1
- 0.5 centers this number around 0, the range at this moment is [-0.5, 0.5]
* sqrt(2 / output->nb_inputs + output->nb_outputs) gives us a random number in the range [-0.088, 0.088]
*/
float   get_random_weight_Xavier_Glorot(t_output *output)
{
    unsigned int j = rand();
    float i = (((float)j / RAND_MAX) - 0.5) * sqrt(2.0 / (output->nb_inputs + output->nb_outputs));
    return (i);
}

void init_hidden_layer(t_hidden *hidden)
{
    int i = 0;
    int j = 0;

    hidden->nb_inputs = 784;
    hidden->nb_outputs = 128;

    while (i < hidden->nb_inputs)
    {
        j = 0;
        while(j < hidden->nb_outputs)
        {
            hidden->weights[i][j] = get_random_weight_He(hidden);
            hidden->weights_grad[i][j] = 0;
            j++;
        }
        i++;
    }
    i = 0;
    while (i < hidden->nb_outputs)
    {
        hidden->biases[i] = 0;
        hidden->biases_grad[i] = 0;
        i++;
    }
}

void init_output_layer(t_output *output)
{
    int i = 0;
    int j = 0;

    output->nb_inputs = 128;
    output->nb_outputs = 10;

    while (i < output->nb_inputs)
    {
        j = 0;
        while(j < output->nb_outputs)
        {
            output->weights[i][j] = get_random_weight_Xavier_Glorot(output);
            output->weights_grad[i][j] = 0;
            j++;
        }
        i++;
    }
    i = 0;
    while (i < output->nb_outputs)
    {
        output->biases[i] = 0;
        output->biases_grad[i] = 0;
        i++;
    }    
}

void    init_sums(t_sums *sums)
{
    int i = 0;

    while(i < 128)
    {
        sums->z1[i] = 0;
        i++;
    }

    i = 0;
    while(i < 10)
    {
        sums->z2[i] = 0;
        i++;
    }
}