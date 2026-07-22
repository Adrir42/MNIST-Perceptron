#include "ocr.h"

void    init_grads(t_grads *grads, t_network *network)
{
    int i = 0;
    int k = 0;

    while (i < 128)
    {
        grads->z1[i] = 0;
        grads->a1[i] = 0;
        k = 0;
        while(k < 784)
        {
            network->hidden.weights_grad[k][i] = 0;
            k++;
        }
        network->hidden.biases_grad[i] = 0;
        i++;
    }
    i = 0;
    while (i < 10)
    {
        k = 0;
        while(k < 128)
        {
            network->output.weights_grad[k][i] = 0;
            k++;
        }
        network->output.biases_grad[i] = 0;
        grads->z2[i] = 0;
        i++;
    }
}

//grads->z2[k] = output->biases_grad donc possibilité d'utiliser moins de mémoire
void    backward_pass_output(t_sums *sums, t_grads *grads, t_image *image, t_output *output)
{
    int k = 0;
    int i;

    while (k < 10)
    {
        grads->z2[k] = sums->a2[k];
        if (k == image->label)
            grads->z2[k]--;
        output->biases_grad[k] = grads->z2[k];
        k++;
    }

    i = 0;
    while (i < 128)
    {
        k = 0;
        while (k < 10)
        {
            output->weights_grad[i][k] = sums->a1[i] * grads->z2[k];
            k++;
        }
        i++;
    }
}

void    backward_pass_hidden(t_sums *sums, t_grads *grads, t_image *image, t_network *network)
{
    int k = 0;
    int i;

    i = 0;
    while (i < 128)
    {
        k = 0;
        while (k < 10)
        {
            grads->a1[i] += grads->z2[k] * network->output.weights[i][k];
            k++;
        }
        if (sums->z1[i] > 0)
            grads->z1[i] = grads->a1[i];
        i++;
    }
    i = 0;
    while (i < 784)
    {
        k = 0;
        while (k < 128)
        {
            network->hidden.weights_grad[i][k] = image->data[i] * grads->z1[k];
            k++;
        }
        i++;
    }
    k = 0;
    while (k < 128)
    {
        network->hidden.biases_grad[k] = grads->z1[k];
        k++;
    }
}

void    update_weights_and_biases(t_network *network)
{
    int i = 0;
    int k;

    while (i < 784)
    {
        k = 0;
        while (k < 128)
        {
            network->hidden.weights[i][k] -= LEARNING_RATE * network->hidden.weights_grad[i][k];
            k++;
        }
        i++;
    }
    k = 0;
    while (k < 128)
    {
        network->hidden.biases[k] -= LEARNING_RATE * network->hidden.biases_grad[k];
        k++;
    }
    i = 0;
    while (i < 128)
    {
        k = 0;
        while (k < 10)
        {
            network->output.weights[i][k] -= LEARNING_RATE * network->output.weights_grad[i][k];
            k++;
        }
        i++;
    }
    k = 0;
    while (k < 10)
    {
        network->output.biases[k] -= LEARNING_RATE * network->output.biases_grad[k];
        k++;
    }
}

void    backward_pass(t_sums *sums, t_grads *grads, t_image *image, t_network *network)
{
    init_grads(grads, network);
    backward_pass_output(sums, grads, image, &(network->output));
    backward_pass_hidden(sums, grads, image, network);
    update_weights_and_biases(network);
}