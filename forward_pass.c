#include "ocr.h"

//l for the index of the image line, c for column
void    forward_pass_hidden(t_sums *sum, t_image *image, t_network *network)
{
    int k = 0;
    int i;

    while (k < 128)
    {
        i = 0;
        while (i < 784)
        {
            sum->z1[k] += image->data[i]* network->hidden.weights[i][k];
            i++;
        }
        sum->z1[k] += network->hidden.biases[k];
        k++;
    }
    k = 0;
    while (k < 128)
    {
        sum->a1[k] = 0;
        if (sum->z1[k] > 0)
            sum->a1[k] = sum->z1[k];
        k++;
    }
}

void    forward_pass_output(t_sums *sum, t_network *network)
{
    int k = 0;
    int i;


    while (k < 10)
    {
        i = 0;
        while (i < 128)
        {
            sum->z2[k] += sum->a1[i] * network->output.weights[i][k];
            i++;
        }
        sum->z2[k] += network->output.biases[k];
        k++;
    }
}

//Softmax
//Standard practice is to subtract max(z2) to exp(z2[j])
void    softmax(t_sums *sum)
{
    float exp_sum = 0;
    float z_max = 0;
    int k = 0;

    while (k < 10)
    {
        if (z_max < sum->z2[k])
            z_max = sum->z2[k];
        k++;
    }
    k = 0;
    while (k < 10)
    {
        exp_sum += exp(sum->z2[k] - z_max);
        k++;
    }
    k = 0;
    while (k < 10)
    {
        sum->a2[k]= exp(sum->z2[k] - z_max) / exp_sum;
        k++;
    }
}

void make_prediction(t_sums *sums)
{
    int     result = 0;
    int     i = 0;

    sums->prob_max = 0;
    while (i < 10)
    {
        if (sums->prob_max < sums->a2[i])
        {
            result = i;
            sums->prob_max = sums->a2[i];
        }
        i++;
    }
    sums->prediction_result = result;
}

//In our case, we only calculate the loss on the right label
void  cross_entropy_loss(t_sums *sums, t_image *image)
{
    sums->loss = - log(sums->a2[image->label]);
}

void    forward_pass(t_sums *sums, t_image *image, t_network *network)
{
    init_sums(sums);
    forward_pass_hidden(sums, image, network);
    forward_pass_output(sums, network);
    softmax(sums);
    make_prediction(sums);
    cross_entropy_loss(sums, image);
}