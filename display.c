#include "ocr.h"

void    display_image(t_image *image)
{
    int i = 0;

    printf("LABEL = %d\n", image->label);
    printf("IMAGE\n");
    while (i < 784)
    {
        printf("%d ", (int)(image->data[i] * 255));
        if (i % 28 == 0)
            printf("\n");
        i++;
    }
    printf("\n");
}

void    display_output(t_output *layer)
{
    int i = 0;
    int j = 0;

    if (layer->nb_inputs == 128)
        printf("OUTPUT LAYER :\n");
    else
        printf("HIDDEN LAYER :\n");
    printf("NB_inputs = %d\n", layer->nb_inputs);
    printf("NB_outputs = %d\n", layer->nb_outputs);

    printf("WEIGHTS\n");
    while (i < layer->nb_inputs)
    {
        j = 0;
        while (j < layer->nb_outputs)
        {
            printf("%.4f ", layer->weights[i][j]);
            j++;
        }
        i++;
    }
    printf("\nBIASES\n");
    i = 0;
        while (i < layer->nb_outputs)
    {
        printf("%.4f ", layer->biases[i]);
        i++;
    }
    printf("\nWEIGHTS_GRADS\n");
    while (i < layer->nb_inputs)
    {
        j = 0;
        while (j < layer->nb_outputs)
        {
            printf("%.4f ", layer->weights_grad[i][j]);
            j++;
        }
        i++;
    }
    printf("\nBIASES_GRADS\n");
    i = 0;
    while (i < layer->nb_outputs)
    {
        printf("%.4f ", layer->biases_grad[i]);
        i++;
    }
    printf("\n");
}

void    display_hidden(t_hidden *layer)
{
    int i = 0;
    int j = 0;

    if (layer->nb_inputs == 128)
        printf("OUTPUT LAYER :\n");
    else
        printf("HIDDEN LAYER :\n");
    printf("NB_inputs = %d\n", layer->nb_inputs);
    printf("NB_outputs = %d\n", layer->nb_outputs);

    printf("WEIGHTS\n");
    while (i < layer->nb_inputs)
    {
        j = 0;
        while (j < layer->nb_outputs)
        {
            printf("%.4f ", layer->weights[i][j]);
            j++;
        }
        i++;
    }
    printf("\nBIASES\n");
    i = 0;
        while (i < layer->nb_outputs)
    {
        printf("%.4f ", layer->biases[i]);
        i++;
    }
    printf("\nWEIGHTS_GRADS\n");
    while (i < layer->nb_inputs)
    {
        j = 0;
        while (j < layer->nb_outputs)
        {
            printf("%.4f ", layer->weights_grad[i][j]);
            j++;
        }
        i++;
    }
    printf("\nBIASES_GRADS\n");
    i = 0;
    while (i < layer->nb_outputs)
    {
        printf("%.4f ", layer->biases_grad[i]);
        i++;
    }
    printf("\n");
}

void    display_sums(t_sums *sums)
{
    int k = 0;

    printf("A1\n");
    while (k < 128)
    {
        printf("[%f]", sums->a1[k]);
        k++;
    }
    printf("\n");
 
    printf("Z2\n");
    k = 0;
    while (k < 128)
    {
        printf("[%f]", sums->z2[k]);
        k++;
    }
    printf("\n");

    printf("A2\n");
    k = 0;
    while (k < 10)
    {
        printf("[%f]", sums->a2[k]);
        k++;
    }
    printf("\n");
    k = 0;
    float prob = 0;
    while (k < 10)
    {
        prob += sums->a2[k];
        k++;
    }
    printf("total probability: %f\n", prob);
}
