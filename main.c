#include "ocr.h"

//784 = 28*28
t_image *get_next_training_image(t_fds *fds)
{
    t_image *result;
    result = malloc(sizeof(t_image));

    unsigned char buf[784];
    ssize_t n;
    ssize_t i = 0;

    i = 0;
    n = read(fds->fd_images, buf, sizeof(buf));
    if (n < 0)
        return (NULL);
    while (i < 784)
    {
        result->data[i] = buf[i] / 255.0f;
        i++;
    }
    n = read(fds->fd_labels, buf, sizeof(unsigned char));
    if (n < 0)
        return (NULL);
    result->label = buf[0];
    return (result);
}

t_image *get_next_test_image(t_fds *fds)
{
    t_image *result;
    result = malloc(sizeof(t_image));

    unsigned char buf[784];
    ssize_t n;
    ssize_t i = 0;

    i = 0;
    n = read(fds->fd_test_images, buf, sizeof(buf));
    if (n < 0)
        return (NULL);
    while (i < 784)
    {
        result->data[i] = buf[i] / 255.0f;
        i++;
    }
    n = read(fds->fd_test_labels, buf, sizeof(unsigned char));
    if (n < 0)
        return (NULL);
    result->label = buf[0];
    return (result);
}

int main()
{
    srand(time(NULL));
    
    t_fds fds;


    t_sums sums;
    t_grads grads;
    t_image *image;
    t_network network;

    init_hidden_layer(&network.hidden);
    init_output_layer(&network.output);

    int i;
    int epoch = 0;
    int count;
    float count_purcents = 0;

    while (epoch < 20)
    {
        open_files(&fds);
        i = 0;
        count = 0;
        image = get_next_training_image(&fds);
        while (image && i < 60000)
        {
            forward_pass(&sums, image, &network);

//        printf("i: %d\nlabel: %d\nresult: %d\nloss: %f\naccuracy: %f\n\n", i, image->label, sums.prediction_result, sums.loss, sums.prob_max * 100);

            backward_pass(&sums, &grads, image, &network);
            if (image->label == sums.prediction_result)
                count++;
            free(image);
            image = get_next_training_image(&fds);
            i++;
        }
        count_purcents = count/600.0;
        printf("Epoch %d : %d/60000 (%.2f)\n", epoch + 1, count, count_purcents);
        close_train_fds(&fds);
        epoch++;
    }

    //Test
    i = 0;
    count = 0;
    image = get_next_test_image(&fds);
    while (image && i < 10000)
    {
        forward_pass(&sums, image, &network);
        if (image->label == sums.prediction_result)
            count++;
        free(image);
        image = get_next_test_image(&fds);
        i++;
    }
    count_purcents = count/100.0;
    printf("TEST: %d/10000 (%.2f)\n", count, count_purcents);
    close_test_fds(&fds);

    return (0);
}