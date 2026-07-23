#include "ocr.h"

static t_image *get_next_image(const int fd_image, const int fd_label)
{
    t_image *result;
    result = malloc(sizeof(t_image));

    unsigned char buf[784];
    ssize_t n;
    ssize_t i = 0;

    i = 0;
    n = read(fd_image, buf, sizeof(buf));
    if (n < 0)
        return (NULL);
    while (i < 784)
    {
        result->data[i] = buf[i] / 255.0f;
        i++;
    }
    n = read(fd_label, buf, sizeof(unsigned char));
    if (n < 0)
        return (NULL);
    result->label = buf[0];
    return (result);
}

static int training_loop(t_fds *fds, t_sums *sums, t_network *network)
{
    t_grads grads;
    int i;
    int epoch = 0;
    int count;
    t_image *image;

    while (epoch < 20)
    {
        i = 0;
        count = 0;
        if (open_training_files(fds) == -1)
            printf("Error while opening the training files");
        image = get_next_image(fds->fd_images, fds->fd_labels);
        if (!image)
        {
            printf("Error: no training image");
            return (-1);
        }
        while (image && i < 60000)
        {
            forward_pass(sums, image, network);
            backward_pass(sums, &grads, image, network);
            if (image->label == sums->prediction_result)
                count++;
            free(image);
            image = get_next_image(fds->fd_images, fds->fd_labels);
            if (!image)
            {
                printf("Error: no training image");
                return (-1);
            }
            i++;
        }
        printf("Epoch %d : %d/60000 (%.2f)\n", epoch + 1, count, count/600.0);
        close_train_fds(fds);
        epoch++;
    }
    return (0);
}

int test_loop(t_fds *fds, t_sums *sums, t_network *network)
{
    int i = 0;
    int count = 0;
    t_image *image;

    if (open_test_files(fds) == -1)
        printf("Error while opening the test files");
    image = get_next_image(fds->fd_test_images, fds->fd_test_labels);
    if (!image)
    {
        printf("Error: no training image");
        return (-1);
    }
    while (image && i < 10000)
    {
        forward_pass(sums, image, network);
        if (image->label == sums->prediction_result)
            count++;
        free(image);
        image = get_next_image(fds->fd_test_images, fds->fd_test_labels);
        if (!image)
        {
            printf("Error: no training image");
            return (-1);
        }
        i++;
    }
    printf("TEST: %d/10000 (%.2f)\n", count, count/100.0);
    close_test_fds(fds);
    return (0);
}

int main()
{
    srand(time(NULL));
    
    t_fds fds;
    t_sums sums;
    t_network network;

    init_hidden_layer(&network.hidden);
    init_output_layer(&network.output);

    training_loop(&fds, &sums, &network);
    test_loop(&fds, &sums, &network);

    close_test_fds(&fds);

    return (0);
}