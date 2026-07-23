#include "ocr.h"

static t_image *get_next_image(const int fd_image, const int fd_label)
{
    unsigned char buf[IMAGE_SIZE];
    ssize_t read_check;
    ssize_t pixel_index = 0;
    t_image *result;
    result = malloc(sizeof(t_image));
    if (!result)
        return (NULL);
    pixel_index = 0;
    read_check = read(fd_image, buf, sizeof(buf));
    if (read_check != IMAGE_SIZE)
    {
        free(result);
        return (NULL);
    }
    while (pixel_index < IMAGE_SIZE)
    {
        result->data[pixel_index] = buf[pixel_index] / 255.0f;
        pixel_index++;
    }
    read_check = read(fd_label, buf, sizeof(unsigned char));
    if (read_check != 1)
    {
        free(result);
        return (NULL);
    }
    result->label = buf[0];
    return (result);
}

static int training_loop(t_fds *fds, t_sums *sums, t_network *network)
{
    t_grads grads;
    int image_index;
    int epoch = 0;
    int count;
    t_image *image;

    while (epoch < NB_EPOCHS)
    {
        image_index = 0;
        count = 0;
        if (open_training_files(fds) == -1)
            fprintf(stderr, "Error while opening the training files\n");
        image = get_next_image(fds->fd_images, fds->fd_labels);
        if (!image)
        {
            fprintf(stderr, "Error while getting training image\n");
            return (-1);
        }
        while (image && image_index < NB_IMAGES_TRAINING_SET)
        {
            forward_pass(sums, image, network);
            backward_pass(sums, &grads, image, network);
            if (image->label == sums->prediction_result)
                count++;
            free(image);
            image = get_next_image(fds->fd_images, fds->fd_labels);
            image_index++;
        }
        printf("Epoch %d : %d/60000 (%.2f)\n", epoch + 1, count, count/600.0);
        close_train_fds(fds);
        epoch++;
    }
    return (0);
}

int test_loop(t_fds *fds, t_sums *sums, t_network *network)
{
    int image_index = 0;
    int count = 0;
    t_image *image;

    if (open_test_files(fds) == -1)
        fprintf(stderr, "Error while opening the test files\n");
    image = get_next_image(fds->fd_test_images, fds->fd_test_labels);
    if (!image)
    {
        fprintf(stderr, "Error while getting test image\n");
        return (-1);
    }
    while (image && image_index < NB_IMAGES_TEST_SET)
    {
        forward_pass(sums, image, network);
        if (image->label == sums->prediction_result)
            count++;
        free(image);
        image = get_next_image(fds->fd_test_images, fds->fd_test_labels);
        image_index++;
    }
    printf("TEST: %d/10000 (%.2f)\n", count, count/100.0);
    close_test_fds(fds);
    return (0);
}

int main( void )
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